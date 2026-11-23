"""Pack a frozen-feature dataset and run Stage 1 on an Alibaba PAI DLC worker."""

from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT = Path(__file__).resolve().parents[1]
FEATURE_NAME = re.compile(r"[0-9]{7}\.npz")


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_files(cache):
    files = sorted(cache.glob("*.npz"))
    provenance = cache / "source.json"
    if not files or not provenance.is_file():
        raise ValueError(f"Expected frame .npz files and source.json in {cache}")
    if any(not FEATURE_NAME.fullmatch(file.name) or not file.is_file() for file in files):
        raise ValueError(f"Invalid feature file name or type in {cache}")
    metadata = json.loads(provenance.read_text(encoding="utf-8"))
    if metadata.get("frames") != len(files):
        raise ValueError(f"source.json declares {metadata.get('frames')} frames but found {len(files)}")
    return [provenance, *files]


def pack(args):
    sources = cache_files(args.cache)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with NamedTemporaryFile(dir=output.parent, prefix=".rlt-feature-pack-") as scratch:
        with ZipFile(scratch, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
            for source in sources:
                archive.write(source, arcname=source.name)
        scratch.flush()
        os.link(scratch.name, output)
    data = {
        "archive": str(output),
        "frames": len(sources) - 1,
        "bytes": output.stat().st_size,
        "sha256": checksum(output),
    }
    print(json.dumps(data), flush=True)


def extract_cache(archive_path, cache, expected_sha256):
    if expected_sha256 and checksum(archive_path) != expected_sha256.lower():
        raise ValueError(f"Feature archive SHA-256 mismatch: {archive_path}")
    with ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = {member.filename for member in members}
        if (
            len(members) != len(names)
            or "source.json" not in names
            or any(name != "source.json" and not FEATURE_NAME.fullmatch(name) for name in names)
        ):
            raise ValueError("Expected an archive containing only source.json and flat numbered feature files")
        if shutil.disk_usage(cache).free < sum(member.file_size for member in members):
            raise OSError(f"Insufficient space to stage the feature cache under {cache}")
        for member in members:
            with archive.open(member) as source, (cache / member.filename).open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1 << 20)
    cache_files(cache)


def run(args):
    import torch

    if not (PROJECT / "rlt/cli.py").is_file():
        raise FileNotFoundError(f"Stage the RLT source next to the launcher below {PROJECT}")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 requires a CUDA-enabled PyTorch worker")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_name(f"{output.name}.log")
    scratch_context = (
        TemporaryDirectory(prefix="rlt-features-", dir=args.scratch) if args.archive else nullcontext(None)
    )
    with scratch_context as features_dir:
        cache = Path(features_dir) if features_dir else args.cache.resolve()
        if args.archive:
            extract_cache(args.archive, cache, args.sha256)
        files = cache_files(cache)
        with TemporaryDirectory(prefix=".rlt-stage1-", dir=output.parent) as staging:
            checkpoint_dir = Path(staging) / "checkpoint"
            command = [
                sys.executable, "-u", "-m", "rlt.cli", "--torch-threads", str(args.torch_threads),
                "train-token", "--cache", str(cache), "--out", str(checkpoint_dir),
                "--device", "cuda:0", "--steps", str(args.steps), "--batch-size", str(args.batch_size),
                "--token-dim", str(args.token_dim), "--lr", str(args.lr), "--seed", str(args.seed),
            ]
            metadata = {
                "mode": args.command,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "steps": args.steps,
                "batch_size": args.batch_size,
                "token_dim": args.token_dim,
                "lr": args.lr,
                "seed": args.seed,
                "frames": len(files) - 1,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "archive_sha256": args.sha256,
            }
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                line = json.dumps(metadata, allow_nan=False)
                print(line, flush=True)
                log.write(line + "\n")
                with subprocess.Popen(
                    command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                ) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    if return_code := process.wait():
                        raise subprocess.CalledProcessError(return_code, command)
            checkpoint = checkpoint_dir / "token.pt"
            if not checkpoint.is_file() or not (checkpoint_dir / "source.json").is_file():
                raise RuntimeError(f"Training did not produce a complete checkpoint; see {log_path}")
            metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["checkpoint_sha256"] = checksum(checkpoint)
            (checkpoint_dir / "run.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
            os.rename(checkpoint_dir, output)
            print(json.dumps({
                "checkpoint": str(output / "token.pt"),
                "sha256": metadata["checkpoint_sha256"],
                "log": str(log_path),
            }), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack", help="Pack the local feature cache for transfer")
    packing.add_argument("--cache", required=True, type=Path)
    packing.add_argument("--output", required=True, type=Path)
    packing.set_defaults(run=pack)

    for name, steps in (("smoke", 8), ("train", 2000)):
        training = commands.add_parser(name, help=f"Run Stage 1 on the DLC GPU ({steps} steps by default)")
        source = training.add_mutually_exclusive_group(required=True)
        source.add_argument("--cache", type=Path, help="Directory containing an extracted feature cache")
        source.add_argument("--archive", type=Path, help="Feature ZIP produced by the pack command")
        training.add_argument("--sha256", help="SHA-256 of the supplied feature archive")
        training.add_argument("--scratch", type=Path, help="Temporary extraction directory's parent")
        training.add_argument("--output", required=True, type=Path, help="New directory on durable storage")
        training.add_argument("--steps", type=int, default=steps)
        training.add_argument("--batch-size", type=int, default=2)
        training.add_argument("--token-dim", type=int, default=2048)
        training.add_argument("--lr", type=float, default=3e-4)
        training.add_argument("--seed", type=int, default=0)
        training.add_argument("--torch-threads", type=int, default=4)
        training.set_defaults(run=run)

    args = parser.parse_args()
    if args.command != "pack" and args.sha256 and not args.archive:
        parser.error("--sha256 is only valid with --archive")
    args.run(args)


if __name__ == "__main__":
    main()

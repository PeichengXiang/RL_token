"""Build a task's frozen π0.5 feature cache from Tianji training data on Alibaba PAI."""

from argparse import ArgumentParser
from importlib.util import module_from_spec, spec_from_file_location
from itertools import groupby
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace


PI05_ROOT = Path("/mnt/workspace/pi05_tianji")


def insertion_prompt(module_count):
    module = "module" if module_count == 1 else "modules"
    return (
        f"Insert {module_count} optical {module} into the matching slots. "
        "Wait until each slot is READY before extracting its module and returning it to its tray."
    )


TASK_PROMPTS = {
    "insert_one": insertion_prompt(1),
    "insert_two": insertion_prompt(2),
    "insert_three": insertion_prompt(3),
    "recover_three": insertion_prompt(3) + " Recover and retry if a module is knocked loose.",
    "timed_single": (
        "Seat the optical module to start the physical five-second clock. "
        "Keep it in the slot until the clock reaches zero, then fully withdraw the module to earn one point."
    ),
}


def load_trainer(path):
    spec = spec_from_file_location("tianji_training_contract", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Tianji trainer at {path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def training_setup(args):
    trainer = load_trainer(args.trainer)
    contract_path = args.training_config or args.checkpoint.parent.parent / "training_config.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for key, expected in {"action_horizon": 50, "action_dim": 32, "precision": "bfloat16"}.items():
        if contract.get(key) != expected:
            raise ValueError(f"Unexpected training contract {key}: {contract.get(key)!r}")
    stats = args.checkpoint / "assets" / "tianji_optical_modules"
    if not (stats / "norm_stats.json").is_file():
        raise FileNotFoundError(f"Checkpoint normalization values are missing from {stats}")
    settings = SimpleNamespace(
        dataset=args.dataset, norm_stats=stats, video_backend=contract["video_backend"],
        base_params=Path(contract["base_params"]), steps=contract["optimizer_steps"],
        run_id=contract["run_id"], run_dir=contract_path.parent, seed=contract["seed"],
        devices=contract["devices"], per_device_batch=contract["per_device_batch"], workers=0,
        save_interval=contract["save_interval"], log_interval=100, resume=False,
    )
    config = trainer.build_config(settings)
    dataset, _, _ = trainer.build_dataset(settings, config)
    return trainer, contract, config, dataset


def select_frames(dataset, task_prompt, stride, seed):
    import numpy as np
    from openpi import transforms

    add_prompt = transforms.PromptFromLeRobotTask(dataset.meta.tasks)
    episodes = []
    for episode in dataset.meta.episodes:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        prompt = add_prompt({"task_index": dataset.hf_dataset[start]["task_index"]})["prompt"]
        if prompt == task_prompt:
            if start >= end:
                raise ValueError(f"Invalid trajectory bounds: {start}, {end}")
            episodes.append((int(episode["episode_index"]), start, end))
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    indices = []
    for episode_index, start, end in episodes:
        frames = np.arange(start, end, stride)
        rng.shuffle(frames)
        indices.extend((episode_index, int(frame)) for frame in frames)
    return add_prompt, len(episodes), indices


def observations(dataset, frames, add_prompt, cameras, task_prompt):
    # The pinned LeRobot v3 reader accepts multiple timestamps from the same
    # episode and applies its usual video offsets and timestamp tolerance.
    camera_keys = tuple(cameras.values())
    for episode_index, episode_frames in groupby(frames, key=lambda frame: frame[0]):
        rows = [dataset.hf_dataset[index] for _, index in episode_frames]
        if any(int(row["episode_index"]) != episode_index for row in rows):
            raise ValueError(f"Frame indices do not match episode {episode_index}")
        timestamps = [float(row["timestamp"]) for row in rows]
        videos = dataset._query_videos(dict.fromkeys(camera_keys, timestamps), episode_index)
        for offset, row in enumerate(rows):
            frame = add_prompt(row)
            for camera, pixels in videos.items():
                image = pixels if len(rows) == 1 else pixels[offset]
                if dataset.image_transforms is not None:
                    image = dataset.image_transforms(image)
                frame[camera] = image
            yield observation(frame, cameras, task_prompt)


def observation(frame, cameras, task_prompt):
    import numpy as np

    prompt = frame["prompt"]
    if prompt != task_prompt:
        raise ValueError(f"Unexpected task prompt: {prompt!r}")
    state = np.asarray(frame["observation.state"], dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("Expected 16 finite Tianji joint and gripper positions")
    images = {}
    for camera, key in cameras.items():
        image = np.asarray(frame[key])
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected a CHW image for {key}: {image.shape}")
        if np.issubdtype(image.dtype, np.floating):
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError(f"Expected LeRobot pixels in [0, 1] for {key}")
            # Match the float-to-byte conversion in training's AlohaInputs.
            image = (255 * image).astype(np.uint8)
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 or normalized float pixels for {key}: {image.dtype}")
        images[camera] = image
    return {"state": state, "images": images, "prompt": prompt}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--openpi", type=Path, default=PI05_ROOT / "vendor/openpi")
    parser.add_argument("--trainer", type=Path, default=PI05_ROOT / "code/tianji_pi05.py")
    parser.add_argument("--dataset", type=Path, default=PI05_ROOT / "datasets/tianji_v3")
    parser.add_argument("--level", choices=TASK_PROMPTS, default="insert_one")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--include-language", action="store_true")
    args = parser.parse_args()
    if min(args.frame_stride, args.expected_episodes, args.max_frames or 1) < 1 or args.seed < 0:
        parser.error("Frame stride/count and episode count must be positive; seed must be nonnegative")
    if args.max_frames == 0:
        parser.error("--max-frames must be positive")
    if not (args.project / "rlt/openpi.py").is_file():
        parser.error(f"No RLT project source under {args.project}")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OPENPI_DATA_HOME", str(PI05_ROOT / "openpi-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    sys.path.insert(0, str(args.project.resolve()))
    sys.path.insert(0, str(args.openpi.resolve() / "src"))
    sys.path.insert(0, str(args.openpi.resolve() / "packages/openpi-client/src"))

    import jax
    import numpy as np
    from openpi.policies import policy_config
    from rlt.openpi import FrozenOpenPI

    trainer, contract, config, dataset = training_setup(args)
    task_prompt = TASK_PROMPTS[args.level]
    add_prompt, episodes, frames = select_frames(dataset, task_prompt, args.frame_stride, args.seed)
    if episodes != args.expected_episodes:
        raise ValueError(f"Expected {args.expected_episodes} {args.level} trajectories, got {episodes}")
    frames = frames[:args.max_frames]
    if not frames:
        raise ValueError(f"No {args.level} frames were selected")
    print(json.dumps({"level": args.level, "episodes": episodes, "frames": len(frames),
                      "devices": str(jax.devices())}), flush=True)
    native = policy_config.create_trained_policy(config, args.checkpoint, sample_kwargs={"num_steps": 10})
    vla = FrozenOpenPI(native, include_language=args.include_language, seed=args.seed)
    with TemporaryDirectory(prefix=".rlt-feature-stage-", dir=output.parent) as stage:
        staging = Path(stage) / "features"
        staging.mkdir()
        source_observations = observations(dataset, frames, add_prompt, trainer.CAMERAS, task_prompt)
        for counter, item in enumerate(source_observations):
            features, valid = vla.embeddings(item)
            np.savez(staging / f"{counter:07}.npz", features=features.astype(np.float16), valid=valid)
            written = counter + 1
            if written == 1 or written % 25 == 0 or written == len(frames):
                print(json.dumps({"written": written, "total": len(frames), "feature_shape": list(features.shape)}),
                      flush=True)
        provenance = {
            "vla_checkpoint": str(args.checkpoint.resolve()), "run_id": contract["run_id"],
            "include_language": args.include_language, "dataset": str(args.dataset.resolve()),
            "level": args.level, "task_prompt": task_prompt,
            "frame_stride": args.frame_stride, "seed": args.seed, "frames": len(frames), "episodes": episodes,
        }
        (staging / "source.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output)
    print(json.dumps({"output": str(output), **provenance}), flush=True)


if __name__ == "__main__":
    main()

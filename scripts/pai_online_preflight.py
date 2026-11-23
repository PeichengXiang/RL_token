"""Validate OpenPI and RTX Tianji in the selected evaluation runtime."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}, allow_nan=False), flush=True)


def diagnostic(name, arguments, keep=None):
    program = shutil.which(name)
    if program is None:
        return {"available": False}
    try:
        result = subprocess.run([program, *arguments], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error": str(exc)}
    lines = result.stdout.splitlines()
    if keep is not None:
        lines = [line.strip() for line in lines if any(field in line for field in keep)]
    return {"available": True, "returncode": result.returncode, "output": lines[:16],
            "stderr": result.stderr.splitlines()[:8]}


def camera_summary(observation, camera_names, action_dim, proprio_dim, np):
    model_input = observation.model_input
    cameras = model_input["images"]
    if set(cameras) != set(camera_names):
        raise RuntimeError(f"Expected three Tianji cameras; received {sorted(cameras)}")
    for label, state, width in (("position", model_input["state"], action_dim),
                                ("proprio", observation.proprio, proprio_dim)):
        if state.shape != (width,) or not np.isfinite(state).all():
            raise RuntimeError(f"Invalid Tianji {label} state: {state.shape}")
    summary = {}
    for name, rgb in cameras.items():
        if rgb.ndim != 3 or rgb.shape[0] != 3 or min(rgb.shape[1:]) < 2 or rgb.dtype != np.uint8:
            raise RuntimeError(f"Invalid RGB payload from {name}: {rgb.shape}, {rgb.dtype}")
        lo, hi = int(rgb.min()), int(rgb.max())
        if lo == hi:
            raise RuntimeError(f"Camera {name} returned a uniform frame at intensity {lo}")
        summary[name] = {"shape": list(rgb.shape), "range": [lo, hi]}
    return summary


def run(args):
    paths = {
        "benchmark": args.benchmark_root / "tools/runtime/isaac_python.sh",
        "Isaac Sim": args.isaac_sim_root / "python.sh",
        "Pi_05 adapter": args.pi_adapter / "openpi/src/openpi/models/pi0.py",
        "Tianji checkpoint": args.vla_checkpoint / "training_config.json",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing {path}")
    if not (args.vla_checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Missing Orbax params directory under {args.vla_checkpoint}")
    if args.openpi_data_home is not None:
        tokenizer = args.openpi_data_home / "big_vision/paligemma_tokenizer.model"
        if not tokenizer.is_file():
            raise FileNotFoundError(f"Missing staged OpenPI tokenizer: {tokenizer}")
        os.environ["OPENPI_DATA_HOME"] = str(args.openpi_data_home.resolve())
    if args.token is not None and not args.token.is_file():
        raise FileNotFoundError(args.token)

    os.environ["ISAAC_SIM_ROOT"] = str(args.isaac_sim_root.resolve())
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.vla_device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, str(PROJECT))
    start = time.monotonic()
    emit("runtime", python=sys.version.split()[0], executable=sys.executable, uid=os.getuid(),
         driver_capabilities=os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
         nvidia_smi=diagnostic("nvidia-smi", ["--query-gpu=name,driver_version,memory.total", "--format=csv"]),
         vulkan=diagnostic("vulkaninfo", ["--summary"], ("deviceName", "deviceType", "driverName", "driverInfo")))

    import jax
    import jax.numpy as jnp
    import numpy as np

    try:
        devices = jax.devices(args.vla_device)
        if not devices or jax.default_backend() != args.vla_device:
            raise RuntimeError(f"JAX is not using the requested {args.vla_device} backend")
        with jax.default_device(devices[0]):
            result = jax.jit(lambda array: array + 1)(jnp.zeros((8,), dtype=jnp.float32))
            result.block_until_ready()
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"JAX {args.vla_device} is unavailable. Check the active Python runtime and JAX_PLATFORMS."
        ) from exc
    stage = "jax_cuda" if args.vla_device == "gpu" else "jax_cpu"
    emit(stage, jax=jax.__version__, devices=[device.device_kind for device in devices])

    from rlt.openpi import FrozenOpenPI
    from rlt.rollout import encode_state
    from rlt.tianji import CAMERAS, TianjiEnv

    vla = FrozenOpenPI.from_tianji(
        args.vla_checkpoint, adapter_dir=args.pi_adapter, xpolicylab_dir=args.xpolicylab_root,
        include_language=args.include_language, num_steps=args.flow_steps, seed=args.vla_seed,
        shared_prefix=args.shared_prefix,
    )
    try:
        with TianjiEnv(args.level, max_steps=1, benchmark_root=args.benchmark_root,
                       timeout_s=args.isaac_timeout) as env:
            observation = env.reset(args.sim_seed)
            reset_cameras = camera_summary(observation, CAMERAS, env.action_dim, env.proprio_dim, np)
            emit("isaac_reset", level=args.level, seed=args.sim_seed, cameras=reset_cameras)

            features, valid, commands = vla.infer(observation.model_input)
            if (features.ndim != 2 or valid.shape != (len(features),) or valid.dtype != np.bool_
                    or not valid.any() or not np.isfinite(features[valid]).all()):
                raise RuntimeError(f"Invalid OpenPI features or validity mask: {features.shape}, {valid.shape}")
            if commands.shape != (vla.action_horizon, env.action_dim) or not np.isfinite(commands).all():
                raise RuntimeError(f"Invalid OpenPI reference chunk: {commands.shape}")
            token_width = None
            if args.token is not None:
                from rlt.token import load_token

                token = load_token(args.token, args.learner_device)
                state = encode_state(token, features, valid, observation.proprio)
                if not np.isfinite(state).all():
                    raise RuntimeError("Stage 1 tokenizer produced a nonfinite RL state")
                token_width = token.config.token_dim
            emit("openpi_inference", features=list(features.shape), valid_tokens=int(valid.sum()),
                 reference=list(commands.shape), rl_token_width=token_width)

            next_obs, reward, terminated, truncated, info = env.step(commands[0])
            step_cameras = camera_summary(next_obs, CAMERAS, env.action_dim, env.proprio_dim, np)
            if info.get("steps") != 1 or not (terminated or truncated) or not np.isfinite(reward):
                raise RuntimeError("Tianji failed to complete the native one-step trial")
            emit("isaac_step", steps=1, terminated=bool(terminated), truncated=bool(truncated),
                 reward=float(reward), cameras=step_cameras)
    except (RuntimeError, OSError):
        print("For camera or Kit failures, check NVIDIA Vulkan ICD/RTX devices in this container, "
              "host driver compatibility, read/write cache permissions and GPU memory.", file=sys.stderr)
        raise

    exit_status = env._process.returncode
    if exit_status != 0:
        raise RuntimeError(f"Isaac worker did not exit cleanly: status {exit_status}")
    emit("complete", isaac_worker_exit=exit_status, seconds=round(time.monotonic() - start, 2))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--isaac-sim-root", type=Path, required=True)
    parser.add_argument("--vla-checkpoint", type=Path, required=True)
    parser.add_argument("--pi-adapter", type=Path, default=PROJECT.parent / "Pi_05")
    parser.add_argument("--xpolicylab-root", type=Path)
    parser.add_argument("--openpi-data-home", type=Path)
    parser.add_argument("--token", type=Path, help="Also load the trained Stage 1 token and encode the reset state")
    parser.add_argument("--vla-device", choices=("gpu", "cpu"), default="gpu",
                        help="Run OpenPI on CPU when RTX is reserved for Isaac Sim")
    parser.add_argument("--learner-device", default="cpu")
    parser.add_argument("--level", default="insert_one")
    parser.add_argument("--sim-seed", type=int, default=21001)
    parser.add_argument("--vla-seed", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--isaac-timeout", type=float, default=300)
    parser.add_argument("--include-language", action="store_true")
    parser.add_argument("--shared-prefix", action="store_true")
    args = parser.parse_args()
    if args.xpolicylab_root is None:
        args.xpolicylab_root = args.benchmark_root / "third_party/XPolicyLab"
    if min(args.sim_seed, args.vla_seed) < 0 or min(args.flow_steps, args.isaac_timeout) <= 0:
        parser.error("Seeds must be nonnegative, and flow steps and Isaac timeout must be positive")
    run(args)


if __name__ == "__main__":
    main()

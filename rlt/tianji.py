"""Tianji optical-module demonstrations and a local Isaac Sim environment."""

from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import signal
import socket
import subprocess
from tempfile import TemporaryFile
import traceback
from typing import Iterator

import numpy as np


DEFAULT_BENCHMARK = Path(__file__).resolve().parents[3] / "insert_benchmark"
CAMERAS = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
CONTROL_HZ = 30
ACTION_DIM = 16
GRIPPERS = [7, 15]


def ready_phase(_observation, info):
    """Switch to RLT after the native countdown has finished in every slot."""
    states = info.get("timer_state", ())
    return bool(states) and all(state == "ready" for state in states)


def _model_input(state: np.ndarray, images: dict[str, np.ndarray], prompt: str) -> dict:
    """Match the input of the benchmark's Pi_05 preprocessing pipeline."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Tianji observations require a nonempty task instruction")
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise ValueError("Tianji observations require 16 finite joint/gripper positions")
    routed = {}
    for destination, source in CAMERAS.items():
        image = np.asarray(images[source])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Expected HWC uint8 RGB from {source}")
        routed[destination] = image.transpose(2, 0, 1).copy()
    return {"state": state.copy(), "images": routed, "prompt": prompt}


def iter_demonstrations(
    dataset_root: Path | str = DEFAULT_BENCHMARK / "benchmark_data",
    level: str = "timed_single",
    *,
    stride: int = 1,
    max_episodes: int | None = None,
    horizon: int = 50,
    seed: int | None = None,
) -> Iterator[dict]:
    """Stream raw OpenPI observations and same-step, absolute action targets.

    Use the recorded pre-action observation. Repeat the last demonstrated
    action when its future window runs past the episode. Setting ``seed``
    shuffles episode order and the selected frame order within each episode.
    """
    import h5py

    if stride < 1 or horizon < 1 or (max_episodes is not None and max_episodes < 1):
        raise ValueError("Tianji stride, horizon, and episode limit must be positive")
    directory = Path(dataset_root).expanduser() / level
    episodes = sorted(directory.glob("seed_*/demonstration.hdf5"))
    if not episodes:
        raise FileNotFoundError(f"No Tianji HDF5 demonstrations found under {directory}")
    rng = np.random.default_rng(seed) if seed is not None else None
    if rng is not None:
        rng.shuffle(episodes)
    offsets = np.arange(horizon)
    for path in episodes[:max_episodes]:
        with h5py.File(path, "r") as demo:
            if (demo.attrs.get("level") != level or demo.attrs.get("frequency") != CONTROL_HZ
                    or demo.attrs.get("control_label") != "same_step_control"):
                raise ValueError(f"Unexpected Tianji task, frequency, or control alignment in {path}")
            prompt = demo.attrs["instruction"]
            actions = np.asarray(demo["action"], dtype=np.float32)
            states = np.asarray(demo["observations/qpos"], dtype=np.float32)
            images = {name: demo[f"observations/images/{name}"] for name in CAMERAS.values()}
            if (actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not len(actions)
                    or states.shape != actions.shape or not np.isfinite(actions).all()
                    or any(len(camera) != len(actions) for camera in images.values())):
                raise ValueError(f"Invalid per-step observation or action arrays in {path}")
            indices = np.arange(0, len(actions), stride)
            if rng is not None:
                rng.shuffle(indices)
            for index in indices:
                observation = _model_input(states[index], {name: camera[index] for name, camera in images.items()}, prompt)
                yield {**observation, "actions": actions[np.minimum(index + offsets, len(actions) - 1)]}


class TianjiEnv:
    """Run the native Isaac Sim benchmark in its bundled Python subprocess.

    This leaves the learner free to use its own Torch and OpenPI/JAX runtime.
    Only the spawned worker has access to the private connection.
    """

    control_hz = CONTROL_HZ
    action_dim = ACTION_DIM
    proprio_dim = 2 * ACTION_DIM

    def __init__(
        self,
        level: str = "timed_single",
        max_steps: int | None = None,
        *,
        benchmark_root: Path | str = DEFAULT_BENCHMARK,
        timeout_s: float = 180,
    ):
        if (max_steps is not None and max_steps < 1) or timeout_s <= 0:
            raise ValueError("Tianji max_steps and timeout must be positive")
        launcher = Path(benchmark_root).expanduser().resolve() / "tools/runtime/isaac_python.sh"
        if not launcher.is_file():
            raise FileNotFoundError(f"Isaac benchmark launcher not found at {launcher}")
        code_root = Path(__file__).resolve().parents[1]
        self._timeout = timeout_s
        self._log = TemporaryFile()
        self._process = None
        self._connection = None
        parent, child = socket.socketpair()
        with parent, child:
            self._connection = Connection(parent.detach())
            command = [str(launcher), "-m", "rlt.tianji", "--worker-fd", str(child.fileno()), "--level", level]
            if max_steps is not None:
                command += ["--max-steps", str(max_steps)]
            env = dict(os.environ, PYTHONPATH=str(code_root), PYTHONNOUSERSITE="1")
            for name in ("PYTHONHOME", "PYTHONEXE", "VIRTUAL_ENV", "CONDA_PREFIX"):
                env.pop(name, None)
            try:
                self._process = subprocess.Popen(
                    command, cwd=code_root, env=env, pass_fds=(child.fileno(),), start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
                )
            except BaseException:
                self.close()
                raise
        try:
            self.action_low, self.action_high = self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        try:
            if not self._connection.poll(self._timeout):
                raise TimeoutError(f"No response within {self._timeout:g} seconds")
            ok, result = self._connection.recv()
        except (EOFError, OSError) as exc:
            log = os.pread(self._log.fileno(), 6000, max(0, os.fstat(self._log.fileno()).st_size - 6000))
            raise RuntimeError(f"Isaac worker connection failed: {exc}\n{log.decode(errors='replace')}") from exc
        if not ok:
            raise RuntimeError(f"Isaac benchmark worker failed\n{result}")
        return result

    def _request(self, operation, argument=None):
        if self._connection is None:
            raise RuntimeError("Tianji environment is closed")
        try:
            self._connection.send((operation, argument))
        except OSError as exc:
            raise RuntimeError("Isaac worker stopped before receiving the request") from exc
        return self._receive()

    def reset(self, seed: int):
        from .rollout import Observation

        return Observation(*self._request("reset", int(seed)))

    @staticmethod
    def action_offset(observation) -> np.ndarray:
        """Use joint deltas from the chunk start and absolute gripper openings."""
        offset = np.asarray(observation.model_input["state"], dtype=np.float32).copy()
        offset[GRIPPERS] = 0.0
        return offset

    def action_projection(self, device):
        """Torch projection for the same absolute limits in learner coordinates."""
        import torch

        low = torch.as_tensor(self.action_low, dtype=torch.float32, device=device)
        high = torch.as_tensor(self.action_high, dtype=torch.float32, device=device)
        arm = torch.ones(ACTION_DIM, dtype=torch.float32, device=device)
        arm[GRIPPERS] = 0

        def project(state, actions):
            joint_position = state[:, -self.proprio_dim:-ACTION_DIM]
            offset = (joint_position * arm)[:, None]
            return actions.clamp(min=low - offset, max=high - offset)

        return project

    def step(self, action: np.ndarray):
        from .rollout import Observation

        inputs, reward, terminated, truncated, info = self._request("step", np.asarray(action, dtype=np.float32))
        return Observation(*inputs), reward, terminated, truncated, info

    def close(self) -> None:
        if self._connection is None:
            return
        # EOF asks the worker to shut down the camera rig and Kit cleanly.
        self._connection.close()
        self._connection = None
        if self._process is not None:
            try:
                self._process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._process.wait()
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _IsaacBench:
    """Worker-side wrapper around the benchmark's unchanged scene and scoring."""

    def __init__(self, level: str, max_steps: int | None):
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import create_new_stage
        from insert_benchmark.benchmark.physics import BenchPhysics
        from insert_benchmark.benchmark.sensors import BenchSensors
        from insert_benchmark.benchmark.suite import LEVELS, ROOT, step_limit

        self.level = level
        self.max_steps = step_limit(level) if max_steps is None else max_steps
        self.finished = True
        config = json.loads((ROOT / "configs/marvin.json").read_text())
        count = LEVELS[level].module_count
        create_new_stage()
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / 240.0, rendering_dt=1 / CONTROL_HZ, backend="numpy")
        self.bench = BenchPhysics(self.world, config, count)
        self.rig = BenchSensors(self.world, self.bench.robot, config, module_count=count, level=level)
        self.lower, self.upper = np.concatenate(
            (self.bench.robot.limits["left"], self.bench.robot.limits["right"])
        ).T
        self.action_low = self.lower.copy()
        self.action_high = self.upper.copy()
        self.action_low[GRIPPERS], self.action_high[GRIPPERS] = 0.0, 1.0

    def _observe(self, metrics):
        observation, state, velocity, _ = self.rig.observe(metrics, self.prompt)
        images = {name: camera["color"] for name, camera in observation["vision"].items()}
        proprio = np.concatenate((state, velocity)).astype(np.float32)
        return _model_input(state, images, self.prompt), proprio

    def reset(self, seed):
        from insert_benchmark.benchmark.runner import _instruction
        from insert_benchmark.benchmark.suite import episode_spec

        spec = episode_spec(self.level, seed)
        self.prompt = _instruction(spec)
        self.bench.reset(spec)
        self.rig.reset()
        self.finished = False
        return self._observe(self.bench.metrics())

    def step(self, action):
        if self.finished:
            raise RuntimeError("Reset Tianji before starting a new episode")
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("Tianji actions must contain 16 finite absolute joint/gripper targets")
        # The environment exposes normalized gripper opening, but Isaac needs
        # each gripper's native mechanical range.
        applied = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        native = applied.astype(float)
        native[GRIPPERS] = self.lower[GRIPPERS] + applied[GRIPPERS] * (self.upper - self.lower)[GRIPPERS]
        metrics = self.bench.step(native)
        success = bool(metrics["success"])
        terminated = success or bool(metrics["done"])
        truncated = not terminated and metrics["steps"] >= self.max_steps
        self.finished = terminated or truncated
        steps = int(metrics["steps"])
        info = {
            "success": success, "action": applied, "steps": steps,
            "timer_state": tuple(metrics["timer_state"]),
            "events": tuple(event for event in metrics["events"] if event["step"] == steps),
        }
        return self._observe(metrics), float(success), terminated, truncated, info

    def close(self):
        from isaacsim.core.api import World

        self.rig.close()
        self.world.stop()
        World.clear_instance()


def _worker(fd: int, level: str, max_steps: int | None) -> None:
    from insert_benchmark.benchmark.recording import HEIGHT, WIDTH
    from insert_benchmark.lifecycle import close_application
    from isaacsim import SimulationApp

    app = bench = None
    status = 1
    with Connection(fd) as connection:
        try:
            app = SimulationApp(
                {"headless": True, "width": WIDTH, "height": HEIGHT, "multi_gpu": False, "limit_cpu_threads": 4,
                 "disable_viewport_updates": True, "enable_crashreporter": False, "renderer": "RaytracedLighting",
                 "fast_shutdown": True},
                experience=str(Path(os.environ["EXP_PATH"]) / "isaacsim.exp.base.zero_delay.kit"),
            )
            bench = _IsaacBench(level, max_steps)
            connection.send((True, (bench.action_low, bench.action_high)))
            commands = {"reset": bench.reset, "step": bench.step}
            while True:
                try:
                    operation, argument = connection.recv()
                except EOFError:
                    break
                connection.send((True, commands[operation](argument)))
            status = 0
        except Exception:
            detail = traceback.format_exc()
            print(detail, flush=True)
            try:
                connection.send((False, detail))
            except (EOFError, OSError):
                pass
        finally:
            try:
                if bench is not None:
                    bench.close()
            finally:
                if app is not None:
                    close_application(app, status)
    raise SystemExit(status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal Tianji Isaac Sim rollout worker")
    parser.add_argument("--worker-fd", type=int, required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    _worker(args.worker_fd, args.level, args.max_steps)

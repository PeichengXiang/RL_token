"""Seed Tianji replay with native successful expert demonstrations."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .online import Transition
from .rollout import Backbone, RolloutConfig, _reference, encode_state
from .tianji import ACTION_DIM, CAMERAS, CONTROL_HZ, DEFAULT_BENCHMARK, GRIPPERS, _model_input
from .token import RLToken


def _episode_transitions(path, report, vla, token, config, stride, phase, condition_on_vla):
    import h5py

    steps = report["steps"]
    stop = steps
    if report["level"] == "timed_single":
        # In both timed benchmark versions, the first complete withdrawal
        # ends the task. The preceding removal event alone is insufficient.
        stop = next((event["step"] for event in report["metrics"]["events"]
                     if event["event"] == "extracted"), None)
        if stop is None or not 1 <= stop <= steps:
            raise ValueError(f"No valid terminal withdrawal in {path.parent}")

    with h5py.File(path, "r") as data:
        expected_attrs = {
            "level": report["level"], "seed": report["seed"], "steps": steps,
            "frequency": CONTROL_HZ, "instruction": report["instruction"],
            "control_label": "same_step_control", "gripper_unit": "fraction_0_closed_1_open",
        }
        if not data.attrs.get("sim") or any(data.attrs.get(k) != v for k, v in expected_attrs.items()):
            raise ValueError(f"Unexpected demonstration contract in {path}")
        actions = np.asarray(data["action"], dtype=np.float32)
        qpos = np.asarray(data["observations/qpos"], dtype=np.float32)
        qvel = np.asarray(data["observations/qvel"], dtype=np.float32)
        ts = np.asarray(data["timestamp_s"], dtype=np.float64)
        images = {name: data[f"observations/images/{name}"] for name in CAMERAS.values()}
        if (
            any(a.shape != (steps, ACTION_DIM) or not np.isfinite(a).all() for a in (actions, qpos, qvel))
            or any(np.any((a[:, GRIPPERS] < 0.) | (a[:, GRIPPERS] > 1.)) for a in (actions, qpos))
            or ts.shape != (steps,) or not np.allclose(ts, np.arange(steps) / CONTROL_HZ, atol=.0005, rtol=0)
            or any(len(camera) != steps for camera in images.values())
        ):
            raise ValueError(f"Invalid same-step observations or actions in {path}")

        first = 0
        if phase == "ready":
            ready_code = list(data.attrs["timer_state_codes"]).index("ready")
            timer_states = data["observations/timer_state"][:stop]
            ready = np.flatnonzero(np.all(timer_states == ready_code, axis=1))
            if not len(ready):
                raise ValueError(f"No recorded READY handoff in {path}")
            # Row i observes action i-1's outcome, before action i is issued.
            first = int(ready[0])

        chunk = config.chunk_size
        starts = set(range(first, stop, stride))
        # Retain the usual overlapping terminal windows even when using a
        # larger interior stride to limit frozen encoder work.
        starts.update(range(max(first, stop - chunk), stop, config.stride))
        starts.add(stop - 1)

        def relative_offset(index):
            offset = qpos[index].copy()
            offset[GRIPPERS] = 0.
            return offset

        def expert_chunk(index):
            end = min(index + chunk, stop)
            window = np.zeros((chunk, ACTION_DIM), dtype=np.float32)
            window[:end - index] = actions[index:end] - relative_offset(index)
            return window

        @lru_cache(maxsize=None)
        def features_at(index):
            model_input = _model_input(
                qpos[index], {name: camera[index] for name, camera in images.items()}, report["instruction"],
            )
            if condition_on_vla:
                features, valid, absolute_reference = vla.infer(model_input)
                reference = _reference(absolute_reference, chunk, ACTION_DIM) - relative_offset(index)
            else:
                features, valid = vla.embeddings(model_input)
                # Preserve the expert intervention path without sampling VLA actions.
                reference = expert_chunk(index)
            state = encode_state(token, features, valid, np.concatenate((qpos[index], qvel[index])))
            return state, reference

        transitions = []
        for start in sorted(starts):
            end = min(start + chunk, stop)
            count = end - start
            terminal = end == stop
            state, reference = features_at(start)
            action = expert_chunk(start)
            if terminal:
                next_state, next_reference = np.zeros_like(state), np.zeros_like(action)
            else:
                next_state, next_reference = features_at(end)
            reference = reference.copy()
            reference[count:] = 0.
            transitions.append(Transition(
                state, action, reference, config.gamma ** (count - 1) if terminal else 0.,
                next_state, next_reference, 0. if terminal else config.gamma ** count,
                np.arange(chunk) < count, bc_target=action if condition_on_vla else None,
            ))
        return transitions


def iter_expert_replay(
    dataset_root: Path | str,
    level: str,
    vla: Backbone,
    token: RLToken,
    config: RolloutConfig,
    *,
    max_episodes: int = 4,
    stride: int = 10,
    seed: int = 0,
    phase: str | None = None,
    benchmark_root: Path | str = DEFAULT_BENCHMARK,
    condition_on_vla: bool = False,
):
    """Yield ``(seed, transitions)`` from verified successful training demos.

    A demonstration gets the native binary success label once, at its final
    control action. Older reports certify success for their recorded version.
    With VLA conditioning, the actor receives a frozen VLA reference and
    imitates the expert action recorded at that same observation.
    """
    if min(max_episodes, stride) < 1 or phase not in (None, "ready"):
        raise ValueError("Invalid demo count, stride, or critical phase")
    manifest = Path(benchmark_root) / "configs/benchmark/seeds.json"
    split = json.loads(manifest.read_text(encoding="utf-8"))[level]
    demo_seeds = sorted(set(split["demo"]) - set(split["test"]))
    np.random.default_rng(seed).shuffle(demo_seeds)
    episodes = []
    for demo_seed in demo_seeds:
        directory = Path(dataset_root) / level / f"seed_{demo_seed}"
        path, report_path = directory / "demonstration.hdf5", directory / "report.json"
        if not path.is_file() or not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = report.get("metrics", {})
        if (
            report.get("seed") == demo_seed and report.get("level") == level and report.get("split") == "demo"
            and report.get("operation") == "cycle" and report.get("policy") == "expert"
            and report.get("success") is True and metrics.get("done") is True and metrics.get("success") is True
            and isinstance(report.get("steps"), int) and report["steps"] > 0
            and report.get("frames") == metrics.get("steps") == report["steps"]
            and not report.get("time_limit_reached") and not report.get("policy_error")
            and not any(metrics.get("early_removal", ()))
        ):
            episodes.append((demo_seed, path, report))
        if len(episodes) == max_episodes:
            break
    if len(episodes) != max_episodes:
        raise ValueError(f"Expected {max_episodes} successful expert demos for {level}; found {len(episodes)}")

    for demo_seed, path, report in episodes:
        if condition_on_vla:
            vla.reset(demo_seed)
        yield demo_seed, _episode_transitions(path, report, vla, token, config, stride, phase, condition_on_vla)

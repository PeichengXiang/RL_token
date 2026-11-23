"""Collect overlapping, off-policy action chunks for the critical phase."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Protocol

import numpy as np
import torch

from .online import Learner, ReplayBuffer, Transition
from .token import RLToken


@dataclass(frozen=True)
class Observation:
    model_input: dict
    proprio: np.ndarray


class Environment(Protocol):
    control_hz: float
    action_dim: int
    proprio_dim: int

    def reset(self, seed: int) -> Observation: ...

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict]: ...


class Backbone(Protocol):
    def reset(self, seed: int) -> None: ...

    def embeddings(self, observation: dict) -> tuple[np.ndarray, np.ndarray]: ...

    def infer(self, observation: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def reference(self, observation: dict) -> np.ndarray: ...


@dataclass(frozen=True)
class RolloutConfig:
    chunk_size: int = 10
    stride: int = 2
    base_chunk_size: int = 20
    gamma: float = 0.99
    warmup_chunk_size: int | None = None

    def __post_init__(self):
        if min(self.chunk_size, self.stride, self.base_chunk_size) < 1 or not 0 <= self.gamma <= 1:
            raise ValueError("Invalid chunk length, stride or discount")
        if self.warmup_chunk_size is not None and self.warmup_chunk_size < 1:
            raise ValueError("Warmup chunk length must be positive")


@dataclass(frozen=True)
class Episode:
    steps: int
    critical_steps: int
    success: bool
    reward: float
    transitions: int
    terminated: bool = False
    truncated: bool = False
    timer_state: tuple[str, ...] = ()
    first_ready_step: int | None = None
    events: tuple[dict, ...] = ()
    actor_steps: int = 0


@torch.no_grad()
def encode_state(model: RLToken, features: np.ndarray, valid: np.ndarray, proprio: np.ndarray):
    device = model.readout.device
    tokens = torch.as_tensor(features, dtype=torch.float32, device=device)
    mask = torch.tensor(valid, dtype=torch.bool, device=device)
    state = model.encode(tokens[None], mask[None])[0].cpu().numpy()
    return np.concatenate((state, np.asarray(proprio, dtype=np.float32)))


def _reference(actions, required, action_dim):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < required or actions.shape[1] != action_dim:
        raise ValueError(f"Need reference actions of shape [at least {required}, {action_dim}]")
    if not np.isfinite(actions[:required]).all():
        raise ValueError("VLA reference contains nonfinite actions")
    return actions[:required].copy()


def run_episode(
    env: Environment,
    vla: Backbone,
    token: RLToken | None,
    config: RolloutConfig,
    seed: int,
    *,
    policy: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    warmup_steps: int = 0,
    on_transitions: Callable[[list[Transition], int], None] | None = None,
    phase: Callable[[Observation, dict], bool] | None = None,
    intervention: Callable[[Observation, np.ndarray], np.ndarray | None] | None = None,
    action_offset: Callable[[Observation], np.ndarray] | None = None,
    vla_seed: int | None = None,
    replay_vla: Backbone | None = None,
) -> Episode:
    """Run a VLA-led prefix, then collect and learn within the selected critical phase.

    `warmup_steps` counts critical control steps before the actor takes over at a chunk boundary.
    An optional `config.warmup_chunk_size` changes VLA execution only, not replay windows.
    `phase` is a hand-off predicate, latched until the episode ends.
    `intervention` may replace any individual control action in environment units.
    `action_offset` defines the chunk origin in environment units for selected channels.
    """
    if (policy is not None or on_transitions is not None) and token is None:
        raise ValueError("The actor and replay require a frozen RL token encoder")

    vla.reset(seed if vla_seed is None else vla_seed)
    if replay_vla is not None:
        replay_vla.reset(seed)
    observation = env.reset(seed)
    info = {}
    active = phase is None
    handoff = phase is not None and phase(observation, info)
    steps = critical_steps = transition_count = next_start = actor_steps = 0
    total_reward = 0.0
    terminated = truncated = False
    first_ready_step = None
    native_events = []
    chunk = config.chunk_size

    # Retain raw frames until their stride windows close, then release them.
    # VLA queries for intermediate frames happen after executing each chunk.
    frames = {}
    features = {}
    actions, rewards, interventions = [], [], []
    if active and on_transitions is not None:
        frames[0] = observation

    def offset(frame):
        if action_offset is None:
            return np.zeros(env.action_dim, dtype=np.float32)
        origin = np.asarray(action_offset(frame), dtype=np.float32)
        if origin.shape != (env.action_dim,) or not np.isfinite(origin).all():
            raise ValueError("Expected a finite action offset for each action dimension")
        return origin

    def warmup_length(index):
        if index < warmup_steps and config.warmup_chunk_size is not None:
            return config.warmup_chunk_size
        return chunk

    def features_at(index, *, behavior=False):
        if index not in features:
            frame = frames[index]
            # Intermediate replay queries must not consume behavior samples.
            sampler = vla if behavior or replay_vla is None else replay_vla
            embedding, valid, reference = sampler.infer(frame.model_input)
            state = encode_state(token, embedding, valid, frame.proprio)
            length = max(chunk, warmup_length(index)) if behavior else chunk
            absolute = _reference(reference, length, env.action_dim)
            features[index] = (state, absolute[:chunk] - offset(frame), absolute)
        return features[index]

    while not (terminated or truncated):
        if not active and handoff:
            active = True
            if on_transitions is not None:
                frames[0] = observation

        use_actor = active and policy is not None and critical_steps >= warmup_steps
        length = warmup_length(critical_steps) if active else config.base_chunk_size
        if active and (on_transitions is not None or use_actor):
            if on_transitions is not None:
                state, reference, absolute = features_at(critical_steps, behavior=True)
            else:
                embedding, valid, reference = vla.infer(observation.model_input)
                state = encode_state(token, embedding, valid, observation.proprio)
                reference = _reference(reference, chunk, env.action_dim) - offset(observation)
            if use_actor:
                planned = policy(state, reference) + offset(observation)
            elif config.warmup_chunk_size is not None and critical_steps < warmup_steps:
                planned = absolute[:length]
            else:
                planned = reference + offset(observation)
        else:
            planned = _reference(vla.reference(observation.model_input), length, env.action_dim)

        if np.shape(planned) != (length, env.action_dim):
            raise ValueError("Policy returned an invalid action chunk")
        for proposed in planned:
            override = intervention(observation, proposed) if active and intervention else None
            command = np.asarray(proposed if override is None else override, dtype=np.float32)
            if command.shape != (env.action_dim,) or not np.isfinite(command).all():
                raise ValueError("Expected a finite environment action")
            observation, reward, terminated, truncated, info = env.step(command.copy())
            steps += 1
            actor_steps += int(use_actor and override is None)
            total_reward += float(reward)
            if first_ready_step is None and "ready" in (info.get("timer_state") or ()):
                first_ready_step = steps
            native_events.extend(info.get("events") or ())
            if active:
                critical_steps += 1
                if on_transitions is not None:
                    applied = np.asarray(info.get("action", command), dtype=np.float32)
                    if applied.shape != (env.action_dim,) or not np.isfinite(applied).all():
                        raise ValueError("Environment reported an invalid applied action")
                    actions.append(applied.copy())
                    rewards.append(float(reward))
                    interventions.append(override is not None)
                    frames[critical_steps] = observation
            if terminated or truncated:
                break
            if not active and phase(observation, info):
                handoff = True
                break

        if on_transitions is None or not active:
            continue
        transitions = []
        while next_start + chunk <= critical_steps or (terminated and next_start < critical_steps):
            start = next_start
            end = min(start + chunk, critical_steps)
            terminal = terminated and end == critical_steps
            state, reference, _ = features_at(start)
            if terminal:
                next_state, next_reference = np.zeros_like(state), np.zeros_like(reference)
            else:
                # The just-collected chunk's endpoint is the next behavior
                # boundary if another action will be executed there.
                next_state, next_reference, _ = features_at(end, behavior=end == critical_steps and not truncated)
            count = end - start
            actual = np.zeros_like(reference)
            actual[:count] = np.asarray(actions[start:end]) - offset(frames[start])
            reference = reference.copy()
            replacements = np.flatnonzero(interventions[start:end])
            reference[replacements] = actual[replacements]
            reference[count:] = 0
            discounted_reward = sum(
                (config.gamma**i) * value for i, value in enumerate(rewards[start:end])
            )
            transitions.append(Transition(
                state, actual, reference, discounted_reward, next_state, next_reference,
                0.0 if terminal else config.gamma**count, np.arange(chunk) < count,
            ))
            next_start += config.stride
        for index in tuple(frames):
            # Keep the current boundary even when stride exceeds chunk size.
            if index < min(next_start, critical_steps):
                frames.pop(index)
                features.pop(index, None)
        if transitions:
            transition_count += len(transitions)
            on_transitions(transitions, critical_steps)

    return Episode(
        steps, critical_steps, bool(info.get("success", total_reward > 0)), total_reward, transition_count,
        terminated=bool(terminated), truncated=bool(truncated), timer_state=tuple(info.get("timer_state") or ()),
        first_ready_step=first_ready_step, events=tuple(native_events), actor_steps=actor_steps,
    )


class OnlineTrainer:
    """Run learner updates alongside collection, with a bounded work queue."""

    def __init__(self, learner: Learner, *, seed: int, warmup_steps: int = 1000,
                 updates_per_transition: int = 5, asynchronous: bool = True):
        if warmup_steps < 0 or updates_per_transition < 1:
            raise ValueError("Invalid warmup or update-to-data ratio")
        self.learner = learner
        self.warmup_steps = warmup_steps
        self.updates_per_transition = updates_per_transition
        self.critical_steps = 0
        self.transitions = 0
        self.metrics = {}
        self.replay = ReplayBuffer(learner.config.replay_capacity, seed=seed)
        self.lock = Lock()
        self.pool = ThreadPoolExecutor(max_workers=1) if asynchronous else None
        self.pending = None

    def _learn(self, transitions, steps):
        for transition in transitions:
            self.replay.add(transition)
        if steps < self.warmup_steps or len(self.replay) < self.learner.config.batch_size:
            return
        for _ in range(self.updates_per_transition * len(transitions)):
            batch = self.replay.sample(self.learner.config.batch_size)
            with self.lock:
                self.metrics.update(self.learner.update(batch))

    def _submit(self, transitions, episode_steps):
        self.finish()
        self.transitions += len(transitions)
        steps = self.critical_steps + episode_steps
        if self.pool is None:
            self._learn(transitions, steps)
        else:
            self.pending = self.pool.submit(self._learn, transitions, steps)

    def finish(self):
        if self.pending is not None:
            pending, self.pending = self.pending, None
            pending.result()

    def state_dict(self):
        self.finish()
        return {
            "critical_steps": self.critical_steps,
            "transitions": self.transitions,
            "metrics": self.metrics.copy(),
            "replay": self.replay.state_dict(),
        }

    def load_state_dict(self, state):
        self.finish()
        self.replay.load_state_dict(state["replay"])
        self.critical_steps = state["critical_steps"]
        self.transitions = state["transitions"]
        self.metrics = state["metrics"].copy()

    def close(self):
        try:
            self.finish()
        finally:
            if self.pool is not None:
                self.pool.shutdown()

    def _act(self, state, reference):
        with self.lock:
            return self.learner.act(state, reference)

    def train_episode(self, env, vla, token, config, seed, **kwargs):
        episode = run_episode(
            env, vla, token, config, seed, policy=self._act,
            warmup_steps=max(0, self.warmup_steps - self.critical_steps),
            on_transitions=self._submit, **kwargs,
        )
        self.critical_steps += episode.critical_steps
        return episode

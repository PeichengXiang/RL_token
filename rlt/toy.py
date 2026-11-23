"""A planar slot task and synthetic frozen features for an end-to-end test."""

import numpy as np

from .rollout import Observation


class SlotEnv:
    control_hz = 50
    action_dim = 2
    proprio_dim = 2
    action_limit = 0.12

    def action_projection(self, state, actions):
        return actions.clamp(min=-self.action_limit, max=self.action_limit)

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        self.slot = rng.uniform(-0.5, 0.5)
        self.xy = np.array([rng.uniform(-0.6, 0.6), 0.0], dtype=np.float32)
        self.steps = 0
        return self._observe()

    def _observe(self):
        return Observation({"scene": np.array([*self.xy, self.slot], dtype=np.float32)}, self.xy.copy())

    def step(self, action):
        executed = np.clip(action, -self.action_limit, self.action_limit).astype(np.float32)
        self.xy[0] = np.clip(self.xy[0] + executed[0], -1.0, 1.0)
        if abs(self.xy[0] - self.slot) < 0.16:
            self.xy[1] = np.clip(self.xy[1] + executed[1], 0, 1)
        self.steps += 1
        success = self.xy[1] >= 1.0
        return (
            self._observe(), float(success), bool(success or self.steps >= 60),
            False, {"success": bool(success), "action": executed},
        )


class ToyVLA:
    action_horizon = 50
    feature_dim = 16
    feature_count = 8

    def __init__(self):
        model_rng = np.random.default_rng(483)
        self.weights = model_rng.normal(size=(8, 3, 16)).astype(np.float32)
        self.bias = model_rng.normal(size=(8, 16)).astype(np.float32)
        self.reset(0)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def embeddings(self, observation):
        features = np.tanh(observation["scene"] @ self.weights + self.bias)
        return features, np.ones(len(features), dtype=bool)

    def reference(self, observation):
        position, _, slot = observation["scene"]
        lateral = np.clip((slot - position + self.rng.normal(0, 0.055)) / 8, -0.07, 0.07)
        ahead = 0.06
        return np.tile(np.array([lateral, ahead], dtype=np.float32), (self.action_horizon, 1))

    def infer(self, observation):
        return (*self.embeddings(observation), self.reference(observation))


def toy_demonstrations(vla, count, seed):
    rng = np.random.default_rng(seed)
    return [vla.embeddings({"scene": state}) for state in rng.uniform(
        [-0.8, 0.0, -0.5], [0.8, 1.0, 0.5], size=(count, 3),
    ).astype(np.float32)]

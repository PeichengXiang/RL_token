"""Chunk-level actor-critic and replay for online RLT (paper §IV-B)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class OnlineConfig:
    hidden_dim: int = 256
    hidden_layers: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    beta: float = 0.1
    action_std: float = 0.05
    reference_dropout: float = 0.5
    tau: float = 0.005
    actor_delay: int = 2
    batch_size: int = 256
    replay_capacity: int = 100_000

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.hidden_layers, self.actor_delay, self.batch_size, self.replay_capacity) < 1:
            raise ValueError("Network dimensions, actor delay, batch size and replay capacity must be positive")
        if min(self.actor_lr, self.critic_lr) <= 0 or min(self.beta, self.action_std) < 0:
            raise ValueError("Learning rates must be positive and beta and action_std nonnegative")
        if not 0 <= self.reference_dropout <= 1 or not 0 < self.tau <= 1:
            raise ValueError("reference_dropout must be in [0, 1] and tau in (0, 1]")


@dataclass(frozen=True)
class Transition:
    """One chunk with discounted reward, zero terminal discount, and optional expert BC labels."""

    state: FloatArray
    action: FloatArray
    reference: FloatArray
    reward: float
    next_state: FloatArray
    next_reference: FloatArray
    discount: float
    mask: BoolArray | None = None
    bc_target: FloatArray | None = None


@dataclass(frozen=True)
class Batch:
    state: FloatArray
    action: FloatArray
    reference: FloatArray
    reward: FloatArray
    next_state: FloatArray
    next_reference: FloatArray
    discount: FloatArray
    mask: BoolArray
    bc_target: FloatArray | None = None


class ReplayBuffer:
    """Uniform finite replay, shared by VLA, actor, and intervention samples."""

    _array_fields = ("state", "action", "reference", "next_state", "next_reference", "mask")

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self._slots: list[Transition] = []
        self._index = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._slots)

    def add(self, transition: Transition) -> None:
        state = np.array(transition.state, dtype=np.float32, copy=True)
        action = np.array(transition.action, dtype=np.float32, copy=True)
        reference = np.array(transition.reference, dtype=np.float32, copy=True)
        next_state = np.array(transition.next_state, dtype=np.float32, copy=True)
        next_reference = np.array(transition.next_reference, dtype=np.float32, copy=True)
        if state.ndim != 1 or state.shape != next_state.shape:
            raise ValueError("state and next_state must have matching shape (state_dim,)")
        if action.ndim != 2 or action.shape != reference.shape or action.shape != next_reference.shape:
            raise ValueError("action, reference and next_reference must have matching shape (chunk_size, action_dim)")
        mask = (
            np.ones(action.shape[0], dtype=np.bool_)
            if transition.mask is None
            else np.array(transition.mask, dtype=np.bool_, copy=True)
        )
        if mask.shape != (action.shape[0],) or not mask.any():
            raise ValueError("mask must describe at least one valid step in the chunk")
        if not mask.all() and transition.discount != 0:
            raise ValueError("Partial chunks require a terminal transition")

        bc_target = None
        if transition.bc_target is not None:
            bc_target = np.array(transition.bc_target, dtype=np.float32, copy=True)
            if bc_target.shape != action.shape or not np.isfinite(bc_target[mask]).all():
                raise ValueError("BC target must have the action shape and finite values at valid steps")

        item = Transition(
            state, action, reference, float(transition.reward), next_state, next_reference,
            float(transition.discount), mask, bc_target,
        )
        if len(self) < self.capacity:
            self._slots.append(item)
        else:
            self._slots[self._index] = item
        self._index = (self._index + 1) % self.capacity

    def sample(self, batch_size: int) -> Batch:
        if not self._slots or batch_size < 1:
            raise ValueError("Replay sampling requires data and a positive batch size")
        items = [self._slots[i] for i in self._rng.integers(len(self), size=batch_size)]
        reference = np.stack([item.reference for item in items])
        bc_target = reference
        if any(item.bc_target is not None for item in items):
            bc_target = np.stack([item.reference if item.bc_target is None else item.bc_target for item in items])
        return Batch(
            state=np.stack([item.state for item in items]),
            action=np.stack([item.action for item in items]),
            reference=reference,
            reward=np.array([item.reward for item in items], dtype=np.float32),
            next_state=np.stack([item.next_state for item in items]),
            next_reference=np.stack([item.next_reference for item in items]),
            discount=np.array([item.discount for item in items], dtype=np.float32),
            mask=np.stack([item.mask for item in items]),
            bc_target=bc_target,
        )

    def state_dict(self) -> dict:
        """Pack physical ring order and sampling state using weights-only-safe CPU tensors."""
        columns = {}
        if self._slots:
            for field in self._array_fields:
                columns[field] = torch.from_numpy(np.stack([getattr(item, field) for item in self._slots]))
            for field in ("reward", "discount"):
                values = np.fromiter((getattr(item, field) for item in self._slots), np.float64, len(self))
                columns[field] = torch.from_numpy(values)
            expert_rows = [i for i, item in enumerate(self._slots) if item.bc_target is not None]
            if expert_rows:
                columns["expert_rows"] = torch.tensor(expert_rows, dtype=torch.int64)
                columns["bc_target"] = torch.from_numpy(np.stack([self._slots[i].bc_target for i in expert_rows]))
        return {
            "format": 1, "capacity": self.capacity, "count": len(self), "index": self._index,
            "rng_state": json.dumps(self._rng.bit_generator.state), "columns": columns,
        }

    def load_state_dict(self, snapshot: dict) -> None:
        """Restore the ring without copying the loaded packed arrays a second time."""
        if snapshot["format"] != 1 or snapshot["capacity"] != self.capacity:
            raise ValueError("Replay checkpoint format or capacity differs from the buffer")
        count, index = snapshot["count"], snapshot["index"]
        if not 0 <= count <= self.capacity or not 0 <= index < self.capacity or (
            count < self.capacity and index != count
        ):
            raise ValueError("Replay checkpoint has invalid ring counters")
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(snapshot["rng_state"])

        slots = []
        columns = snapshot["columns"]
        if count:
            state, action, reference, next_state, next_reference, mask = (
                columns[field].detach().numpy() for field in self._array_fields
            )
            reward = columns["reward"].detach().numpy()
            discount = columns["discount"].detach().numpy()
            float_arrays = (state, action, reference, next_state, next_reference)
            if (
                any(array.dtype != np.float32 for array in float_arrays)
                or state.ndim != 2 or state.shape[0] != count or state.shape != next_state.shape
                or action.ndim != 3 or action.shape[0] != count
                or action.shape != reference.shape or action.shape != next_reference.shape
                or mask.shape != action.shape[:2] or mask.dtype != np.bool_
                or reward.shape != (count,) or reward.dtype != np.float64
                or discount.shape != (count,) or discount.dtype != np.float64
            ):
                raise ValueError("Replay checkpoint contains incompatible transition arrays")

            expert_targets = {}
            if "expert_rows" in columns or "bc_target" in columns:
                rows = columns["expert_rows"].detach().numpy()
                targets = columns["bc_target"].detach().numpy()
                if (
                    rows.ndim != 1 or rows.dtype != np.int64
                    or np.any((rows < 0) | (rows >= count)) or np.any(rows[1:] <= rows[:-1])
                    or targets.dtype != np.float32 or targets.shape != (len(rows), *action.shape[1:])
                ):
                    raise ValueError("Replay checkpoint contains incompatible expert BC labels")
                expert_targets = dict(zip(rows.tolist(), targets, strict=True))
            slots = [
                Transition(state[i], action[i], reference[i], float(reward[i]), next_state[i], next_reference[i],
                           float(discount[i]), mask[i], expert_targets.get(i))
                for i in range(count)
            ]
        elif columns:
            raise ValueError("Replay checkpoint marked empty contains transition arrays")

        self._slots, self._index, self._rng = slots, index, rng


def _mlp(in_dim: int, out_dim: int, config: OnlineConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(config.hidden_layers):
        layers.extend((nn.Linear(in_dim, config.hidden_dim), nn.ReLU()))
        in_dim = config.hidden_dim
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class Learner:
    def __init__(
        self,
        state_dim: int,
        chunk_size: int,
        action_dim: int,
        config: OnlineConfig = OnlineConfig(),
        device: str | torch.device = "cpu",
        action_projection: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> None:
        if min(state_dim, chunk_size, action_dim) < 1:
            raise ValueError("State and action dimensions and chunk size must be positive")
        self.config = config
        self.device = torch.device(device)
        self.action_projection = action_projection
        self.state_dim = state_dim
        self.chunk_shape = (chunk_size, action_dim)
        action_width = chunk_size * action_dim
        self.actor = _mlp(state_dim + action_width, action_width, config).to(device)
        self.critics = nn.ModuleList(
            [_mlp(state_dim + action_width, 1, config) for _ in range(2)]
        ).to(device)
        self.target_critics = deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.critic_lr)
        self.critic_updates = 0
        self.actor_updates = 0

    def _sample_action(self, state: Tensor, reference: Tensor, deterministic: bool = False) -> Tensor:
        mean = self.actor(torch.cat((state, reference.flatten(1)), dim=1)).reshape_as(reference)
        if deterministic:
            return mean
        return mean + self.config.action_std * torch.randn_like(mean)

    def _executed(self, state: Tensor, actions: Tensor) -> Tensor:
        if self.action_projection is None:
            return actions
        return self.action_projection(state, actions)

    @torch.no_grad()
    def act(self, state: FloatArray, reference: FloatArray, deterministic: bool = False) -> FloatArray:
        if state.shape != (self.state_dim,) or reference.shape != self.chunk_shape:
            raise ValueError("Unexpected state or reference shape for action inference")
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
        ref_tensor = torch.as_tensor(reference, dtype=torch.float32, device=self.device)[None]
        sample = self._sample_action(state_tensor, ref_tensor, deterministic)
        return self._executed(state_tensor, sample)[0].cpu().numpy()

    def update(self, batch: Batch) -> dict[str, float]:
        if batch.bc_target is not None and batch.bc_target.shape != batch.reference.shape:
            raise ValueError("BC target and reference batches must have matching shapes")
        state = torch.as_tensor(batch.state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch.action, dtype=torch.float32, device=self.device)
        reference = torch.as_tensor(batch.reference, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(batch.next_state, dtype=torch.float32, device=self.device)
        next_reference = torch.as_tensor(batch.next_reference, dtype=torch.float32, device=self.device)
        discount = torch.as_tensor(batch.discount, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(batch.mask, dtype=torch.bool, device=self.device)[..., None]

        with torch.no_grad():
            next_action = self._executed(next_state, self._sample_action(next_state, next_reference))
            next_input = torch.cat((next_state, next_action.flatten(1)), dim=1)
            q1_next, q2_next = (critic(next_input).squeeze(-1) for critic in self.target_critics)
            bootstrap = torch.minimum(q1_next, q2_next).masked_fill(discount == 0, 0)
            target = reward + discount * bootstrap

        action = action.masked_fill(~mask, 0)
        critic_input = torch.cat((state, action.flatten(1)), dim=1)
        critic_loss = sum(F.mse_loss(critic(critic_input).squeeze(-1), target) for critic in self.critics)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Critic loss became nonfinite")
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_updates += 1
        metrics = {"critic_loss": critic_loss.detach().item()}

        if self.critic_updates % self.config.actor_delay == 0:
            bc_target = None if batch.bc_target is None else torch.as_tensor(
                batch.bc_target, dtype=torch.float32, device=self.device,
            )
            metrics.update(self._update_actor(state, reference, mask, bc_target))
        return metrics

    def _update_actor(self, state: Tensor, reference: Tensor, mask: Tensor,
                      bc_target: Tensor | None = None) -> dict[str, float]:
        reference = reference.masked_fill(~mask, 0)
        target = reference if bc_target is None else bc_target
        drop_reference = torch.rand((state.shape[0], 1, 1), device=self.device) < self.config.reference_dropout
        actor_input_ref = reference.masked_fill(drop_reference, 0)
        sample = self._sample_action(state, actor_input_ref)
        action = self._executed(state, sample).masked_fill(~mask, 0)
        critic_input = torch.cat((state, action.flatten(1)), dim=1)
        self.critics.requires_grad_(False)
        try:
            q = self.critics[0](critic_input).squeeze(-1)
            difference = (sample - target).masked_fill(~mask, 0)
            bc_loss = difference.square().flatten(1).sum(1).mean()
            actor_loss = -q.mean() + self.config.beta * bc_loss
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("Actor loss became nonfinite")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critics.requires_grad_(True)

        with torch.no_grad():
            for target, source in zip(self.target_critics.parameters(), self.critics.parameters(), strict=True):
                target.lerp_(source, self.config.tau)
        self.actor_updates += 1
        return {"actor_loss": actor_loss.detach().item(), "bc_loss": bc_loss.detach().item()}

    def state_dict(self) -> dict:
        """Learner state for policy export or continuation with externally managed replay."""
        return {
            "config": asdict(self.config),
            "state_dim": self.state_dim,
            "chunk_size": self.chunk_shape[0],
            "action_dim": self.chunk_shape[1],
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic_updates": self.critic_updates,
            "actor_updates": self.actor_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["config"] != asdict(self.config):
            raise ValueError("Saved configuration differs from the learner configuration")
        if (state["state_dim"], state["chunk_size"], state["action_dim"]) != (self.state_dim, *self.chunk_shape):
            raise ValueError("Saved state and action dimensions differ from the learner dimensions")
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.target_critics.load_state_dict(state["target_critics"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.critic_updates = state["critic_updates"]
        self.actor_updates = state["actor_updates"]

import unittest

import numpy as np
import torch

from rlt.online import Batch, Learner, OnlineConfig, ReplayBuffer, Transition


class OnlineTests(unittest.TestCase):
    def test_chunk_target_uses_lower_critic_and_stops_at_terminal(self):
        learner = Learner(1, 2, 1, OnlineConfig(hidden_dim=4, action_std=0))
        with torch.no_grad():
            for network in (learner.critics, learner.target_critics):
                for parameter in network.parameters():
                    parameter.zero_()
            learner.target_critics[0][-1].bias.fill_(8)
            learner.target_critics[1][-1].bias.fill_(4)

        state = np.zeros((2, 1), dtype=np.float32)
        next_state = state.copy()
        next_state[0] = np.nan
        action = np.zeros((2, 2, 1), dtype=np.float32)
        batch = Batch(
            state, action, action, np.ones(2, dtype=np.float32), next_state, action,
            np.array([0, 0.5], dtype=np.float32), np.ones((2, 2), dtype=np.bool_),
        )
        first = learner.update(batch)
        self.assertEqual(first, {"critic_loss": 10.0})
        second = learner.update(batch)
        self.assertIn("actor_loss", second)
        self.assertEqual((learner.critic_updates, learner.actor_updates), (2, 1))

    def test_dropout_preserves_bc_labels_and_partial_mask(self):
        config = OnlineConfig(hidden_dim=4, reference_dropout=1, action_std=0, actor_delay=1)
        learner = Learner(1, 2, 1, config)
        with torch.no_grad():
            for parameter in learner.actor.parameters():
                parameter.zero_()
            learner.actor[-1].bias.copy_(torch.tensor([2.0, 9.0]))
        state = np.zeros(1, dtype=np.float32)
        action = np.array([[3], [999]], dtype=np.float32)
        reference = np.array([[1], [50]], dtype=np.float32)
        replay = ReplayBuffer(capacity=1)
        replay.add(Transition(state, action, reference, 0, state, reference, 0, np.array([True, False])))

        actor_inputs, critic_inputs = [], []
        actor_hook = learner.actor.register_forward_pre_hook(
            lambda _module, args: actor_inputs.append(args[0].detach().clone())
        )
        critic_hook = learner.critics[0].register_forward_pre_hook(
            lambda _module, args: critic_inputs.append(args[0].detach().clone())
        )
        try:
            metrics = learner.update(replay.sample(1))
            learner.act(state, reference, deterministic=True)
        finally:
            actor_hook.remove()
            critic_hook.remove()

        self.assertEqual(metrics["bc_loss"], 1.0)
        self.assertEqual([entry.tolist() for entry in actor_inputs], [
            [[0, 1, 50]], [[0, 0, 0]], [[0, 1, 50]],
        ])
        self.assertEqual([entry.tolist() for entry in critic_inputs], [
            [[0, 3, 0]], [[0, 2, 0]],
        ])


    def test_q_uses_applied_action_and_bc_uses_gaussian_sample(self):
        learner = Learner(
            1, 2, 1, OnlineConfig(hidden_dim=4, action_std=0, actor_delay=1),
            action_projection=lambda state, actions: actions.clamp(-0.5, 0.5),
        )
        with torch.no_grad():
            for parameter in learner.actor.parameters():
                parameter.zero_()
            learner.actor[-1].bias.copy_(torch.tensor([2.0, -3.0]))
        state = np.zeros(1, dtype=np.float32)
        reference = np.zeros((2, 1), dtype=np.float32)
        replay = ReplayBuffer(1)
        replay.add(Transition(state, reference, reference, 0, state, reference, 0.5))

        targets, current = [], []
        target_hook = learner.target_critics[0].register_forward_pre_hook(
            lambda _module, args: targets.append(args[0].tolist())
        )
        critic_hook = learner.critics[0].register_forward_pre_hook(
            lambda _module, args: current.append(args[0].tolist())
        )
        try:
            metrics = learner.update(replay.sample(1))
        finally:
            target_hook.remove()
            critic_hook.remove()

        self.assertEqual(targets, [[[0.0, 0.5, -0.5]]])
        self.assertEqual(current[-1], [[0.0, 0.5, -0.5]])
        self.assertEqual(metrics["bc_loss"], 13.0)
        np.testing.assert_array_equal(
            learner.act(state, reference, deterministic=True), [[0.5], [-0.5]],
        )


if __name__ == "__main__":
    unittest.main()

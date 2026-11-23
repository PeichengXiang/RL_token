import unittest

import numpy as np
import torch

from rlt.rollout import Observation, RolloutConfig, run_episode
from rlt.token import RLToken, TokenConfig


class SmallEnv:
    action_dim = 1
    proprio_dim = 1

    def _observation(self):
        position = np.array([self.step_index], dtype=np.float32)
        return Observation({"state": position}, position)

    def reset(self, seed):
        self.step_index = 0
        return self._observation()

    def step(self, action):
        self.step_index += 1
        terminal = self.step_index == 9
        return self._observation(), float(terminal), terminal, False, {
            "success": terminal, "steps": self.step_index, "action": action.copy(),
        }


class SmallVLA:
    def reset(self, seed):
        pass

    def reference(self, observation):
        return observation["state"] + np.arange(1, 7, dtype=np.float32)[:, None]

    def infer(self, observation):
        return (
            np.array([[0, 1]], dtype=np.float32),
            np.array([True]),
            self.reference(observation),
        )


class StageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.model = RLToken(TokenConfig(2, 8, 8, 1, 1, 2, 2))

    def test_causal_teacher_forcing_and_encoder_stop_gradient(self):
        features = torch.randn(2, 4, 2, requires_grad=True)
        valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
        token = torch.randn(2, 8)
        changed = features.detach().clone()
        changed[0, 2] += 10
        reference = self.model.decode(token, features, valid)
        actual = self.model.decode(token, changed, valid)
        torch.testing.assert_close(reference[0, :3], actual[0, :3])
        self.assertFalse(torch.allclose(reference[0, 3], actual[0, 3]))
        torch.testing.assert_close(
            self.model.encode(features[1:2], valid[1:2]),
            self.model.encode(features[1:2, :2], valid[1:2, :2]),
        )
        loss, _ = self.model.reconstruction_loss(features, valid)
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertGreater(self.model.readout.grad.abs().max().item(), 0)

    def test_stride_handoff_terminal_and_partial_intervention(self):
        frames = []
        config = RolloutConfig(chunk_size=4, stride=2, base_chunk_size=6, gamma=0.5)
        result = run_episode(
            SmallEnv(), SmallVLA(), self.model.eval(), config, 0,
            phase=lambda obs, info: info.get("steps") == 2,
            intervention=lambda obs, action: np.array([20]) if obs.proprio[0] == 5 else None,
            on_transitions=lambda transitions, steps: frames.extend(transitions),
            action_offset=lambda obs: obs.proprio,
        )
        self.assertEqual((result.steps, result.critical_steps, result.transitions), (9, 7, 4))
        self.assertEqual([t.state[-1] for t in frames], [2, 4, 6, 8])
        self.assertEqual([t.discount for t in frames], [0.5**4, 0.5**4, 0, 0])
        self.assertEqual([t.reward for t in frames], [0, 0, 0.5**2, 1])
        np.testing.assert_array_equal(frames[1].action[:, 0], [1, 16, 3, 4])
        np.testing.assert_array_equal(frames[1].reference[:, 0], [1, 16, 3, 4])
        np.testing.assert_array_equal(frames[1].next_reference[:, 0], [1, 2, 3, 4])
        np.testing.assert_array_equal(frames[-1].mask, [True, False, False, False])


if __name__ == "__main__":
    unittest.main()

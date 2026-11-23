"""Run with the OpenPI runtime to check shared-prefix flow sampling."""

from importlib.util import find_spec
from types import SimpleNamespace
import unittest

import numpy as np

from rlt.openpi import _PrefixCapture


@unittest.skipUnless(find_spec("openpi") and find_spec("jax"), "OpenPI/JAX not installed")
class OpenPIPrefixTests(unittest.TestCase):
    def test_shared_prefix_matches_native_sampler(self):
        import jax
        import jax.numpy as jnp
        from openpi.models import model, pi0

        class TinyModel:
            action_dim = 2
            action_horizon = 3

            def __init__(self):
                self.PaliGemma = SimpleNamespace(llm=self.language_model)

            def embed_prefix(self, observation):
                hidden = jnp.stack((observation.state, observation.state + 1), axis=1)
                return hidden, jnp.array([[True, False]]), jnp.zeros(2, dtype=bool)

            def embed_suffix(self, observation, actions, time):
                mask = jnp.ones(actions.shape[:-1], dtype=bool)
                return actions, mask, jnp.array([True, False, False]), time

            def action_out_proj(self, suffix):
                return suffix

            def language_model(self, tokens, *, mask, positions, kv_cache=None, adarms_cond=None):
                prefix, suffix = tokens
                if prefix is not None:
                    hidden = prefix + (positions[:, :, None] + 1) * 0.2
                    return (hidden, None), hidden[:, 0]
                denoised = jnp.tanh(suffix + kv_cache[:, None] + adarms_cond[1][:, None, None])
                return (None, denoised), kv_cache

        tiny = TinyModel()
        observation = model.Observation(
            images={k: jnp.zeros((1, 224, 224, 3), dtype=jnp.float32) for k in model.IMAGE_KEYS},
            image_masks={k: jnp.ones((1,), dtype=bool) for k in model.IMAGE_KEYS},
            state=jnp.array([[0.3, -0.1]]),
            tokenized_prompt_mask=None,
            tokenized_prompt=None,
        )
        rng = jax.random.key(991)

        @jax.jit
        def native(key, obs, num_steps):
            return pi0.Pi0.sample_actions(tiny, key, obs, num_steps=num_steps)

        @jax.jit
        def shared(key, obs, num_steps):
            captured = _PrefixCapture(tiny)
            actions = pi0.Pi0.sample_actions(captured, key, obs, num_steps=num_steps)
            return captured.hidden, captured.valid, actions

        hidden, valid, actions = shared(rng, observation, 3)
        prefix, expected_valid, ar = tiny.embed_prefix(observation)
        (expected_hidden, _), _ = tiny.PaliGemma.llm(
            [prefix, None],
            mask=pi0.make_attn_mask(expected_valid, ar),
            positions=jnp.cumsum(expected_valid, axis=1) - 1,
        )
        np.testing.assert_array_equal(expected_valid, valid)
        np.testing.assert_allclose(expected_hidden, hidden)
        np.testing.assert_array_equal(native(rng, observation, 3), actions)


if __name__ == "__main__":
    unittest.main()

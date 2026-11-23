"""Frozen JAX OpenPI π0/π0.5 with a readout of final PaliGemma prefix states."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


_POLICY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIANJI_ADAPTER = _POLICY_ROOT / "Pi_05"
DEFAULT_XPOLICYLAB = _POLICY_ROOT.parent / "insert_benchmark/third_party/XPolicyLab"


class _PrefixUnavailable(RuntimeError):
    pass


class _PrefixCapture:
    """Capture the prefix from Pi0.sample_actions during JAX tracing."""

    def __init__(self, vla):
        self._vla = vla
        self.PaliGemma = SimpleNamespace(llm=self._llm)
        self.hidden = self.valid = None

    def __getattr__(self, name):
        return getattr(self._vla, name)

    def embed_prefix(self, observation):
        tokens, self.valid, ar_mask = self._vla.embed_prefix(observation)
        return tokens, self.valid, ar_mask

    def _llm(self, tokens, **kwargs):
        hidden, cache = self._vla.PaliGemma.llm(tokens, **kwargs)
        if (
            isinstance(tokens, (list, tuple)) and len(tokens) == 2
            and tokens[0] is not None and tokens[1] is None
        ):
            self.hidden = hidden[0]
        return hidden, cache


def _add_import_path(path: Path) -> None:
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)


def _add_openpi_source(openpi_root: Path | str | None) -> None:
    if openpi_root is None:
        return
    root = Path(openpi_root).expanduser().resolve()
    if not (root / "src/openpi/models/pi0.py").is_file():
        raise FileNotFoundError(f"OpenPI JAX source not found at {root}")
    _add_import_path(root / "src")
    _add_import_path(root / "packages/openpi-client/src")


def _load_tianji_module(name: str, adapter_dir: Path | str, xpolicylab_dir: Path | str):
    adapter = Path(adapter_dir).expanduser().resolve()
    workspace = Path(xpolicylab_dir).expanduser().resolve()
    if not (adapter / f"{name}.py").is_file():
        raise FileNotFoundError(f"Tianji Pi_05 adapter module {name} not found at {adapter}")
    if not (workspace / "utils/process_data.py").is_file():
        raise FileNotFoundError(f"XPolicyLab workspace not found at {workspace}")
    _add_import_path(adapter.parent)
    _add_import_path(workspace.parent)
    return import_module(f"{adapter.name}.{name}")


def prepare_tianji_observation(
    observation: dict,
    *,
    adapter_dir: Path | str = DEFAULT_TIANJI_ADAPTER,
    xpolicylab_dir: Path | str = DEFAULT_XPOLICYLAB,
) -> dict:
    """Apply the existing adapter to an XPolicyLab observation with decoded RGB images."""
    preprocessing = _load_tianji_module("preprocessing", adapter_dir, xpolicylab_dir)
    return preprocessing.encode_observation(observation, {"arm_dim": [7, 7], "ee_dim": [1, 1]})


class FrozenOpenPI:
    """Single-observation adapter using a single frozen JAX model per instance.

    Observations follow the loaded OpenPI policy's pre-transform schema. The RLT
    reader retains visual tokens by default. Prompt and discrete state still
    participate in the original prefix attention. With ``include_language=True``,
    the original nonvisual prefix positions are also returned with their mask.
    ``shared_prefix=True`` reuses the native sampler's prefix calculation in
    ``infer`` under a single JAX compilation.
    """

    def __init__(self, policy, *, include_language: bool = False, seed: int = 0, shared_prefix: bool = False):
        from flax import nnx
        import jax
        import jax.numpy as jnp
        from openpi.models import model
        from openpi.models import pi0
        from openpi.shared import nnx_utils

        if policy._is_pytorch_model or not isinstance(policy._model, pi0.Pi0):
            raise TypeError("The RLT readout requires a JAX OpenPI π0/π0.5 policy")

        class PrefixReader(nnx.Module):
            def __init__(self, vla):
                self.vla = vla

            def read(self, observation):
                observation = model.preprocess_observation(None, observation, train=False)
                tokens, valid, ar_mask = self.vla.embed_prefix(observation)
                hidden, _ = self.vla.PaliGemma.llm(
                    [tokens, None],
                    mask=pi0.make_attn_mask(valid, ar_mask),
                    positions=jnp.cumsum(valid, axis=1) - 1,
                )
                return self.select_prefix(hidden[0], valid, observation)

            def read_and_sample(self, sample_rng, observation, **sample_kwargs):
                capture = _PrefixCapture(self.vla)
                # Use the native sampler, including its exact denoising loop
                # and KV cache. The temporary proxy never mutates the model.
                actions = pi0.Pi0.sample_actions(capture, sample_rng, observation, **sample_kwargs)
                if capture.hidden is None or capture.valid is None:
                    raise _PrefixUnavailable("The OpenPI sampler did not expose a standalone prefix")
                hidden, valid = self.select_prefix(capture.hidden, capture.valid, observation)
                return hidden, valid, actions

            @staticmethod
            def select_prefix(hidden, valid, observation):
                if not include_language and observation.tokenized_prompt is not None:
                    visual_length = valid.shape[1] - observation.tokenized_prompt.shape[-1]
                    return hidden[:, :visual_length], valid[:, :visual_length]
                return hidden, valid

        self._policy = policy
        self._jax = jax
        self._jnp = jnp
        self._observation_type = model.Observation
        reader = PrefixReader(policy._model)
        self._read_prefix = nnx_utils.module_jit(reader.read)
        self._read_and_sample = (
            nnx_utils.module_jit(reader.read_and_sample) if shared_prefix and type(policy._model) is pi0.Pi0 else None
        )
        self.action_horizon = policy._model.action_horizon
        self.include_language = bool(include_language)
        self.seed = int(seed)
        self.reset()

    @classmethod
    def from_config(
        cls,
        config_name: str,
        checkpoint: Path | str,
        *,
        openpi_root: Path | str | None = None,
        include_language: bool = False,
        num_steps: int = 10,
        seed: int = 0,
        shared_prefix: bool = False,
    ) -> FrozenOpenPI:
        """Load a named OpenPI training config and a local Orbax policy checkpoint."""
        _add_openpi_source(openpi_root)
        from openpi.training import config

        return cls._from_training_config(
            config.get_config(config_name), Path(checkpoint).expanduser().resolve(),
            include_language=include_language, num_steps=num_steps, seed=seed, shared_prefix=shared_prefix,
        )

    @classmethod
    def from_tianji(
        cls,
        checkpoint: Path | str,
        *,
        adapter_dir: Path | str = DEFAULT_TIANJI_ADAPTER,
        xpolicylab_dir: Path | str = DEFAULT_XPOLICYLAB,
        include_language: bool = False,
        num_steps: int = 10,
        seed: int = 0,
        shared_prefix: bool = False,
    ) -> FrozenOpenPI:
        """Load the sibling Pi_05 adapter's actual Tianji train and transform contract."""
        root = Path(checkpoint).expanduser().resolve()
        if not (root / "training_config.json").is_file():
            raise FileNotFoundError(f"Tianji training_config.json not found at {root}")
        _add_openpi_source(Path(adapter_dir) / "openpi")
        adapter = _load_tianji_module("model", adapter_dir, xpolicylab_dir)
        contract = adapter.checkpoint_contract(root)
        return cls._from_training_config(
            adapter.training_config(root, contract), root,
            include_language=include_language, num_steps=num_steps, seed=seed, shared_prefix=shared_prefix,
        )

    @classmethod
    def _from_training_config(cls, config, checkpoint, *, include_language, num_steps, seed, shared_prefix):
        from openpi.models import model
        from openpi.policies import policy_config

        if config.model.model_type not in (model.ModelType.PI0, model.ModelType.PI05):
            raise TypeError("The RLT readout supports π0 and π0.5")
        if (checkpoint / "model.safetensors").is_file() or not (checkpoint / "params").is_dir():
            raise FileNotFoundError(f"Expected a JAX Orbax params/ checkpoint at {checkpoint}")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        policy = policy_config.create_trained_policy(
            config, checkpoint, sample_kwargs={"num_steps": num_steps},
        )
        return cls(policy, include_language=include_language, seed=seed, shared_prefix=shared_prefix)

    def reset(self, seed: int | None = None) -> None:
        """Reset the reference action sampler at an episode boundary."""
        self._rng = self._jax.random.key(self.seed if seed is None else int(seed))

    def _prepare(self, observation):
        inputs = {key: value for key, value in observation.items() if key != "actions"}
        inputs = self._policy._input_transform(self._jax.tree.map(lambda value: value, inputs))
        if np.ndim(inputs["state"]) != 1:
            raise ValueError("Supply one OpenPI observation at a time")
        inputs = self._jax.tree.map(lambda value: self._jnp.asarray(value)[None, ...], inputs)
        return self._observation_type.from_dict(inputs)

    def _embeddings(self, observation):
        return self._format_embeddings(*self._read_prefix(observation))

    @staticmethod
    def _format_embeddings(hidden, mask):
        hidden = np.asarray(hidden[0], dtype=np.float32)
        mask = np.asarray(mask[0], dtype=bool)
        if not mask.any() or not np.isfinite(hidden[mask]).all():
            raise ValueError("OpenPI returned no finite valid prefix tokens")
        return np.where(mask[:, None], hidden, 0.0), mask

    def _next_sample_rng(self):
        self._rng, sample_rng = self._jax.random.split(self._rng)
        return sample_rng

    def _reference(self, observation, sample_rng):
        actions = self._policy._sample_actions(sample_rng, observation, **self._policy._sample_kwargs)
        return self._format_actions(observation, actions)

    def _format_actions(self, observation, actions):
        output = self._policy._output_transform({
            "state": np.asarray(observation.state[0]),
            "actions": np.asarray(actions[0]),
        })
        actions = np.asarray(output["actions"], dtype=np.float32)
        if actions.ndim != 2 or len(actions) != self.action_horizon or not np.isfinite(actions).all():
            raise ValueError(f"OpenPI returned an invalid reference action chunk of shape {actions.shape}")
        return actions

    def embeddings(self, observation: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return final prefix hidden states [S, D] and a boolean validity mask [S]."""
        return self._embeddings(self._prepare(observation))

    def reference(self, observation: dict) -> np.ndarray:
        """Return [H, A] from OpenPI's unchanged sampler and output transforms."""
        return self._reference(self._prepare(observation), self._next_sample_rng())

    def infer(self, observation: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return frozen prefix states, their validity mask, and an action chunk."""
        prepared = self._prepare(observation)
        sample_rng = self._next_sample_rng()
        if self._read_and_sample is not None:
            try:
                hidden, valid, actions = self._read_and_sample(sample_rng, prepared, **self._policy._sample_kwargs)
            except _PrefixUnavailable:
                self._read_and_sample = None
            else:
                embeddings, mask = self._format_embeddings(hidden, valid)
                return embeddings, mask, self._format_actions(prepared, actions)
        embeddings, mask = self._embeddings(prepared)
        return embeddings, mask, self._reference(prepared, sample_rng)

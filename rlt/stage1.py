"""Cache demonstration features and optimize the RL token autoencoder."""

from pathlib import Path

import numpy as np
import torch

from .token import RLToken


def write_features(directory: Path, vla, observations, *, limit=None):
    if limit is not None and limit < 1:
        raise ValueError("Feature cache limit must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError(f"Choose an empty feature cache directory: {directory}")
    count = 0
    for index, observation in enumerate(observations):
        features, valid = vla.embeddings(observation)
        np.savez(directory / f"{index:07}.npz", features=features.astype(np.float16), valid=valid)
        count += 1
        if limit is not None and count == limit:
            break
    if not count:
        raise ValueError("No demonstration frames found")
    return count


def read_features(path):
    with np.load(path, allow_pickle=False) as data:
        return data["features"].astype(np.float32), data["valid"].astype(bool)


def make_batch(samples, device):
    length = max(len(features) for features, valid in samples)
    width = samples[0][0].shape[1]
    features_batch = np.zeros((len(samples), length, width), dtype=np.float32)
    validity_batch = np.zeros((len(samples), length), dtype=bool)
    for i, (features, valid) in enumerate(samples):
        if features.shape != (len(valid), width) or not valid.any() or not np.isfinite(features[valid]).all():
            raise ValueError("Cached VLA features or validity are malformed")
        features_batch[i, :len(features)] = features
        validity_batch[i, :len(valid)] = valid
    return torch.from_numpy(features_batch).to(device), torch.from_numpy(validity_batch).to(device)


def train_token(
    model: RLToken, get_sample, count: int, *, steps: int, batch_size: int,
    lr: float, seed: int, report=None,
):
    if min(count, batch_size, steps) < 1 or lr <= 0:
        raise ValueError("Training needs positive frame count, batch size, steps and learning rate")
    device = model.readout.device
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for step in range(1, steps + 1):
        indices = rng.choice(count, size=batch_size, replace=batch_size > count)
        features, valid = make_batch([get_sample(index) for index in indices], device)
        loss, mse = model.reconstruction_loss(features, valid)
        if not torch.isfinite(loss):
            raise FloatingPointError("RL token reconstruction loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if report is not None and (step == 1 or step % 100 == 0 or step == steps):
            report({"step": step, "loss": float(loss.detach()), "mse": float(mse)})


@torch.no_grad()
def mean_reconstruction_mse(model, samples):
    features, valid = make_batch(samples, model.readout.device)
    model.eval()
    return float(model.reconstruction_loss(features, valid)[1])

"""Read out a single token and reconstruct frozen VLA features (paper §IV-A)."""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TokenConfig:
    feature_dim: int
    token_dim: int = 2048
    max_tokens: int = 1024
    encoder_layers: int = 2
    decoder_layers: int = 2
    heads: int = 8
    ff_multiplier: int = 4

    def __post_init__(self):
        if min(asdict(self).values()) < 1 or self.token_dim % self.heads:
            raise ValueError("Token dimensions and layer counts must be positive, and token_dim divisible by heads")


class RLToken(nn.Module):
    def __init__(self, config: TokenConfig):
        super().__init__()
        self.config = config
        width = config.token_dim
        self.input_projection = (
            nn.Identity() if config.feature_dim == width else nn.Linear(config.feature_dim, width)
        )
        self.readout = nn.Parameter(torch.empty(1, 1, width))
        self.encoder_position = nn.Parameter(torch.empty(1, config.max_tokens + 1, width))
        self.decoder_position = nn.Parameter(torch.empty(1, config.max_tokens, width))
        self.encoder = self._transformer(config.encoder_layers)
        self.decoder = self._transformer(config.decoder_layers)
        self.output = nn.Linear(width, config.feature_dim)
        for embedding in (self.readout, self.encoder_position, self.decoder_position):
            nn.init.normal_(embedding, std=0.02)

    def _transformer(self, depth):
        c = self.config
        layer = nn.TransformerEncoderLayer(
            c.token_dim, c.heads, c.ff_multiplier * c.token_dim,
            dropout=0, activation="gelu", batch_first=True, norm_first=True,
        )
        transformer = nn.TransformerEncoder(
            layer, depth, norm=nn.LayerNorm(c.token_dim), enable_nested_tensor=False,
        )
        for block in transformer.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        return transformer

    def _features(self, features, valid):
        if features.ndim != 3 or features.shape[-1] != self.config.feature_dim:
            raise ValueError("Expected VLA features with shape [batch, sequence, feature_dim]")
        if valid.shape != features.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Expected a boolean validity mask with shape [batch, sequence]")
        if not 0 < features.shape[1] <= self.config.max_tokens:
            raise ValueError("VLA sequence exceeds configured length or is empty")
        # Invalid camera or text slots must not leak through keys, targets or
        # the teacher-forcing history. Valid positions keep their original order.
        return features.detach().masked_fill(~valid[..., None], 0)

    def encode(self, features, valid):
        features = self._features(features, valid)
        return self._encode(features, valid)

    def _encode(self, features, valid):
        batch, length, _ = features.shape
        # In each row insert <rl> directly after the last valid source token.
        # This makes its position invariant to trailing padding from batching.
        positions = torch.arange(length, device=features.device)
        end = torch.where(valid, positions + 1, 0).amax(dim=1)
        readout_position = torch.arange(length + 1, device=features.device) == end[:, None]
        padding = ~torch.cat((valid, valid.new_zeros((batch, 1))), dim=1)
        projected = self.input_projection(features)
        sequence = torch.cat((projected, projected.new_zeros((batch, 1, projected.shape[-1]))), dim=1)
        sequence = torch.where(readout_position[..., None], self.readout, sequence)
        encoded = self.encoder(
            sequence + self.encoder_position[:, :length + 1],
            src_key_padding_mask=padding & ~readout_position,
        )
        return encoded[torch.arange(batch, device=features.device), end]

    def decode(self, token, features, valid):
        features = self._features(features, valid)
        return self._decode(token, features, valid)

    def _decode(self, token, features, valid):
        length = features.shape[1]
        prefix = torch.cat((token[:, None], self.input_projection(features[:, :-1])), dim=1)
        prefix_valid = torch.cat((valid.new_ones((len(valid), 1)), valid[:, :-1]), dim=1)
        causal = torch.ones((length, length), dtype=torch.bool, device=features.device).triu(1)
        decoded = self.decoder(
            prefix + self.decoder_position[:, :length],
            mask=causal, is_causal=True, src_key_padding_mask=~prefix_valid,
        )
        return self.output(decoded)

    def reconstruction_loss(self, features, valid):
        features = self._features(features, valid)
        reconstruction = self._decode(self._encode(features, valid), features, valid)
        residual = (reconstruction.float() - features.float()).masked_fill(~valid[..., None], 0)
        squared_error = residual.square()
        # Equation 2 sums all valid token coordinates per example.
        loss = squared_error.sum(dim=(1, 2)).mean()
        return loss, squared_error.sum().detach() / (valid.sum().clamp_min(1) * features.shape[-1])


def save_token(path, model):
    torch.save({"config": asdict(model.config), "model": model.state_dict()}, path)


def load_token(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = RLToken(TokenConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval().requires_grad_(False)

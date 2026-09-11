"""Six compact wearable time-series baselines; deliberately EEG-independent."""

from __future__ import annotations

import torch
from torch import nn


class _Head(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        return self.out(self.norm(x.mean(dim=1))).squeeze(-1)


class TTM(nn.Module):
    """Tiny temporal mixer: token and channel mixing with edge-sized layers."""
    def __init__(self, channels=3, hidden=32):
        super().__init__()
        self.proj = nn.Linear(channels, hidden)
        self.mix = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.head = _Head(hidden)

    def forward(self, x, secondary=None):
        return self.head(self.proj(x) + self.mix(self.proj(x)))


class PatchTST(nn.Module):
    def __init__(self, channels=3, hidden=32, patch=16, heads=4):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(channels * patch, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = _Head(hidden)

    def forward(self, x, secondary=None):
        n = x.size(1) // self.patch
        x = x[:, :n * self.patch].reshape(x.size(0), n, -1)
        return self.head(self.encoder(self.proj(x)))


class ITransformer(nn.Module):
    """Inverted tokenization: each channel is a token over time."""
    def __init__(self, channels=3, length=160, hidden=32, heads=4):
        super().__init__()
        self.proj = nn.Linear(length, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = _Head(hidden)

    def forward(self, x, secondary=None):
        return self.head(self.encoder(self.proj(x.transpose(1, 2))))


class TimesNet(nn.Module):
    def __init__(self, channels=3, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(channels, hidden, 5, padding=2), nn.GELU(), nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        self.head = _Head(hidden)

    def forward(self, x, secondary=None):
        return self.head(self.conv(x.transpose(1, 2)).transpose(1, 2))


class MambaSSM(nn.Module):
    """Dependency-free gated state-space approximation for long sequences."""
    def __init__(self, channels=3, hidden=32):
        super().__init__()
        self.in_proj = nn.Linear(channels, hidden * 2)
        self.state = nn.GRU(hidden, hidden, batch_first=True)
        self.head = _Head(hidden)

    def forward(self, x, secondary=None):
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        state, _ = self.state(value * torch.sigmoid(gate))
        return self.head(state)


class CNNTransformerHybrid(nn.Module):
    """Local convolution plus global attention, with optional late fusion."""
    def __init__(self, channels=3, hidden=32, heads=4):
        super().__init__()
        self.conv = nn.Conv1d(channels, hidden, 5, padding=2)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.fuse = nn.LazyLinear(hidden)
        self.head = _Head(hidden)

    def forward(self, x, secondary=None):
        z = self.encoder(torch.relu(self.conv(x.transpose(1, 2)).transpose(1, 2)))
        if secondary is not None:
            s = secondary.mean(dim=1) if secondary.ndim == 3 else secondary
            z = z + self.fuse(torch.cat([z.mean(dim=1), s], dim=-1)).unsqueeze(1)
        return self.head(z)


MODERN_METHODS = {
    "ttm": TTM,
    "patchtst": PatchTST,
    "itransformer": ITransformer,
    "timesnet": TimesNet,
    "mamba_ssm": MambaSSM,
    "cnn_transformer_hybrid": CNNTransformerHybrid,
}


def build_all(channels=3, length=160):
    return {
        name: (cls(channels=channels, length=length) if name == "itransformer" else cls(channels=channels))
        for name, cls in MODERN_METHODS.items()
    }

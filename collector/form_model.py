"""
Binary squat form classifier for canonical [N,T,12,4] skeleton sequences.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from canonical import CANONICAL_CONNECTIONS, NUM_CHANNELS, NUM_JOINTS


def canonical_adjacency(device=None) -> torch.Tensor:
    adjacency = torch.eye(NUM_JOINTS, dtype=torch.float32, device=device)
    for a, b in CANONICAL_CONNECTIONS:
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inv_sqrt = torch.diag(torch.pow(degree, -0.5))
    return inv_sqrt @ adjacency @ inv_sqrt


class GraphTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.spatial = nn.Linear(in_channels, out_channels)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: [N,T,V,C]. Aggregate neighboring joints, then mix channels.
        aggregated = torch.einsum("vw,ntwc->ntvc", adjacency, x)
        y = self.spatial(aggregated)
        y = y.permute(0, 3, 1, 2).contiguous()
        y = self.temporal(y).permute(0, 2, 3, 1).contiguous()
        return y + self.residual(x)


class FormClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        hidden_channels: tuple[int, int, int] = (64, 96, 128),
        num_classes: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.register_buffer("adjacency", canonical_adjacency())
        channels = (in_channels,) + hidden_channels
        self.blocks = nn.ModuleList(
            GraphTemporalBlock(channels[i], channels[i + 1], dropout=dropout)
            for i in range(len(channels) - 1)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_channels[-1]),
            nn.Linear(hidden_channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [N,T,V,C], got {tuple(x.shape)}")
        for block in self.blocks:
            x = block(x, self.adjacency)
        x = x.mean(dim=(1, 2))
        return self.head(x)


def predict_probabilities(model: nn.Module, sample: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(sample)
        return F.softmax(logits, dim=-1)

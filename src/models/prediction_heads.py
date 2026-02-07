"""Prediction heads for the multi-modal fusion model."""

import torch
import torch.nn as nn


class DiseaseClassificationHead(nn.Module):
    """Disease classification from fused features.

    Input: (batch, fused_dim=640) -> Output: (batch, num_classes)
    """

    def __init__(self, input_dim: int = 640, num_classes: int = 15):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class OutbreakRegressionHead(nn.Module):
    """7-day outbreak risk regression.

    Input: (batch, fused_dim=640) -> Output: (batch, forecast_days=7)
    Each output is a risk score [0, 1] for that day.
    """

    def __init__(self, input_dim: int = 640, forecast_days: int = 7):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, forecast_days),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SeverityHead(nn.Module):
    """Ordinal severity regression (healthy/initial/moderate/severe).

    Uses ordinal encoding: output K-1 cumulative probabilities.
    Input: (batch, fused_dim=640) -> Output: (batch, num_levels)
    """

    def __init__(self, input_dim: int = 640, num_levels: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_levels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

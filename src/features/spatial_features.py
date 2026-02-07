"""Spatial feature encoding via MLP for geographic coordinates."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("solinfitec.spatial_features")


class SpatialMLP(nn.Module):
    """MLP encoder for geographic features (lat, lon, elevation).

    Input: (batch, 3) -> Output: (batch, output_dim=128)

    Args:
        input_dim: Number of spatial features (lat, lon, elevation).
        hidden_dim: Hidden layer dimension.
        output_dim: Output embedding dimension.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

        logger.info(f"SpatialMLP: {input_dim} -> {hidden_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) spatial features [lat, lon, elevation].

        Returns:
            embedding: (batch, output_dim) spatial embedding.
        """
        return self.mlp(x)


def normalize_coordinates(
    lat: torch.Tensor, lon: torch.Tensor, elevation: torch.Tensor
) -> torch.Tensor:
    """Normalize geographic coordinates to [-1, 1] range.

    Args:
        lat: Latitude values (-90 to 90).
        lon: Longitude values (-180 to 180).
        elevation: Elevation in meters (0 to ~8848).

    Returns:
        Normalized (batch, 3) tensor.
    """
    lat_norm = lat / 90.0
    lon_norm = lon / 180.0
    elev_norm = elevation / 5000.0  # Normalize by reasonable max

    return torch.stack([lat_norm, lon_norm, elev_norm], dim=-1)

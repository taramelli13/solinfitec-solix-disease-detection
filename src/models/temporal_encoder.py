"""Temporal Transformer Encoder for IoT time series."""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger("solinfitec.temporal_encoder")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (B, seq_len, d_model)."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    """Transformer Encoder for temporal IoT sequences.

    Input: (batch, seq_len=30, num_features=7)
    Output: (batch, output_dim=256) temporal embedding

    Args:
        num_features: Number of input features per timestep.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
        output_dim: Output embedding dimension.
        seq_length: Expected sequence length.
    """

    def __init__(
        self,
        num_features: int = 7,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_dim: int = 256,
        seq_length: int = 30,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        # Project input features to d_model
        self.input_proj = nn.Linear(num_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length + 10, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
            nn.GELU(),
        )

        logger.info(
            f"TemporalEncoder: features={num_features}, d_model={d_model}, "
            f"heads={nhead}, layers={num_layers}, output={output_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_features) input tensor.

        Returns:
            embedding: (batch, output_dim) temporal embedding.
        """
        # Project to d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encode
        x = self.transformer_encoder(x)  # (B, T, d_model)

        # Global average pooling over time dimension
        x = x.mean(dim=1)  # (B, d_model)

        # Project to output dim
        embedding = self.output_proj(x)  # (B, output_dim)

        return embedding

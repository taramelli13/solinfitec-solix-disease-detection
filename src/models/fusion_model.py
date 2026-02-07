"""Multi-Modal Fusion Model with Cross-Attention and Gated Fusion.

Architecture:
    Swin backbone (frozen) -> proj 768->256
    Temporal Transformer -> 256-d
    Spatial MLP -> 128-d
    Cross-Attention: queries=temporal, keys/values=visual+spatial
    Gated Fusion: sigmoid(W*concat) * proj(concat) -> 640-d
    Three prediction heads in parallel
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from .prediction_heads import (
    DiseaseClassificationHead,
    OutbreakRegressionHead,
    SeverityHead,
)
from .swin_classifier import SwinClassifier
from .temporal_encoder import TemporalTransformerEncoder
from src.features.spatial_features import SpatialMLP

logger = logging.getLogger("solinfitec.fusion_model")


class CrossAttentionFusion(nn.Module):
    """Cross-attention: temporal queries attend to visual + spatial keys/values.

    Args:
        query_dim: Temporal embedding dimension.
        kv_dim: Visual + spatial combined dimension.
        nhead: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        query_dim: int = 256,
        kv_dim: int = 384,  # 256 (visual) + 128 (spatial)
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, 256)
        self.key_proj = nn.Linear(kv_dim, 256)
        self.value_proj = nn.Linear(kv_dim, 256)

        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(256)

    def forward(
        self,
        temporal: torch.Tensor,
        visual: torch.Tensor,
        spatial: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            temporal: (B, 256) temporal embedding (query).
            visual: (B, 256) visual embedding.
            spatial: (B, 128) spatial embedding.

        Returns:
            attended: (B, 256) cross-attended temporal features.
        """
        # Combine visual and spatial as key/value
        kv = torch.cat([visual, spatial], dim=-1)  # (B, 384)

        # Add sequence dimension for attention (B, 1, D)
        q = self.query_proj(temporal).unsqueeze(1)
        k = self.key_proj(kv).unsqueeze(1)
        v = self.value_proj(kv).unsqueeze(1)

        attended, _ = self.attention(q, k, v)
        attended = attended.squeeze(1)  # (B, 256)

        # Residual connection
        attended = self.norm(attended + temporal)
        return attended


class GatedFusion(nn.Module):
    """Gated fusion: sigmoid(W*concat) * proj(concat) -> fused_dim.

    Args:
        input_dim: Total concatenated dimension.
        fused_dim: Output dimension.
    """

    def __init__(self, input_dim: int = 640, fused_dim: int = 640):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        proj = self.projection(x)
        return gate * proj


class MultiModalFusionModel(nn.Module):
    """Full multi-modal model: Swin + Temporal + Spatial -> Fusion -> Heads.

    Args:
        swin_checkpoint: Path to pretrained Swin classifier checkpoint.
        num_classes: Number of disease classes.
        freeze_swin: Whether to freeze Swin backbone.
        temporal_config: Dict with temporal encoder config.
        spatial_config: Dict with spatial MLP config.
        fusion_config: Dict with fusion config.
    """

    def __init__(
        self,
        num_classes: int = 15,
        swin_model_name: str = "swin_tiny_patch4_window7_224",
        swin_checkpoint: Optional[str] = None,
        freeze_swin: bool = True,
        feature_dim: int = 768,
        visual_proj_dim: int = 256,
        temporal_config: Optional[Dict] = None,
        spatial_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        forecast_days: int = 7,
        num_severity_levels: int = 4,
    ):
        super().__init__()
        temporal_config = temporal_config or {}
        spatial_config = spatial_config or {}
        fusion_config = fusion_config or {}

        # Visual backbone
        self.swin = SwinClassifier(
            model_name=swin_model_name,
            num_classes=num_classes,
            pretrained=swin_checkpoint is None,
            feature_dim=feature_dim,
        )
        if swin_checkpoint:
            ckpt = torch.load(swin_checkpoint, map_location="cpu")
            self.swin.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.info(f"Loaded Swin checkpoint: {swin_checkpoint}")

        if freeze_swin:
            for param in self.swin.parameters():
                param.requires_grad = False
            logger.info("Swin backbone frozen")

        # Visual projection
        self.visual_proj = nn.Sequential(
            nn.Linear(feature_dim, visual_proj_dim),
            nn.GELU(),
            nn.LayerNorm(visual_proj_dim),
        )

        # Temporal encoder
        self.temporal_encoder = TemporalTransformerEncoder(
            num_features=temporal_config.get("num_features", 7),
            d_model=temporal_config.get("d_model", 128),
            nhead=temporal_config.get("nhead", 8),
            num_layers=temporal_config.get("num_layers", 4),
            dim_feedforward=temporal_config.get("dim_feedforward", 512),
            dropout=temporal_config.get("dropout", 0.1),
            output_dim=temporal_config.get("output_dim", 256),
            seq_length=temporal_config.get("sequence_length", 30),
        )

        # Spatial MLP
        self.spatial_mlp = SpatialMLP(
            input_dim=spatial_config.get("input_dim", 3),
            hidden_dim=spatial_config.get("hidden_dim", 64),
            output_dim=spatial_config.get("output_dim", 128),
        )

        # Cross-Attention
        fused_dim = fusion_config.get("fused_dim", 640)
        ca_config = fusion_config.get("cross_attention", {})
        self.cross_attention = CrossAttentionFusion(
            query_dim=temporal_config.get("output_dim", 256),
            kv_dim=visual_proj_dim + spatial_config.get("output_dim", 128),
            nhead=ca_config.get("nhead", 8),
            dropout=ca_config.get("dropout", 0.1),
        )

        # Gated Fusion
        # Input: visual(256) + cross_attended_temporal(256) + spatial(128) = 640
        concat_dim = visual_proj_dim + temporal_config.get("output_dim", 256) + spatial_config.get("output_dim", 128)
        self.gated_fusion = GatedFusion(input_dim=concat_dim, fused_dim=fused_dim)

        # Prediction Heads
        self.disease_head = DiseaseClassificationHead(fused_dim, num_classes)
        self.outbreak_head = OutbreakRegressionHead(fused_dim, forecast_days)
        self.severity_head = SeverityHead(fused_dim, num_severity_levels)

        logger.info(
            f"MultiModalFusionModel: classes={num_classes}, fused_dim={fused_dim}, "
            f"forecast={forecast_days}d"
        )

    def forward(
        self,
        image: torch.Tensor,
        iot_sequence: torch.Tensor,
        geo: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: (B, 3, 224, 224) input images.
            iot_sequence: (B, seq_len, num_features) IoT time series.
            geo: (B, 3) normalized geographic features.

        Returns:
            Dict with keys: disease_logits, outbreak_risk, severity_logits
        """
        # Visual features
        with torch.no_grad() if not any(p.requires_grad for p in self.swin.parameters()) else torch.enable_grad():
            visual_features = self.swin.get_features(image)  # (B, 768)
        visual_proj = self.visual_proj(visual_features)  # (B, 256)

        # Temporal features
        temporal_features = self.temporal_encoder(iot_sequence)  # (B, 256)

        # Spatial features
        spatial_features = self.spatial_mlp(geo)  # (B, 128)

        # Cross-attention
        cross_attended = self.cross_attention(
            temporal_features, visual_proj, spatial_features
        )  # (B, 256)

        # Gated Fusion
        fused = torch.cat([visual_proj, cross_attended, spatial_features], dim=-1)  # (B, 640)
        fused = self.gated_fusion(fused)  # (B, 640)

        # Prediction heads
        disease_logits = self.disease_head(fused)
        outbreak_risk = self.outbreak_head(fused)
        severity_logits = self.severity_head(fused)

        return {
            "disease_logits": disease_logits,
            "outbreak_risk": outbreak_risk,
            "severity_logits": severity_logits,
        }

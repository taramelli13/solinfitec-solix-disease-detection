"""Swin Transformer classifier with custom classification head."""

import logging
from typing import List, Optional

import timm
import torch
import torch.nn as nn

logger = logging.getLogger("solinfitec.swin_classifier")


class SwinClassifier(nn.Module):
    """Swin Transformer with custom classification head.

    Architecture:
        Swin backbone (768-d features) -> Linear(768, 256) -> GELU -> Dropout -> Linear(256, num_classes)

    Args:
        model_name: timm model name (e.g., 'swin_tiny_patch4_window7_224').
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
        hidden_dim: Hidden dimension in the classification head.
        dropout: Dropout rate in the classification head.
        feature_dim: Feature dimension from backbone (768 for swin_tiny).
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 15,
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        feature_dim: int = 768,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Load Swin backbone without the default head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Custom classification head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        logger.info(
            f"SwinClassifier: {model_name}, classes={num_classes}, "
            f"pretrained={pretrained}, head={feature_dim}->{hidden_dim}->{num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        features = self.backbone(x)  # (B, feature_dim)
        logits = self.head(features)  # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 768-d feature vector for fusion model."""
        return self.backbone(x)

    def freeze_stages(self, stages: Optional[List[int]] = None) -> None:
        """Freeze specific Swin stages (0-3).

        Stage 0: patch_embed + layers[0]
        Stage 1: layers[1]
        Stage 2: layers[2]
        Stage 3: layers[3]
        """
        if stages is None:
            stages = [0, 1]

        # Always freeze patch embedding if stage 0 is frozen
        if 0 in stages:
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False
            logger.info("Froze patch_embed")

        for stage_idx in stages:
            if hasattr(self.backbone, "layers") and stage_idx < len(self.backbone.layers):
                for param in self.backbone.layers[stage_idx].parameters():
                    param.requires_grad = False
                logger.info(f"Froze stage {stage_idx}")

        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        total = sum(1 for p in self.parameters())
        logger.info(f"Frozen params: {frozen}/{total}")

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfroze all parameters")

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

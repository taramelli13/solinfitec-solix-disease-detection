"""Tests for Swin Transformer classifier."""

import pytest
import torch

from src.models.swin_classifier import SwinClassifier


class TestSwinClassifier:
    @pytest.fixture
    def model(self):
        return SwinClassifier(
            model_name="swin_tiny_patch4_window7_224",
            num_classes=15,
            pretrained=False,
            hidden_dim=256,
            dropout=0.3,
        )

    def test_forward_shape(self, model):
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 15)

    def test_get_features_shape(self, model):
        x = torch.randn(2, 3, 224, 224)
        features = model.get_features(x)
        assert features.shape == (2, 768)

    def test_freeze_stages(self, model):
        total_before = sum(1 for p in model.parameters() if p.requires_grad)
        model.freeze_stages([0, 1])
        total_after = sum(1 for p in model.parameters() if p.requires_grad)
        assert total_after < total_before

    def test_unfreeze_all(self, model):
        model.freeze_stages([0, 1])
        model.unfreeze_all()
        all_trainable = all(p.requires_grad for p in model.parameters())
        assert all_trainable

    def test_param_counts(self, model):
        total = model.get_total_params()
        trainable = model.get_trainable_params()
        assert total > 0
        assert trainable == total  # No freezing yet

    def test_head_structure(self, model):
        """Verify custom head architecture."""
        head = model.head
        assert isinstance(head[0], torch.nn.Linear)
        assert head[0].in_features == 768
        assert head[0].out_features == 256
        assert isinstance(head[1], torch.nn.GELU)
        assert isinstance(head[2], torch.nn.Dropout)
        assert isinstance(head[3], torch.nn.Linear)
        assert head[3].out_features == 15

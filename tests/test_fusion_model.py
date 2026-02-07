"""Tests for the multi-modal fusion model."""

import pytest
import torch

from src.models.fusion_model import (
    CrossAttentionFusion,
    GatedFusion,
    MultiModalFusionModel,
)
from src.models.losses import MultiTaskLoss
from src.models.prediction_heads import (
    DiseaseClassificationHead,
    OutbreakRegressionHead,
    SeverityHead,
)
from src.models.temporal_encoder import TemporalTransformerEncoder
from src.features.spatial_features import SpatialMLP


class TestTemporalEncoder:
    def test_output_shape(self):
        encoder = TemporalTransformerEncoder(
            num_features=7, d_model=128, nhead=8,
            num_layers=4, output_dim=256, seq_length=30,
        )
        x = torch.randn(4, 30, 7)
        out = encoder(x)
        assert out.shape == (4, 256)


class TestSpatialMLP:
    def test_output_shape(self):
        mlp = SpatialMLP(input_dim=3, hidden_dim=64, output_dim=128)
        x = torch.randn(4, 3)
        out = mlp(x)
        assert out.shape == (4, 128)


class TestCrossAttention:
    def test_output_shape(self):
        ca = CrossAttentionFusion(query_dim=256, kv_dim=384, nhead=8)
        temporal = torch.randn(4, 256)
        visual = torch.randn(4, 256)
        spatial = torch.randn(4, 128)
        out = ca(temporal, visual, spatial)
        assert out.shape == (4, 256)


class TestGatedFusion:
    def test_output_shape(self):
        gf = GatedFusion(input_dim=640, fused_dim=640)
        x = torch.randn(4, 640)
        out = gf(x)
        assert out.shape == (4, 640)


class TestPredictionHeads:
    def test_disease_head(self):
        head = DiseaseClassificationHead(640, 15)
        x = torch.randn(4, 640)
        out = head(x)
        assert out.shape == (4, 15)

    def test_outbreak_head(self):
        head = OutbreakRegressionHead(640, 7)
        x = torch.randn(4, 640)
        out = head(x)
        assert out.shape == (4, 7)
        assert (out >= 0).all() and (out <= 1).all()  # Sigmoid output

    def test_severity_head(self):
        head = SeverityHead(640, 4)
        x = torch.randn(4, 640)
        out = head(x)
        assert out.shape == (4, 4)


class TestMultiTaskLoss:
    def test_forward(self):
        mtl = MultiTaskLoss(num_tasks=3, task_names=["disease", "outbreak", "severity"])
        losses = [
            torch.tensor(1.5, requires_grad=True),
            torch.tensor(0.3, requires_grad=True),
            torch.tensor(0.8, requires_grad=True),
        ]
        total, loss_dict = mtl(losses)
        assert total.requires_grad
        assert "disease_raw" in loss_dict
        assert "total" in loss_dict

    def test_learnable_weights(self):
        mtl = MultiTaskLoss(num_tasks=3)
        assert mtl.log_vars.requires_grad
        assert mtl.log_vars.shape == (3,)


class TestMultiModalFusionModel:
    @pytest.fixture
    def model(self):
        return MultiModalFusionModel(
            num_classes=15,
            swin_model_name="swin_tiny_patch4_window7_224",
            swin_checkpoint=None,
            freeze_swin=True,
            feature_dim=768,
            visual_proj_dim=256,
            temporal_config={
                "num_features": 7, "d_model": 128, "nhead": 8,
                "num_layers": 2, "output_dim": 256, "sequence_length": 30,
            },
            spatial_config={"input_dim": 3, "hidden_dim": 64, "output_dim": 128},
            fusion_config={"fused_dim": 640},
        )

    def test_forward_shapes(self, model):
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        iot_seq = torch.randn(batch_size, 30, 7)
        geo = torch.randn(batch_size, 3)

        outputs = model(images, iot_seq, geo)

        assert outputs["disease_logits"].shape == (batch_size, 15)
        assert outputs["outbreak_risk"].shape == (batch_size, 7)
        assert outputs["severity_logits"].shape == (batch_size, 4)

    def test_swin_frozen(self, model):
        for param in model.swin.parameters():
            assert not param.requires_grad

    def test_trainable_components(self, model):
        # Visual projection should be trainable
        for param in model.visual_proj.parameters():
            assert param.requires_grad
        # Temporal encoder should be trainable
        for param in model.temporal_encoder.parameters():
            assert param.requires_grad

"""Tests for ONNX export pipeline."""

import pytest
import torch

from src.models.swin_classifier import SwinClassifier
from src.utils.onnx_export import ONNXExporter


class TestONNXExporter:
    @pytest.fixture
    def model(self):
        return SwinClassifier(
            model_name="swin_tiny_patch4_window7_224",
            num_classes=15,
            pretrained=False,
        )

    @pytest.fixture
    def exporter(self):
        return ONNXExporter(opset_version=14, max_diff=1e-4)

    def test_export_creates_file(self, model, exporter, tmp_path):
        onnx_path = str(tmp_path / "model.onnx")
        result = exporter.export_swin(model, onnx_path)
        assert (tmp_path / "model.onnx").exists()
        assert result == onnx_path

    def test_export_validate(self, model, exporter, tmp_path):
        onnx_path = str(tmp_path / "model.onnx")
        exporter.export_swin(model, onnx_path)
        valid = exporter.validate(model, onnx_path)
        assert valid

    def test_benchmark(self, model, exporter, tmp_path):
        onnx_path = str(tmp_path / "model.onnx")
        exporter.export_swin(model, onnx_path)
        results = exporter.benchmark_latency(onnx_path, num_runs=5, warmup=2)
        assert "mean_ms" in results
        assert results["mean_ms"] > 0
        assert results["num_runs"] == 5

    def test_dynamic_batch(self, model, exporter, tmp_path):
        """Test that exported model accepts different batch sizes."""
        import onnxruntime as ort
        import numpy as np

        onnx_path = str(tmp_path / "model.onnx")
        exporter.export_swin(model, onnx_path, dynamic_batch=True)

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name

        # Test batch size 1
        out1 = session.run(None, {input_name: np.random.randn(1, 3, 224, 224).astype(np.float32)})
        assert out1[0].shape == (1, 15)

        # Test batch size 4
        out4 = session.run(None, {input_name: np.random.randn(4, 3, 224, 224).astype(np.float32)})
        assert out4[0].shape == (4, 15)

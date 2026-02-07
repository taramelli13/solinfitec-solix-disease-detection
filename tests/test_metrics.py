"""Tests for metrics calculation."""

import numpy as np
import pytest

from src.utils.metrics import MetricsCalculator


class TestMetricsCalculator:
    @pytest.fixture
    def calc(self):
        return MetricsCalculator(
            class_names=["class_a", "class_b", "class_c"],
            num_classes=3,
        )

    def test_compute_all_without_probs(self, calc):
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1])
        metrics = calc.compute_all(y_true, y_pred)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert metrics["accuracy"] > 0.5

    def test_compute_all_with_probs(self, calc):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 2])
        y_prob = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.2, 0.6],
        ])
        metrics = calc.compute_all(y_true, y_pred, y_prob)
        assert "auroc_macro" in metrics
        assert "mAP@0.5" in metrics

    def test_per_class_metrics(self, calc):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 2])
        per_class = calc.per_class_metrics(y_true, y_pred)
        assert "class_a" in per_class
        assert "precision" in per_class["class_a"]
        assert per_class["class_a"]["precision"] > 0

    def test_confusion_matrix_shape(self, calc):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 1, 1])
        cm = calc.compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)

    def test_perfect_predictions(self, calc):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        metrics = calc.compute_all(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

"""Evaluation metrics: F1, precision, recall, AUROC, mAP, confusion matrix."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("solinfitec.metrics")


class MetricsCalculator:
    """Computes classification metrics from predictions and targets."""

    def __init__(self, class_names: List[str], num_classes: int):
        self.class_names = class_names
        self.num_classes = num_classes

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute all metrics.

        Args:
            y_true: Ground truth labels (N,).
            y_pred: Predicted labels (N,).
            y_prob: Prediction probabilities (N, C) for AUROC/mAP.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        if y_prob is not None:
            try:
                metrics["auroc_macro"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
            except ValueError:
                metrics["auroc_macro"] = 0.0

            try:
                metrics["mAP@0.5"] = self._compute_map(y_true, y_prob)
            except Exception:
                metrics["mAP@0.5"] = 0.0

        return metrics

    def _compute_map(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute mean Average Precision."""
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        aps = []
        for i in range(self.num_classes):
            if y_true_bin[:, i].sum() > 0:
                ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                aps.append(ap)
        return float(np.mean(aps)) if aps else 0.0

    def per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Get precision, recall, F1 per class."""
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        return {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in self.class_names
            if name in report
        }

    def compute_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute confusion matrix."""
        return confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))

    def log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log metrics to logger."""
        for name, value in metrics.items():
            logger.info(f"{prefix}{name}: {value:.4f}")

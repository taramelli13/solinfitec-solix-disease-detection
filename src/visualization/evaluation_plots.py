"""Evaluation visualizations: confusion matrix, ROC/PR curves, error gallery."""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

logger = logging.getLogger("solinfitec.evaluation_plots")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
    normalize: bool = True,
) -> None:
    """Plot confusion matrix heatmap."""
    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-6)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> None:
    """Plot per-class ROC curves."""
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curves saved to {save_path}")
    plt.close(fig)


def plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> None:
    """Plot per-class Precision-Recall curves."""
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        if y_true_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=color, lw=1.5, label=f"{name} (AP={pr_auc:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14)
    ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_gallery(
    image_paths: List[str],
    true_labels: List[str],
    pred_labels: List[str],
    save_path: Optional[str] = None,
    max_images: int = 20,
    figsize: tuple = (16, 10),
) -> None:
    """Display a grid of misclassified images."""
    n = min(len(image_paths), max_images)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 and cols == 1 else axes.flatten()

    for i in range(n):
        img = Image.open(image_paths[i]).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {true_labels[i]}\nPred: {pred_labels[i]}",
            fontsize=7,
            color="red",
        )
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Misclassified Samples", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Error gallery saved to {save_path}")
    plt.close(fig)

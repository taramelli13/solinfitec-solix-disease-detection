"""Training visualization: loss/accuracy curves, LR schedule."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("solinfitec.training_plots")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> None:
    """Plot training and validation loss/accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    ncols = 2 if train_accs else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    # Loss
    axes[0].plot(epochs, train_losses, label="Train Loss", color="steelblue")
    axes[0].plot(epochs, val_losses, label="Val Loss", color="coral")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    if train_accs and val_accs:
        axes[1].plot(epochs, train_accs, label="Train Acc", color="steelblue")
        axes[1].plot(epochs, val_accs, label="Val Acc", color="coral")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Curves")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lr_schedule(
    lrs: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 4),
) -> None:
    """Plot learning rate schedule over epochs."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(len(lrs)), lrs, color="purple", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(
    metrics_dict: Dict[str, float],
    title: str = "Per-Class F1 Score",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> None:
    """Horizontal bar chart for per-class metrics."""
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["green" if v >= 0.9 else "orange" if v >= 0.7 else "red" for v in values]
    bars = ax.barh(names, values, color=colors)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

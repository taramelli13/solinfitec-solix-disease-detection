"""Dataset visualization: class distributions, sample grids."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger("solinfitec.dataset_plots")


def plot_class_distribution(
    class_counts: Dict[str, int],
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> None:
    """Bar chart of class distribution."""
    names = list(class_counts.keys())
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.bar(range(len(names)), counts, color=colors)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title(title, fontsize=14)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(
    image_paths: List[str],
    class_names: Optional[List[str]] = None,
    title: str = "Sample Images",
    save_path: Optional[str] = None,
    max_images: int = 25,
    figsize: tuple = (15, 12),
) -> None:
    """Display a grid of sample images."""
    n = min(len(image_paths), max_images)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        row, col = divmod(i, cols)
        img = Image.open(image_paths[i]).convert("RGB")
        axes[row, col].imshow(img)
        if class_names and i < len(class_names):
            axes[row, col].set_title(class_names[i], fontsize=7)
        axes[row, col].axis("off")

    for i in range(n, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_image_size_distribution(
    sizes: List[tuple],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 4),
) -> None:
    """Histogram of image widths and heights."""
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.hist(widths, bins=30, color="steelblue", edgecolor="white")
    ax1.set_title("Width Distribution")
    ax1.set_xlabel("Width (px)")

    ax2.hist(heights, bins=30, color="coral", edgecolor="white")
    ax2.set_title("Height Distribution")
    ax2.set_xlabel("Height (px)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

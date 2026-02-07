"""Outbreak prediction visualizations: risk timelines, spatial heatmaps, temporal progression."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("solinfitec.outbreak_plots")


def plot_risk_timeline(
    risk_7d: List[float],
    threshold_high: float = 0.75,
    threshold_medium: float = 0.5,
    field_id: str = "0",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
) -> None:
    """Plot 7-day outbreak risk timeline with risk level zones."""
    days = list(range(1, len(risk_7d) + 1))

    fig, ax = plt.subplots(figsize=figsize)

    # Risk zones
    ax.axhspan(0, threshold_medium, alpha=0.1, color="green", label="Low Risk")
    ax.axhspan(threshold_medium, threshold_high, alpha=0.1, color="orange", label="Medium Risk")
    ax.axhspan(threshold_high, 1.0, alpha=0.1, color="red", label="High Risk")

    # Risk line
    colors = ["green" if r < threshold_medium else "orange" if r < threshold_high else "red"
              for r in risk_7d]
    ax.bar(days, risk_7d, color=colors, edgecolor="white", width=0.6)
    ax.plot(days, risk_7d, "ko-", markersize=6, linewidth=2)

    for d, r in zip(days, risk_7d):
        ax.text(d, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)

    ax.set_xlabel("Day Ahead", fontsize=12)
    ax.set_ylabel("Outbreak Risk", fontsize=12)
    ax.set_title(f"7-Day Outbreak Risk Forecast - Field {field_id}", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(days)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_heatmap(
    field_locations: Dict[str, Tuple[float, float]],
    risk_values: Dict[str, float],
    title: str = "Spatial Risk Distribution",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> None:
    """Scatter plot of field locations colored by risk level."""
    fig, ax = plt.subplots(figsize=figsize)

    lats = [loc[0] for loc in field_locations.values()]
    lons = [loc[1] for loc in field_locations.values()]
    risks = [risk_values.get(fid, 0) for fid in field_locations.keys()]

    scatter = ax.scatter(
        lons, lats,
        c=risks,
        cmap="RdYlGn_r",
        s=150,
        edgecolors="black",
        linewidth=0.5,
        vmin=0,
        vmax=1,
    )

    for fid, (lat, lon) in field_locations.items():
        ax.annotate(fid, (lon, lat), fontsize=7, ha="center", va="bottom")

    plt.colorbar(scatter, ax=ax, label="Risk Score")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_progression(
    dates: List[str],
    disease_prevalence: List[float],
    temperature: Optional[List[float]] = None,
    humidity: Optional[List[float]] = None,
    title: str = "Disease Progression Over Time",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> None:
    """Plot disease prevalence over time with optional environmental factors."""
    fig, ax1 = plt.subplots(figsize=figsize)

    # Disease prevalence
    ax1.fill_between(range(len(dates)), disease_prevalence, alpha=0.3, color="red")
    ax1.plot(range(len(dates)), disease_prevalence, "r-", linewidth=2, label="Disease Prevalence")
    ax1.set_ylabel("Disease Prevalence", color="red", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_ylim(0, max(max(disease_prevalence) * 1.2, 0.1))

    # Environmental factors on second y-axis
    if temperature or humidity:
        ax2 = ax1.twinx()
        if temperature:
            ax2.plot(range(len(dates)), temperature, "b--", alpha=0.6, label="Temperature")
        if humidity:
            ax2.plot(range(len(dates)), humidity, "g--", alpha=0.6, label="Humidity")
        ax2.set_ylabel("Environmental Factor", fontsize=12)
        ax2.legend(loc="upper right")

    # X-axis labels (show every Nth date)
    step = max(1, len(dates) // 10)
    ax1.set_xticks(range(0, len(dates), step))
    ax1.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45, fontsize=8)

    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_alert_summary(
    alerts: List[Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Summary chart of alert counts by risk level."""
    from collections import Counter

    risk_counts = Counter(a.get("risk_level", "UNKNOWN") for a in alerts)
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts = [risk_counts.get(lvl, 0) for lvl in levels]
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(levels, counts, color=colors, edgecolor="white")

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(count),
                ha="center",
                fontsize=14,
                fontweight="bold",
            )

    ax.set_ylabel("Number of Alerts", fontsize=12)
    ax.set_title("Alert Summary by Risk Level", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

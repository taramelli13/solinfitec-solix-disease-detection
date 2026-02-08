"""Evaluation script for the Multi-Modal Fusion Model on the test set."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.datamodule import PlantVillageDataModule
from src.data.iot_simulator import IoTSimulator
from src.data.multimodal_dataset import MultiModalDataset
from src.features.augmentation import get_val_transforms
from src.models.fusion_model import MultiModalFusionModel
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from src.utils.metrics import MetricsCalculator
from src.utils.seed import set_seed
from src.visualization.evaluation_plots import (
    plot_confusion_matrix,
    plot_roc_curves,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Modal Fusion Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_fusion_model.pth",
    )
    parser.add_argument("--output_dir", type=str, default="reports/metrics/fusion")
    return parser.parse_args()


def generate_geo_data(num_fields: int, seed: int = 42):
    """Generate synthetic geographic data for fields (must match training)."""
    rng = np.random.default_rng(seed)
    geo_data = {}
    for i in range(num_fields):
        lat = -23.0 + rng.uniform(-2, 2)
        lon = -49.0 + rng.uniform(-2, 2)
        elevation = 400 + rng.uniform(0, 600)
        geo_data[i] = (lat, lon, elevation)
    return geo_data


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    all_disease_preds = []
    all_disease_labels = []
    all_disease_probs = []
    all_outbreak_preds = []
    all_outbreak_targets = []
    all_severity_preds = []
    all_severity_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        iot_seq = batch["iot_sequence"].to(device)
        geo = batch["geo"].to(device)

        outputs = model(images, iot_seq, geo)

        # Disease classification
        disease_probs = torch.softmax(outputs["disease_logits"], dim=1)
        disease_preds = disease_probs.argmax(dim=1)
        all_disease_preds.extend(disease_preds.cpu().numpy())
        all_disease_labels.extend(batch["disease_label"].numpy())
        all_disease_probs.append(disease_probs.cpu().numpy())

        # Outbreak regression
        all_outbreak_preds.append(outputs["outbreak_risk"].cpu().numpy())
        all_outbreak_targets.append(batch["outbreak_target"].numpy())

        # Severity classification
        severity_preds = outputs["severity_logits"].argmax(dim=1)
        all_severity_preds.extend(severity_preds.cpu().numpy())
        all_severity_labels.extend(batch["severity"].numpy())

    return {
        "disease_preds": np.array(all_disease_preds),
        "disease_labels": np.array(all_disease_labels),
        "disease_probs": np.concatenate(all_disease_probs, axis=0),
        "outbreak_preds": np.concatenate(all_outbreak_preds, axis=0),
        "outbreak_targets": np.concatenate(all_outbreak_targets, axis=0),
        "severity_preds": np.array(all_severity_preds),
        "severity_labels": np.array(all_severity_labels),
    }


def main():
    args = parse_args()
    cfg = ConfigManager(args.config)
    set_seed(cfg.seed)

    logger = setup_logger("solinfitec", level=cfg.logging.level)
    logger.info("Starting Fusion Model evaluation")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Generate IoT data (same seed as training)
    iot_config = cfg.get_raw("iot_simulation", {})
    iot_sim = IoTSimulator(iot_config, seed=cfg.seed)
    iot_data = iot_sim.generate_all()
    geo_data = generate_geo_data(iot_config.get("num_fields", 50), seed=cfg.seed)

    # Image data
    data_dir = os.path.join(cfg.data.raw_dir, cfg.data.dataset_name)
    val_transform = get_val_transforms(img_size=cfg.data.img_size[0])

    dm = PlantVillageDataModule(
        data_dir=data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_transform=val_transform,
        seed=cfg.seed,
    )

    # Create multi-modal test dataset
    test_mm = MultiModalDataset.create_from_datasets(
        dm.test_dataset, iot_data, geo_data, transform=val_transform,
    )
    test_loader = DataLoader(
        test_mm, batch_size=cfg.data.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers,
    )

    # Model
    temporal_config = {
        "num_features": cfg.temporal_model.num_features,
        "d_model": cfg.temporal_model.d_model,
        "nhead": cfg.temporal_model.nhead,
        "num_layers": cfg.temporal_model.num_layers,
        "dim_feedforward": cfg.temporal_model.dim_feedforward,
        "dropout": cfg.temporal_model.dropout,
        "output_dim": cfg.temporal_model.output_dim,
        "sequence_length": cfg.temporal_model.sequence_length,
    }
    spatial_config = {
        "input_dim": cfg.spatial_model.input_dim,
        "hidden_dim": cfg.spatial_model.hidden_dim,
        "output_dim": cfg.spatial_model.output_dim,
    }

    model = MultiModalFusionModel(
        num_classes=dm.num_classes,
        swin_model_name=cfg.model.variant,
        swin_checkpoint=None,
        freeze_swin=True,
        feature_dim=cfg.model.feature_dim,
        visual_proj_dim=cfg.fusion.visual_proj_dim,
        temporal_config=temporal_config,
        spatial_config=spatial_config,
        fusion_config={
            "fused_dim": cfg.fusion.fused_dim,
            "cross_attention": {
                "nhead": cfg.fusion.cross_attention.nhead,
                "dropout": cfg.fusion.cross_attention.dropout,
            },
        },
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded fusion checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Evaluate
    results = evaluate_model(model, test_loader, device)

    # === Disease Classification Metrics ===
    disease_calc = MetricsCalculator(dm.class_names, dm.num_classes)
    disease_overall = disease_calc.compute_all(
        results["disease_labels"], results["disease_preds"], results["disease_probs"]
    )
    disease_per_class = disease_calc.per_class_metrics(
        results["disease_labels"], results["disease_preds"]
    )
    disease_cm = disease_calc.compute_confusion_matrix(
        results["disease_labels"], results["disease_preds"]
    )

    logger.info("=== Disease Classification Metrics ===")
    disease_calc.log_metrics(disease_overall, prefix="  ")
    logger.info("=== Per-Class Disease Metrics ===")
    for cls_name, cls_metrics in disease_per_class.items():
        logger.info(
            f"  {cls_name}: P={cls_metrics['precision']:.3f} "
            f"R={cls_metrics['recall']:.3f} F1={cls_metrics['f1']:.3f}"
        )

    # === Outbreak Regression Metrics ===
    outbreak_mae = np.mean(np.abs(results["outbreak_preds"] - results["outbreak_targets"]))
    outbreak_rmse = np.sqrt(np.mean((results["outbreak_preds"] - results["outbreak_targets"]) ** 2))
    per_day_mae = np.mean(np.abs(results["outbreak_preds"] - results["outbreak_targets"]), axis=0)

    logger.info("=== Outbreak Risk Metrics (7-day forecast) ===")
    logger.info(f"  MAE: {outbreak_mae:.4f}")
    logger.info(f"  RMSE: {outbreak_rmse:.4f}")
    for day_i, day_mae in enumerate(per_day_mae):
        logger.info(f"  Day {day_i + 1} MAE: {day_mae:.4f}")

    # === Severity Classification Metrics ===
    severity_names = ["healthy", "initial", "moderate", "severe"]
    severity_num_classes = len(severity_names)
    severity_acc = np.mean(results["severity_preds"] == results["severity_labels"])

    logger.info("=== Severity Classification Metrics ===")
    logger.info(f"  Accuracy: {severity_acc:.4f}")
    for i, name in enumerate(severity_names):
        mask = results["severity_labels"] == i
        if mask.sum() > 0:
            cls_acc = np.mean(results["severity_preds"][mask] == i)
            logger.info(f"  {name} (n={mask.sum()}): Acc={cls_acc:.3f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {
        "disease_classification": {
            "overall": disease_overall,
            "per_class": disease_per_class,
        },
        "outbreak_regression": {
            "mae": float(outbreak_mae),
            "rmse": float(outbreak_rmse),
            "per_day_mae": [float(m) for m in per_day_mae],
        },
        "severity_classification": {
            "accuracy": float(severity_acc),
        },
    }

    with open(output_dir / "fusion_test_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Plots
    plot_confusion_matrix(
        disease_cm, dm.class_names,
        save_path=str(output_dir / "fusion_disease_confusion_matrix.png"),
    )
    plot_roc_curves(
        results["disease_labels"], results["disease_probs"], dm.class_names,
        save_path=str(output_dir / "fusion_disease_roc_curves.png"),
    )

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

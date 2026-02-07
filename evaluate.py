"""Evaluation script for the Swin classifier on the test set."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.datamodule import PlantVillageDataModule
from src.features.augmentation import get_val_transforms
from src.models.swin_classifier import SwinClassifier
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from src.utils.metrics import MetricsCalculator
from src.utils.seed import set_seed
from src.visualization.evaluation_plots import (
    plot_confusion_matrix,
    plot_error_gallery,
    plot_roc_curves,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Swin Classifier")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_swin_classifier.pth",
    )
    parser.add_argument("--output_dir", type=str, default="reports/metrics")
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    return np.array(all_preds), np.array(all_labels), all_probs


def main():
    args = parse_args()
    cfg = ConfigManager(args.config)
    set_seed(cfg.seed)

    logger = setup_logger("solinfitec", level=cfg.logging.level)
    logger.info("Starting evaluation")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Data
    data_dir = os.path.join(cfg.data.raw_dir, cfg.data.dataset_name)
    val_transform = get_val_transforms(img_size=cfg.data.img_size[0])

    dm = PlantVillageDataModule(
        data_dir=data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_transform=val_transform,
        seed=cfg.seed,
    )
    test_loader = dm.test_dataloader()

    # Model
    model = SwinClassifier(
        model_name=cfg.model.variant,
        num_classes=dm.num_classes,
        pretrained=False,
        hidden_dim=cfg.model.head.hidden_dim,
        dropout=cfg.model.head.dropout,
        feature_dim=cfg.model.feature_dim,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Evaluate
    y_pred, y_true, y_prob = evaluate_model(model, test_loader, device)

    # Metrics
    metrics_calc = MetricsCalculator(dm.class_names, dm.num_classes)
    overall = metrics_calc.compute_all(y_true, y_pred, y_prob)
    per_class = metrics_calc.per_class_metrics(y_true, y_pred)
    cm = metrics_calc.compute_confusion_matrix(y_true, y_pred)

    # Log results
    logger.info("=== Overall Metrics ===")
    metrics_calc.log_metrics(overall, prefix="  ")
    logger.info("=== Per-Class Metrics ===")
    for cls_name, cls_metrics in per_class.items():
        logger.info(
            f"  {cls_name}: P={cls_metrics['precision']:.3f} "
            f"R={cls_metrics['recall']:.3f} F1={cls_metrics['f1']:.3f}"
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(
            {"overall": overall, "per_class": per_class},
            f,
            indent=2,
            default=str,
        )

    # Plots
    plot_confusion_matrix(
        cm, dm.class_names, save_path=str(output_dir / "confusion_matrix.png")
    )
    plot_roc_curves(
        y_true, y_prob, dm.class_names, save_path=str(output_dir / "roc_curves.png")
    )

    # Error gallery
    error_mask = y_pred != y_true
    if error_mask.any():
        error_indices = np.where(error_mask)[0][:20]
        error_paths = [dm.test_dataset.image_paths[i] for i in error_indices]
        error_true = [dm.class_names[y_true[i]] for i in error_indices]
        error_pred = [dm.class_names[y_pred[i]] for i in error_indices]
        plot_error_gallery(
            error_paths, error_true, error_pred,
            save_path=str(output_dir / "error_gallery.png"),
        )

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

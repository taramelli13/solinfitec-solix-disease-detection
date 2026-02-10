"""Training script for the Multi-Modal Fusion Model.

Trains temporal encoder + fusion layers with frozen Swin backbone.
Multi-task learning: disease classification + outbreak regression + severity.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.datamodule import PlantVillageDataModule
from src.data.iot_simulator import IoTSimulator
from src.data.multimodal_dataset import MultiModalDataset
from src.features.augmentation import get_train_transforms, get_val_transforms
from src.models.fusion_model import MultiModalFusionModel
from src.models.losses import FocalLoss, MultiTaskLoss
from src.utils.callbacks import EarlyStopping, LRSchedulerWrapper, MetricsLogger, ModelCheckpoint
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed


class _TransformWrapper(torch.utils.data.Dataset):
    """Wraps a Subset to apply a specific image transform.

    Used to apply train/val transforms to random_split subsets of DiaMOS.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]
        if self.transform is not None:
            # The image comes as a raw numpy array since the parent dataset
            # was created with transform=None
            img = sample["image"]
            if isinstance(img, np.ndarray):
                img = self.transform(image=img)["image"]
            sample = {**sample, "image": img}
        return sample


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Modal Fusion")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--swin_checkpoint",
        type=str,
        default="models/checkpoints/best_swin_classifier.pth",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to fusion checkpoint to resume training from",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="plantvillage",
        choices=["plantvillage", "diamos"],
        help="Dataset to use: 'plantvillage' (default) or 'diamos' (real severity labels)",
    )
    return parser.parse_args()


def generate_geo_data(num_fields: int, seed: int = 42):
    """Generate synthetic geographic data for fields."""
    rng = np.random.default_rng(seed)
    geo_data = {}
    for i in range(num_fields):
        lat = -23.0 + rng.uniform(-2, 2)  # Sao Paulo region
        lon = -49.0 + rng.uniform(-2, 2)
        elevation = 400 + rng.uniform(0, 600)
        geo_data[i] = (lat, lon, elevation)
    return geo_data


def train_one_epoch(model, dataloader, criterion, multitask_loss, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        iot_seq = batch["iot_sequence"].to(device)
        geo = batch["geo"].to(device)
        disease_labels = batch["disease_label"].to(device)
        outbreak_targets = batch["outbreak_target"].to(device)
        severity_labels = batch["severity"].to(device)

        outputs = model(images, iot_seq, geo)

        # Individual losses
        disease_loss = criterion(outputs["disease_logits"], disease_labels)
        outbreak_loss = nn.MSELoss()(outputs["outbreak_risk"], outbreak_targets)
        severity_loss = nn.CrossEntropyLoss()(outputs["severity_logits"], severity_labels)

        # Multi-task weighted loss
        loss, loss_dict = multitask_loss([disease_loss, outbreak_loss, severity_loss])

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, multitask_loss, device):
    model.eval()
    total_loss = 0
    all_disease_preds = []
    all_disease_labels = []
    all_outbreak_preds = []
    all_outbreak_targets = []
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch["image"].to(device)
        iot_seq = batch["iot_sequence"].to(device)
        geo = batch["geo"].to(device)
        disease_labels = batch["disease_label"].to(device)
        outbreak_targets = batch["outbreak_target"].to(device)
        severity_labels = batch["severity"].to(device)

        outputs = model(images, iot_seq, geo)

        disease_loss = criterion(outputs["disease_logits"], disease_labels)
        outbreak_loss = nn.MSELoss()(outputs["outbreak_risk"], outbreak_targets)
        severity_loss = nn.CrossEntropyLoss()(outputs["severity_logits"], severity_labels)

        loss, _ = multitask_loss([disease_loss, outbreak_loss, severity_loss])
        total_loss += loss.item()
        num_batches += 1

        all_disease_preds.extend(outputs["disease_logits"].argmax(1).cpu().numpy())
        all_disease_labels.extend(disease_labels.cpu().numpy())
        all_outbreak_preds.append(outputs["outbreak_risk"].cpu().numpy())
        all_outbreak_targets.append(outbreak_targets.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)

    # Disease accuracy
    disease_acc = np.mean(np.array(all_disease_preds) == np.array(all_disease_labels))

    # Outbreak MAE
    all_outbreak_preds = np.concatenate(all_outbreak_preds)
    all_outbreak_targets = np.concatenate(all_outbreak_targets)
    outbreak_mae = np.mean(np.abs(all_outbreak_preds - all_outbreak_targets))

    return avg_loss, disease_acc, outbreak_mae


def train(
    config_path="configs/config.yaml",
    dataset="plantvillage",
    swin_checkpoint="models/checkpoints/best_swin_classifier.pth",
    resume=None,
    overrides=None,
):
    """Run fusion training. Returns best val_total_loss (negated, higher is better for Optuna maximize).

    Args:
        config_path: Path to config YAML.
        dataset: "plantvillage" or "diamos".
        swin_checkpoint: Path to pretrained Swin checkpoint.
        resume: Checkpoint path to resume from.
        overrides: Dict of dotted config keys to override (for Optuna).
    """
    cfg = ConfigManager(config_path)
    if overrides:
        cfg = cfg.with_overrides(overrides)
    set_seed(cfg.seed)

    logger = setup_logger("solinfitec", log_file=cfg.logging.log_file, level=cfg.logging.level)
    logger.info("Starting Multi-Modal Fusion training")

    # MLflow setup
    mlflow_cfg = cfg.get_raw("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_fusion", "fusion_multimodal"))
    mlflow.start_run(
        run_name=f"fusion_{dataset}_e{cfg.fusion_training.epochs}_lr{cfg.fusion_training.learning_rate}",
        nested=bool(mlflow.active_run()),
    )

    mlflow.log_params({
        "dataset": dataset,
        "swin_checkpoint": swin_checkpoint,
        "batch_size": cfg.data.batch_size,
        "fusion_lr": cfg.fusion_training.learning_rate,
        "freeze_swin": cfg.fusion_training.freeze_swin_backbone,
        "temporal_d_model": cfg.temporal_model.d_model,
        "temporal_layers": cfg.temporal_model.num_layers,
        "temporal_nhead": cfg.temporal_model.nhead,
        "fusion_dim": cfg.fusion.fused_dim,
        "epochs": cfg.fusion_training.epochs,
        "seed": cfg.seed,
    })

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Generate IoT data
    iot_config = cfg.get_raw("iot_simulation", {})
    iot_sim = IoTSimulator(iot_config, seed=cfg.seed)

    # Calibrate from real sensor data if available
    calibration_csv = iot_config.get("calibration_csv")
    if calibration_csv and Path(calibration_csv).exists():
        logger.info(f"Calibrating IoT simulator from {calibration_csv}")
        stats = iot_sim.calibrate_from_real_data(calibration_csv)
        logger.info(f"Calibration stats: {stats}")

    iot_data = iot_sim.generate_all()
    geo_data = generate_geo_data(iot_config.get("num_fields", 50), seed=cfg.seed)
    logger.info(f"Generated IoT data for {len(iot_data)} fields")

    train_transform = get_train_transforms(img_size=cfg.data.img_size[0])
    val_transform = get_val_transforms(img_size=cfg.data.img_size[0])

    if dataset == "diamos":
        # DiaMOS dataset with real severity labels
        from src.data.diamos_dataset import DiaMOSMultiModalDataset

        diamos_dir = cfg.get_raw("data", {}).get("diamos_dir", "data/diamos")
        logger.info(f"Using DiaMOS dataset from {diamos_dir}")

        full_dataset = DiaMOSMultiModalDataset(
            data_dir=diamos_dir,
            iot_data=iot_data,
            geo_data=geo_data,
            transform=None,  # set per-split below
            sequence_length=cfg.temporal_model.sequence_length,
        )
        num_classes = full_dataset.num_classes

        # Split into train/val
        n = len(full_dataset)
        n_train = int(n * cfg.data.train_split)
        n_val = n - n_train
        generator = torch.Generator().manual_seed(cfg.seed)
        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val], generator=generator,
        )

        # Wrap subsets to apply different transforms
        train_mm = _TransformWrapper(train_subset, train_transform)
        val_mm = _TransformWrapper(val_subset, val_transform)

        # Compute class weights from full dataset
        class_weights_tensor = full_dataset.get_class_weights().to(device)

        logger.info(f"DiaMOS: {n_train} train, {n_val} val, {num_classes} classes")
    else:
        # PlantVillage dataset (default)
        data_dir = os.path.join(cfg.data.raw_dir, cfg.data.dataset_name)

        dm = PlantVillageDataModule(
            data_dir=data_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            train_transform=train_transform,
            val_transform=val_transform,
            seed=cfg.seed,
        )
        num_classes = dm.num_classes

        train_mm = MultiModalDataset.create_from_datasets(
            dm.train_dataset, iot_data, geo_data, transform=train_transform,
        )
        val_mm = MultiModalDataset.create_from_datasets(
            dm.val_dataset, iot_data, geo_data, transform=val_transform,
        )
        class_weights_tensor = dm.train_dataset.get_class_weights().to(device)

    train_loader = DataLoader(
        train_mm, batch_size=cfg.data.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_mm, batch_size=cfg.data.batch_size,
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

    swin_ckpt = swin_checkpoint if Path(swin_checkpoint).exists() else None
    model = MultiModalFusionModel(
        num_classes=num_classes,
        swin_model_name=cfg.model.variant,
        swin_checkpoint=swin_ckpt,
        freeze_swin=cfg.fusion_training.freeze_swin_backbone,
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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Fusion model: {total:,} total params, {trainable:,} trainable")

    # Loss
    class_weights = class_weights_tensor
    criterion = FocalLoss(gamma=cfg.training.focal_gamma, alpha=class_weights)
    multitask_loss = MultiTaskLoss(
        num_tasks=3, task_names=["disease", "outbreak", "severity"]
    ).to(device)

    # Optimizer (include multitask loss params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad]
        + list(multitask_loss.parameters()),
        lr=cfg.fusion_training.learning_rate,
        weight_decay=cfg.fusion_training.weight_decay,
    )

    lr_scheduler = LRSchedulerWrapper(
        optimizer,
        scheduler_name=cfg.fusion_training.scheduler,
        scheduler_params={
            "T_0": cfg.fusion_training.scheduler_params.T_0,
            "T_mult": cfg.fusion_training.scheduler_params.T_mult,
            "eta_min": cfg.fusion_training.scheduler_params.eta_min,
        },
    )

    early_stopping = EarlyStopping(
        patience=cfg.fusion_training.early_stopping.patience,
        monitor=cfg.fusion_training.early_stopping.monitor,
        mode=cfg.fusion_training.early_stopping.mode,
    )
    ckpt_filename = "best_fusion_diamos.pth" if dataset == "diamos" else "best_fusion_model.pth"
    checkpoint = ModelCheckpoint(
        save_dir=cfg.paths.checkpoint_dir,
        monitor="val_total_loss",
        mode="min",
        filename=ckpt_filename,
    )
    metrics_logger = MetricsLogger(log_dir=cfg.logging.tensorboard_dir)

    # Resume from checkpoint
    start_epoch = 0
    if resume and Path(resume).exists():
        logger.info(f"Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "multitask_loss_state" in ckpt:
            multitask_loss.load_state_dict(ckpt["multitask_loss_state"])
        if "scheduler_state_dict" in ckpt:
            lr_scheduler.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "early_stopping_state" in ckpt:
            es_state = ckpt["early_stopping_state"]
            early_stopping.best_score = es_state["best_score"]
            early_stopping.counter = es_state["counter"]
        if "best_val_total_loss" in ckpt:
            checkpoint.best_score = ckpt["best_val_total_loss"]
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch - 1}, continuing from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, cfg.fusion_training.epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{cfg.fusion_training.epochs} ---")

        train_loss = train_one_epoch(
            model, train_loader, criterion, multitask_loss, optimizer, device,
        )
        val_loss, val_disease_acc, val_outbreak_mae = validate(
            model, val_loader, criterion, multitask_loss, device,
        )

        lr_scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(
            f"Val Loss: {val_loss:.4f} | Disease Acc: {val_disease_acc:.4f} | "
            f"Outbreak MAE: {val_outbreak_mae:.4f} | LR: {current_lr:.2e}"
        )

        # Log to TensorBoard
        metrics_logger.log_scalar("fusion/train_loss", train_loss, epoch)
        metrics_logger.log_scalar("fusion/val_loss", val_loss, epoch)
        metrics_logger.log_scalar("fusion/val_disease_acc", val_disease_acc, epoch)
        metrics_logger.log_scalar("fusion/val_outbreak_mae", val_outbreak_mae, epoch)

        # MLflow metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_disease_acc": val_disease_acc,
            "val_outbreak_mae": val_outbreak_mae,
            "learning_rate": current_lr,
        }, step=epoch)

        # Checkpoint (save all states for resume)
        checkpoint(model, val_loss, epoch, optimizer, extra={
            "multitask_loss_state": multitask_loss.state_dict(),
            "scheduler_state_dict": lr_scheduler.scheduler.state_dict(),
            "early_stopping_state": {
                "best_score": early_stopping.best_score,
                "counter": early_stopping.counter,
            },
        })

        if early_stopping(val_loss):
            logger.info("Early stopping triggered.")
            break

    # Log model via mlflow.pytorch (enables serving + schema)
    mlflow.pytorch.log_model(model, "model")

    # Register model
    model_name = mlflow_cfg.get("registered_model_fusion", "MultiModalFusion")
    run_id = mlflow.active_run().info.run_id
    mlflow.register_model(f"runs:/{run_id}/model", model_name)

    mlflow.end_run()
    metrics_logger.close()
    logger.info("Fusion training finished.")

    best_loss = checkpoint.best_score if checkpoint.best_score is not None else float("inf")
    return best_loss


def main():
    args = parse_args()
    train(
        config_path=args.config,
        dataset=args.dataset,
        swin_checkpoint=args.swin_checkpoint,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

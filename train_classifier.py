"""Training script for Swin Transformer classifier on PlantVillage.

Training schedule:
    Epochs 1-10: Stages 0,1 frozen, lr=1e-4
    Epoch 10: Unfreeze all, lr *= 0.1
    Epochs 11-50: Full fine-tuning
    Optimizer: AdamW (wd=0.05)
    Scheduler: CosineAnnealingWarmRestarts
    Early stopping: patience=10 on val_f1
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.datamodule import PlantVillageDataModule
from src.features.augmentation import MixUpCutMix, get_train_transforms, get_val_transforms
from src.models.losses import FocalLoss
from src.models.swin_classifier import SwinClassifier
from src.utils.callbacks import EarlyStopping, LRSchedulerWrapper, MetricsLogger, ModelCheckpoint
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from src.utils.metrics import MetricsCalculator
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin Classifier")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config file path"
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    return parser.parse_args()


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, mixup_fn=None, grad_clip=1.0
):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # Apply MixUp/CutMix
        if mixup_fn is not None:
            images, labels_a, labels_b, lam = mixup_fn(
                images, labels, num_classes=model.num_classes
            )
            logits = model(images)
            loss = criterion.forward_mixup(logits, labels_a, labels_b, lam)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    all_probs = np.concatenate(all_probs, axis=0)
    return avg_loss, np.array(all_preds), np.array(all_labels), all_probs


def main():
    args = parse_args()
    cfg = ConfigManager(args.config)
    set_seed(cfg.seed)

    logger = setup_logger(
        "solinfitec",
        log_file=cfg.logging.log_file,
        level=cfg.logging.level,
    )
    logger.info("Starting Swin Classifier training")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Data
    data_dir = os.path.join(cfg.data.raw_dir, cfg.data.dataset_name)
    train_transform = get_train_transforms(img_size=cfg.data.img_size[0])
    val_transform = get_val_transforms(img_size=cfg.data.img_size[0])

    dm = PlantVillageDataModule(
        data_dir=data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_transform=train_transform,
        val_transform=val_transform,
        train_ratio=cfg.data.train_split,
        val_ratio=cfg.data.val_split,
        pin_memory=cfg.data.pin_memory,
        seed=cfg.seed,
    )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Model
    model = SwinClassifier(
        model_name=cfg.model.variant,
        num_classes=dm.num_classes,
        pretrained=cfg.model.pretrained,
        hidden_dim=cfg.model.head.hidden_dim,
        dropout=cfg.model.head.dropout,
        feature_dim=cfg.model.feature_dim,
    ).to(device)

    # Freeze initial stages
    model.freeze_stages(cfg.model.freeze.stages_frozen)

    logger.info(
        f"Model: {cfg.model.variant}, params={model.get_total_params():,}, "
        f"trainable={model.get_trainable_params():,}"
    )

    # Loss with class weights
    class_weights = dm.train_dataset.get_class_weights().to(device)
    criterion = FocalLoss(gamma=cfg.training.focal_gamma, alpha=class_weights)

    # MixUp/CutMix
    mixup_fn = MixUpCutMix(
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        prob=0.5,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Scheduler
    lr_scheduler = LRSchedulerWrapper(
        optimizer,
        scheduler_name=cfg.training.scheduler,
        scheduler_params={
            "T_0": cfg.training.scheduler_params.T_0,
            "T_mult": cfg.training.scheduler_params.T_mult,
            "eta_min": cfg.training.scheduler_params.eta_min,
        },
    )

    # Callbacks
    early_stopping = EarlyStopping(
        patience=cfg.training.early_stopping.patience,
        monitor=cfg.training.early_stopping.monitor,
        mode=cfg.training.early_stopping.mode,
    )
    checkpoint = ModelCheckpoint(
        save_dir=cfg.paths.checkpoint_dir,
        monitor="val_f1",
        mode="max",
        filename="best_swin_classifier.pth",
    )
    metrics_logger = MetricsLogger(log_dir=cfg.logging.tensorboard_dir)
    metrics_calc = MetricsCalculator(dm.class_names, dm.num_classes)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{cfg.training.epochs} ---")

        # Unfreeze at designated epoch
        if epoch == cfg.training.unfreeze_epoch:
            model.unfreeze_all()
            # Reduce learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] *= cfg.training.lr_reduction_factor
            logger.info(
                f"Unfreezing all layers, lr reduced to "
                f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        # Train
        train_loss, train_preds, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_fn=mixup_fn, grad_clip=cfg.training.grad_clip_norm,
        )
        train_metrics = metrics_calc.compute_all(train_labels, train_preds)

        # Validate
        val_loss, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )
        val_metrics = metrics_calc.compute_all(val_labels, val_preds, val_probs)

        # Scheduler step
        lr_scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        logger.info(
            f"Train Loss: {train_loss:.4f} | F1: {train_metrics['f1_macro']:.4f}"
        )
        logger.info(
            f"Val   Loss: {val_loss:.4f} | F1: {val_metrics['f1_macro']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | LR: {current_lr:.2e}"
        )

        metrics_logger.log_scalar("loss/train", train_loss, epoch)
        metrics_logger.log_scalar("loss/val", val_loss, epoch)
        metrics_logger.log_metrics(val_metrics, epoch, prefix="val/")
        metrics_logger.log_scalar("lr", current_lr, epoch)

        # Checkpoint
        val_f1 = val_metrics["f1_macro"]
        checkpoint(model, val_f1, epoch, optimizer)

        # Early stopping
        if early_stopping(val_f1):
            logger.info("Early stopping triggered. Training complete.")
            break

    metrics_logger.close()
    logger.info("Training finished.")


if __name__ == "__main__":
    main()

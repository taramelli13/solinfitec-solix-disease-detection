"""Training callbacks: EarlyStopping, ModelCheckpoint, LRScheduler, MetricsLogger."""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("solinfitec.callbacks")


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Epochs to wait before stopping.
        monitor: Metric to monitor.
        mode: 'min' or 'max'.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "val_f1",
        mode: str = "max",
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement on {self.monitor}"
                )

        return self.should_stop


class ModelCheckpoint:
    """Save model checkpoint when monitored metric improves.

    Args:
        save_dir: Directory for checkpoints.
        monitor: Metric to monitor.
        mode: 'min' or 'max'.
        filename: Checkpoint filename template.
    """

    def __init__(
        self,
        save_dir: str = "models/checkpoints",
        monitor: str = "val_f1",
        mode: str = "max",
        filename: str = "best_model.pth",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.best_score = None

    def __call__(
        self,
        model: nn.Module,
        score: float,
        epoch: int,
        optimizer=None,
        extra: Optional[Dict] = None,
    ) -> bool:
        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                f"best_{self.monitor}": score,
            }
            if optimizer:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            if extra:
                checkpoint.update(extra)

            save_path = self.save_dir / self.filename
            torch.save(checkpoint, save_path)
            logger.info(
                f"Checkpoint saved: {save_path} "
                f"({self.monitor}={score:.4f}, epoch={epoch})"
            )
            return True
        return False


class LRSchedulerWrapper:
    """Wrapper for learning rate schedulers with optional warmup.

    Args:
        optimizer: The optimizer.
        scheduler_name: Scheduler type.
        scheduler_params: Scheduler kwargs.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_name: str = "CosineAnnealingWarmRestarts",
        scheduler_params: Optional[Dict] = None,
    ):
        params = scheduler_params or {}

        if scheduler_name == "CosineAnnealingWarmRestarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.get("T_0", 10),
                T_mult=params.get("T_mult", 2),
                eta_min=params.get("eta_min", 1e-6),
            )
        elif scheduler_name == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=params.get("T_max", 50),
                eta_min=params.get("eta_min", 1e-6),
            )
        elif scheduler_name == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=params.get("step_size", 10),
                gamma=params.get("gamma", 0.1),
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.scheduler.step(epoch)
        else:
            self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


class MetricsLogger:
    """Log metrics to TensorBoard.

    Args:
        log_dir: TensorBoard log directory.
    """

    def __init__(self, log_dir: str = "logs/tensorboard"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self._available = True

    def _get_writer(self):
        if not self._available:
            return None
        if self.writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                logging.getLogger(__name__).warning(
                    "tensorboard not installed. Metrics logging disabled. "
                    "Install with: pip install tensorboard"
                )
                self._available = False
                return None
        return self.writer

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        writer = self._get_writer()
        if writer:
            writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        writer = self._get_writer()
        if writer:
            writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(self, tag: str, img_tensor, step: int) -> None:
        writer = self._get_writer()
        if writer:
            writer.add_image(tag, img_tensor, step)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        for name, value in metrics.items():
            self.log_scalar(f"{prefix}{name}", value, step)

    def close(self) -> None:
        if self.writer:
            self.writer.close()

"""DataModule: provides DataLoaders with WeightedRandomSampler for imbalanced data."""

import logging
from typing import Optional

from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import PlantVillageDataset

logger = logging.getLogger("solinfitec.datamodule")


class PlantVillageDataModule:
    """Creates train/val/test DataLoaders with balanced sampling.

    Args:
        data_dir: Path to PlantVillage root.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        train_transform: Transform for training set.
        val_transform: Transform for val/test set.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        pin_memory: Whether to pin memory for CUDA.
        seed: Random seed.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform=None,
        val_transform=None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = PlantVillageDataset(
            data_dir=data_dir,
            split="train",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            transform=train_transform,
            seed=seed,
        )
        self.val_dataset = PlantVillageDataset(
            data_dir=data_dir,
            split="val",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            transform=val_transform,
            seed=seed,
        )
        self.test_dataset = PlantVillageDataset(
            data_dir=data_dir,
            split="test",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            transform=val_transform,
            seed=seed,
        )

        logger.info(
            f"DataModule: train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )

    @property
    def num_classes(self) -> int:
        return len(self.train_dataset.class_names)

    @property
    def class_names(self):
        return self.train_dataset.class_names

    def train_dataloader(self) -> DataLoader:
        sample_weights = self.train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.train_dataset),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

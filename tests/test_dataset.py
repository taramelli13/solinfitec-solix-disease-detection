"""Tests for PlantVillage dataset and data pipeline."""

import numpy as np
import pytest
import torch

from src.data.dataset import PlantVillageDataset
from src.data.datamodule import PlantVillageDataModule
from src.data.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    def test_valid_class_dirs_skip_nested(self, tmp_dataset):
        preprocessor = DataPreprocessor(
            data_dir=str(tmp_dataset.parent),
            skip_nested="PlantVillage/PlantVillage",
        )
        class_dirs = preprocessor.get_valid_class_dirs()
        names = [d.name for d in class_dirs]
        assert "PlantVillage" not in names
        assert len(names) == 3

    def test_class_counts(self, tmp_dataset):
        preprocessor = DataPreprocessor(
            data_dir=str(tmp_dataset.parent),
            skip_nested="PlantVillage/PlantVillage",
        )
        counts = preprocessor.get_class_counts()
        assert counts["Tomato_healthy"] == 20
        assert counts["Tomato_Early_blight"] == 15
        assert counts["Potato___healthy"] == 10

    def test_detect_corrupted(self, tmp_dataset):
        preprocessor = DataPreprocessor(
            data_dir=str(tmp_dataset.parent),
            skip_nested="PlantVillage/PlantVillage",
        )
        corrupted = preprocessor.detect_corrupted()
        assert len(corrupted) == 0


class TestPlantVillageDataset:
    def test_split_sizes(self, tmp_dataset):
        total = 45  # 20 + 15 + 10
        train_ds = PlantVillageDataset(str(tmp_dataset), split="train", seed=42)
        val_ds = PlantVillageDataset(str(tmp_dataset), split="val", seed=42)
        test_ds = PlantVillageDataset(str(tmp_dataset), split="test", seed=42)

        assert len(train_ds) + len(val_ds) + len(test_ds) == total

    def test_no_overlap_between_splits(self, tmp_dataset):
        train_ds = PlantVillageDataset(str(tmp_dataset), split="train", seed=42)
        val_ds = PlantVillageDataset(str(tmp_dataset), split="val", seed=42)
        test_ds = PlantVillageDataset(str(tmp_dataset), split="test", seed=42)

        train_paths = set(train_ds.image_paths)
        val_paths = set(val_ds.image_paths)
        test_paths = set(test_ds.image_paths)

        assert len(train_paths & val_paths) == 0
        assert len(train_paths & test_paths) == 0
        assert len(val_paths & test_paths) == 0

    def test_skips_nested_directory(self, tmp_dataset):
        ds = PlantVillageDataset(str(tmp_dataset), split="train", seed=42)
        assert "PlantVillage" not in ds.class_names
        assert len(ds.class_names) == 3

    def test_getitem_returns_tensor_and_label(self, tmp_dataset):
        ds = PlantVillageDataset(str(tmp_dataset), split="train", seed=42)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 3
        assert isinstance(label, int)
        assert 0 <= label < len(ds.class_names)

    def test_class_weights(self, tmp_dataset):
        ds = PlantVillageDataset(str(tmp_dataset), split="train", seed=42)
        weights = ds.get_class_weights()
        assert weights.shape[0] == len(ds.class_names)
        assert (weights > 0).all()

    def test_sample_weights_length(self, tmp_dataset):
        ds = PlantVillageDataset(str(tmp_dataset), split="train", seed=42)
        sample_weights = ds.get_sample_weights()
        assert len(sample_weights) == len(ds)


class TestDataModule:
    def test_dataloaders(self, tmp_dataset):
        dm = PlantVillageDataModule(
            data_dir=str(tmp_dataset),
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        test_dl = dm.test_dataloader()

        batch = next(iter(train_dl))
        assert len(batch) == 2
        images, labels = batch
        assert images.shape[0] <= 4

    def test_num_classes(self, tmp_dataset):
        dm = PlantVillageDataModule(
            data_dir=str(tmp_dataset),
            batch_size=4,
            num_workers=0,
        )
        assert dm.num_classes == 3

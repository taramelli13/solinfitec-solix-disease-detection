"""Shared test fixtures."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a minimal fake PlantVillage dataset for testing."""
    classes = ["Tomato_healthy", "Tomato_Early_blight", "Potato___healthy"]
    counts = [20, 15, 10]

    dataset_dir = tmp_path / "PlantVillage"
    dataset_dir.mkdir()

    for cls_name, count in zip(classes, counts):
        cls_dir = dataset_dir / cls_name
        cls_dir.mkdir()
        for i in range(count):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(cls_dir / f"img_{i:04d}.jpg")

    # Create nested duplicate directory (should be skipped)
    nested = dataset_dir / "PlantVillage"
    nested.mkdir()
    dup_dir = nested / "Tomato_healthy"
    dup_dir.mkdir(parents=True)
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(dup_dir / "dup_img.jpg")

    return dataset_dir


@pytest.fixture
def config_path(tmp_path):
    """Create a minimal config file for testing."""
    config_content = """
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  img_size: [224, 224]
  batch_size: 4
  num_workers: 0
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  skip_nested: "PlantVillage/PlantVillage"

model:
  architecture: "swin_transformer"
  variant: "swin_tiny_patch4_window7_224"
  pretrained: false
  num_classes: 3
  feature_dim: 768

training:
  epochs: 2
  learning_rate: 0.001
  optimizer: "AdamW"

seed: 42
"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(config_content)
    return str(cfg_file)

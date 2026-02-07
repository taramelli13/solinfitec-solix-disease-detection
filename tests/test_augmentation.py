"""Tests for augmentation pipelines."""

import numpy as np
import pytest
import torch

from src.features.augmentation import (
    MixUpCutMix,
    get_minority_classes,
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)


class TestTransforms:
    def test_train_transforms_output_shape(self):
        transform = get_train_transforms(img_size=224)
        image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result = transform(image=image)
        tensor = result["image"]
        assert tensor.shape == (3, 224, 224)
        assert isinstance(tensor, torch.Tensor)

    def test_val_transforms_output_shape(self):
        transform = get_val_transforms(img_size=224, resize_size=256)
        image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result = transform(image=image)
        tensor = result["image"]
        assert tensor.shape == (3, 224, 224)

    def test_test_transforms_same_as_val(self):
        val_t = get_val_transforms(img_size=224)
        test_t = get_test_transforms(img_size=224)
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        np.random.seed(0)
        val_out = val_t(image=image)["image"]
        np.random.seed(0)
        test_out = test_t(image=image)["image"]
        assert torch.allclose(val_out, test_out)

    def test_minority_boost_transforms(self):
        transform = get_train_transforms(img_size=224, minority_boost=True)
        image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result = transform(image=image)
        assert result["image"].shape == (3, 224, 224)


class TestMixUpCutMix:
    def test_mixup_cutmix_shapes(self):
        mixer = MixUpCutMix(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0)
        images = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 10, (8,))

        mixed, la, lb, lam = mixer(images, labels, num_classes=10)
        assert mixed.shape == images.shape
        assert la.shape == labels.shape
        assert lb.shape == labels.shape
        assert 0 <= lam <= 1

    def test_no_augmentation_when_prob_zero(self):
        mixer = MixUpCutMix(prob=0.0)
        images = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 3, (4,))

        mixed, la, lb, lam = mixer(images, labels, num_classes=3)
        assert torch.allclose(mixed, images)
        assert lam == 1.0


class TestMinorityClasses:
    def test_identify_minority(self):
        counts = {
            "Potato___healthy": 152,
            "Tomato_healthy": 1593,
            "Tomato_mosaic_virus": 375,
        }
        minorities = get_minority_classes(counts, threshold=500)
        assert "Potato___healthy" in minorities
        assert "Tomato_mosaic_virus" in minorities
        assert "Tomato_healthy" not in minorities

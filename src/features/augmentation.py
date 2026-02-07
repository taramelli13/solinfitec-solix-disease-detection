"""Albumentations augmentation pipelines with minority class boosting."""

from typing import Dict, List, Optional

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    img_size: int = 224,
    crop_scale: tuple = (0.7, 1.0),
    minority_boost: bool = False,
) -> A.Compose:
    """Training augmentation pipeline.

    Args:
        img_size: Target image size.
        crop_scale: Scale range for RandomResizedCrop.
        minority_boost: If True, apply stronger augmentations (3x) for minority classes.
    """
    base_transforms = [
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=crop_scale,
            ratio=(0.8, 1.2),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.3,
        ),
    ]

    if minority_boost:
        base_transforms.extend([
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5
            ),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.GridDistortion(p=0.2),
        ])

    base_transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return A.Compose(base_transforms)


def get_val_transforms(
    img_size: int = 224,
    resize_size: int = 256,
) -> A.Compose:
    """Validation transform: resize + center crop + normalize."""
    return A.Compose([
        A.Resize(resize_size, resize_size),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_test_transforms(
    img_size: int = 224,
    resize_size: int = 256,
) -> A.Compose:
    """Test transforms (same as validation)."""
    return get_val_transforms(img_size=img_size, resize_size=resize_size)


class MixUpCutMix:
    """MixUp and CutMix applied at batch level during training.

    Args:
        mixup_alpha: MixUp alpha parameter (0 to disable).
        cutmix_alpha: CutMix alpha parameter (0 to disable).
        prob: Probability of applying either augmentation.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(self, images, labels, num_classes: int):
        """Apply MixUp or CutMix to a batch.

        Returns:
            images_mixed: Mixed images tensor.
            labels_a: Original labels.
            labels_b: Shuffled labels.
            lam: Mixing coefficient.
        """
        if np.random.rand() > self.prob:
            return images, labels, labels, 1.0

        batch_size = images.size(0)
        indices = np.random.permutation(batch_size)

        # Choose MixUp or CutMix randomly
        if np.random.rand() < 0.5 and self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            images_mixed = lam * images + (1 - lam) * images[indices]
        elif self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            images_mixed = images.clone()
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            images_mixed[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2))
        else:
            return images, labels, labels, 1.0

        import torch
        labels_a = labels
        labels_b = labels[torch.tensor(indices, dtype=torch.long, device=labels.device)]
        return images_mixed, labels_a, labels_b, lam

    @staticmethod
    def _rand_bbox(size, lam):
        """Generate random bounding box for CutMix."""
        H, W = size[2], size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


def get_minority_classes(
    class_counts: Dict[str, int],
    threshold: int = 500,
) -> List[str]:
    """Identify minority classes below the threshold."""
    return [name for name, count in class_counts.items() if count < threshold]

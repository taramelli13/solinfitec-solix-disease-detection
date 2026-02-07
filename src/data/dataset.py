"""PlantVillage dataset with stratified splits and class weighting."""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

logger = logging.getLogger("solinfitec.dataset")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class PlantVillageDataset(Dataset):
    """PlantVillage dataset loading only top-level class directories.

    Args:
        data_dir: Path to PlantVillage root (containing class subdirs).
        split: One of 'train', 'val', 'test'.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        transform: Albumentations or callable transform.
        skip_nested: Relative path of nested duplicate to skip.
        seed: Random seed for splitting.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        transform: Optional[Callable] = None,
        skip_nested: str = "PlantVillage",
        seed: int = 42,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Collect all image paths and labels from top-level class dirs
        self.class_names: List[str] = []
        all_paths: List[str] = []
        all_labels: List[int] = []

        for d in sorted(self.data_dir.iterdir()):
            if not d.is_dir():
                continue
            if d.name == skip_nested:
                continue
            self.class_names.append(d.name)

        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }
        self.idx_to_class: Dict[int, str] = {
            idx: name for name, idx in self.class_to_idx.items()
        }

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_dir / class_name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                    all_paths.append(str(img_path))
                    all_labels.append(class_idx)

        all_paths = np.array(all_paths)
        all_labels = np.array(all_labels)

        # Stratified split: train / (val + test)
        test_ratio = 1.0 - train_ratio - val_ratio
        train_idx, temp_idx = train_test_split(
            np.arange(len(all_paths)),
            test_size=val_ratio + test_ratio,
            stratify=all_labels,
            random_state=seed,
        )
        # Split temp into val / test
        temp_labels = all_labels[temp_idx]
        relative_val = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1.0 - relative_val,
            stratify=temp_labels,
            random_state=seed,
        )

        if split == "train":
            indices = train_idx
        elif split == "val":
            indices = val_idx
        else:
            indices = test_idx

        self.image_paths = all_paths[indices].tolist()
        self.labels = all_labels[indices].tolist()

        logger.info(
            f"PlantVillageDataset [{split}]: {len(self)} samples, "
            f"{len(self.class_names)} classes"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for WeightedRandomSampler."""
        counts = np.bincount(self.labels, minlength=len(self.class_names))
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(self.class_names)
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return torch.FloatTensor([class_weights[label] for label in self.labels])

    def get_label_counts(self) -> Dict[str, int]:
        """Get image count per class for this split."""
        from collections import Counter

        counter = Counter(self.labels)
        return {self.idx_to_class[k]: v for k, v in sorted(counter.items())}

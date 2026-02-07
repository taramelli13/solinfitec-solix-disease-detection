"""Data preprocessing: duplicate scanner, corruption detector, normalization stats."""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("solinfitec.preprocessing")


class DataPreprocessor:
    """Scans, validates, and computes statistics for the PlantVillage dataset."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, data_dir: str, skip_nested: str = "PlantVillage/PlantVillage"):
        self.data_dir = Path(data_dir)
        self.skip_nested = skip_nested

    def get_valid_class_dirs(self) -> List[Path]:
        """Return top-level class directories, skipping nested duplicates."""
        dataset_dir = self.data_dir / "PlantVillage"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        class_dirs = []
        for d in sorted(dataset_dir.iterdir()):
            if not d.is_dir():
                continue
            # Skip the nested duplicate directory
            rel = d.relative_to(self.data_dir)
            if str(rel) == self.skip_nested:
                logger.info(f"Skipping nested duplicate: {rel}")
                continue
            class_dirs.append(d)
        return class_dirs

    def get_class_counts(self) -> Dict[str, int]:
        """Count images per class (top-level only)."""
        counts = {}
        for class_dir in self.get_valid_class_dirs():
            n = sum(
                1
                for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS
            )
            counts[class_dir.name] = n
        return counts

    def scan_duplicates(self) -> Dict[str, List[str]]:
        """Find duplicate images by MD5 hash within the valid class dirs."""
        hash_to_files: Dict[str, List[str]] = {}
        for class_dir in self.get_valid_class_dirs():
            for img_path in class_dir.iterdir():
                if not img_path.is_file() or img_path.suffix.lower() not in self.VALID_EXTENSIONS:
                    continue
                md5 = hashlib.md5(img_path.read_bytes()).hexdigest()
                hash_to_files.setdefault(md5, []).append(str(img_path))

        duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
        if duplicates:
            total_dupes = sum(len(v) - 1 for v in duplicates.values())
            logger.warning(f"Found {total_dupes} duplicate images across {len(duplicates)} groups")
        else:
            logger.info("No duplicate images found")
        return duplicates

    def detect_corrupted(self) -> List[str]:
        """Find images that cannot be opened or are truncated."""
        corrupted = []
        for class_dir in self.get_valid_class_dirs():
            for img_path in class_dir.iterdir():
                if not img_path.is_file() or img_path.suffix.lower() not in self.VALID_EXTENSIONS:
                    continue
                try:
                    img = Image.open(img_path)
                    img.verify()
                except Exception:
                    corrupted.append(str(img_path))
                    logger.warning(f"Corrupted image: {img_path}")

        logger.info(
            f"Corruption scan complete: {len(corrupted)} corrupted out of total images"
        )
        return corrupted

    def compute_channel_stats(
        self,
        sample_size: Optional[int] = 2000,
    ) -> Tuple[List[float], List[float]]:
        """Compute per-channel mean and std over a sample of images."""
        all_images = []
        for class_dir in self.get_valid_class_dirs():
            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    all_images.append(img_path)

        if sample_size and sample_size < len(all_images):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(all_images), sample_size, replace=False)
            all_images = [all_images[i] for i in indices]

        pixel_sum = np.zeros(3, dtype=np.float64)
        pixel_sq_sum = np.zeros(3, dtype=np.float64)
        count = 0

        for img_path in tqdm(all_images, desc="Computing stats"):
            try:
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img, dtype=np.float64) / 255.0
                pixel_sum += arr.reshape(-1, 3).sum(axis=0)
                pixel_sq_sum += (arr.reshape(-1, 3) ** 2).sum(axis=0)
                count += arr.shape[0] * arr.shape[1]
            except Exception:
                continue

        mean = pixel_sum / count
        std = np.sqrt(pixel_sq_sum / count - mean ** 2)

        logger.info(f"Channel mean: {mean.tolist()}")
        logger.info(f"Channel std:  {std.tolist()}")
        return mean.tolist(), std.tolist()

    def get_image_sizes(self, sample_size: Optional[int] = 500) -> List[Tuple[int, int]]:
        """Get width x height for a sample of images."""
        all_images = []
        for class_dir in self.get_valid_class_dirs():
            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    all_images.append(img_path)

        if sample_size and sample_size < len(all_images):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(all_images), sample_size, replace=False)
            all_images = [all_images[i] for i in indices]

        sizes = []
        for img_path in all_images:
            try:
                with Image.open(img_path) as img:
                    sizes.append(img.size)
            except Exception:
                continue
        return sizes

"""DiaMOS Plant dataset for multi-modal training with real severity labels.

DiaMOS Plant contains 3,505 pear leaf images with:
- 4 disease classes: healthy, spot, curl, slug
- 5 severity levels: 0 (healthy), 1 (very low), 2 (low), 3 (medium), 4 (high)

We map the 5 DiaMOS severity levels to our 4-level system:
  DiaMOS 0 -> 0 (healthy)
  DiaMOS 1 -> 1 (initial)
  DiaMOS 2 -> 2 (moderate)
  DiaMOS 3,4 -> 3 (severe)

Since DiaMOS has no IoT sensor data, we generate synthetic IoT sequences
using the calibrated IoTSimulator.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger("solinfitec.diamos_dataset")

# DiaMOS disease classes
DIAMOS_CLASSES = ["healthy", "spot", "curl", "slug"]
DIAMOS_CLASS_TO_IDX = {c: i for i, c in enumerate(DIAMOS_CLASSES)}

# Severity mapping: DiaMOS 5-level -> our 4-level
SEVERITY_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}


def parse_diamos_annotations(data_dir: str) -> List[Dict]:
    """Parse DiaMOS dataset structure and return list of sample dicts.

    DiaMOS organizes images in subdirectories by disease class, with CSV
    annotation files containing severity information.

    Args:
        data_dir: Root directory of the extracted DiaMOS dataset.

    Returns:
        List of dicts with keys: image_path, disease_class, disease_idx, severity.
    """
    data_path = Path(data_dir)
    samples = []

    # Try to find annotation CSVs first
    csv_files = list(data_path.rglob("*.csv"))

    if csv_files:
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df.columns = df.columns.str.strip().str.lower()

                # Look for image path and severity columns
                img_col = None
                sev_col = None
                class_col = None

                for col in df.columns:
                    if "image" in col or "file" in col or "path" in col or "name" in col:
                        img_col = col
                    if "severity" in col or "grade" in col or "level" in col:
                        sev_col = col
                    if "class" in col or "disease" in col or "label" in col or "category" in col:
                        class_col = col

                if img_col is None:
                    continue

                for _, row in df.iterrows():
                    img_name = str(row[img_col]).strip()

                    # Resolve image path
                    img_path = csv_file.parent / img_name
                    if not img_path.exists():
                        # Try searching for the file
                        found = list(data_path.rglob(img_name))
                        if found:
                            img_path = found[0]
                        else:
                            continue

                    # Parse severity
                    severity_raw = int(row[sev_col]) if sev_col and pd.notna(row.get(sev_col)) else 0
                    severity = SEVERITY_MAP.get(severity_raw, min(severity_raw, 3))

                    # Parse disease class
                    if class_col and pd.notna(row.get(class_col)):
                        disease_str = str(row[class_col]).strip().lower()
                    else:
                        # Infer from directory name
                        disease_str = img_path.parent.name.lower()

                    disease_idx = DIAMOS_CLASS_TO_IDX.get(disease_str, 0)

                    samples.append({
                        "image_path": str(img_path),
                        "disease_class": disease_str,
                        "disease_idx": disease_idx,
                        "severity": severity,
                    })

            except Exception as e:
                logger.warning(f"Failed to parse {csv_file}: {e}")
                continue

    # Fallback: scan directories for images if no CSV annotations found
    if not samples:
        logger.info("No CSV annotations found, scanning directories...")
        image_exts = {".jpg", ".jpeg", ".png"}
        for class_name in DIAMOS_CLASSES:
            class_dir = data_path / class_name
            if not class_dir.exists():
                # Try case-insensitive search
                for d in data_path.iterdir():
                    if d.is_dir() and d.name.lower() == class_name:
                        class_dir = d
                        break

            if class_dir.exists():
                for img_file in sorted(class_dir.iterdir()):
                    if img_file.suffix.lower() in image_exts:
                        disease_idx = DIAMOS_CLASS_TO_IDX.get(class_name, 0)
                        # Without CSV, severity is unknown; default healthy=0, diseased=1
                        severity = 0 if class_name == "healthy" else 1

                        samples.append({
                            "image_path": str(img_file),
                            "disease_class": class_name,
                            "disease_idx": disease_idx,
                            "severity": severity,
                        })

    logger.info(f"Parsed {len(samples)} DiaMOS samples from {data_dir}")
    return samples


class DiaMOSMultiModalDataset(Dataset):
    """Multi-modal dataset wrapping DiaMOS images with synthetic IoT data.

    Returns the same dict format as MultiModalDataset:
        - image: (3, 224, 224) RGB tensor
        - iot_sequence: (30, 7) temporal features
        - geo: (3,) normalized lat/lon/elevation
        - disease_label: int class index
        - outbreak_target: (7,) daily risk
        - severity: int (0-3, from real DiaMOS labels)

    Args:
        data_dir: Root directory of DiaMOS dataset.
        iot_data: Dict of field_id -> DataFrame from IoTSimulator.
        geo_data: Dict of field_id -> (lat, lon, elevation).
        transform: Image transform (Albumentations).
        sequence_length: IoT sequence length in days.
        forecast_days: Outbreak forecast horizon.
    """

    IOT_FEATURES = [
        "temperature", "humidity", "soil_moisture",
        "wind_speed", "rain_mm", "disease_prevalence", "gdd",
    ]

    def __init__(
        self,
        data_dir: str,
        iot_data: Dict[int, pd.DataFrame],
        geo_data: Dict[int, Tuple[float, float, float]],
        transform: Optional[Callable] = None,
        sequence_length: int = 30,
        forecast_days: int = 7,
    ):
        self.samples = parse_diamos_annotations(data_dir)
        if not self.samples:
            raise RuntimeError(
                f"No samples found in DiaMOS dataset at {data_dir}. "
                "Run scripts/download_diamos.py first."
            )

        self.iot_data = iot_data
        self.geo_data = geo_data
        self.transform = transform
        self.sequence_length = sequence_length
        self.forecast_days = forecast_days

        # Round-robin field assignment
        field_ids = sorted(iot_data.keys())
        self.field_assignments = [
            field_ids[i % len(field_ids)] for i in range(len(self.samples))
        ]

        # Class distribution
        class_counts = {}
        for s in self.samples:
            class_counts[s["disease_class"]] = class_counts.get(s["disease_class"], 0) + 1
        logger.info(f"DiaMOS class distribution: {class_counts}")

        sev_counts = {}
        for s in self.samples:
            sev_counts[s["severity"]] = sev_counts.get(s["severity"], 0) + 1
        logger.info(f"DiaMOS severity distribution: {sev_counts}")

    @property
    def num_classes(self) -> int:
        return len(DIAMOS_CLASSES)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Image
        image = Image.open(sample["image_path"]).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Disease label
        label = sample["disease_idx"]

        # Real severity from DiaMOS annotations
        severity = sample["severity"]

        # IoT temporal sequence (synthetic, from calibrated simulator)
        field_id = self.field_assignments[idx]
        iot_df = self.iot_data[field_id]

        max_start = len(iot_df) - self.sequence_length - self.forecast_days
        if max_start > 0:
            start_idx = np.random.randint(0, max_start)
        else:
            start_idx = 0

        end_idx = start_idx + self.sequence_length
        iot_window = iot_df.iloc[start_idx:end_idx]

        iot_features = []
        for feat in self.IOT_FEATURES:
            if feat in iot_window.columns:
                iot_features.append(iot_window[feat].values)
            else:
                iot_features.append(np.zeros(self.sequence_length))

        iot_sequence = np.stack(iot_features, axis=1).astype(np.float32)

        if iot_sequence.shape[0] < self.sequence_length:
            pad = np.zeros(
                (self.sequence_length - iot_sequence.shape[0], iot_sequence.shape[1]),
                dtype=np.float32,
            )
            iot_sequence = np.concatenate([pad, iot_sequence], axis=0)

        # Outbreak target
        forecast_window = iot_df.iloc[end_idx : end_idx + self.forecast_days]
        if "disease_prevalence" in forecast_window.columns and len(forecast_window) == self.forecast_days:
            outbreak_target = forecast_window["disease_prevalence"].values.astype(np.float32)
        else:
            outbreak_target = np.zeros(self.forecast_days, dtype=np.float32)

        # Geo
        geo = self.geo_data.get(field_id, (0.0, 0.0, 0.0))
        geo_tensor = torch.FloatTensor([
            geo[0] / 90.0,
            geo[1] / 180.0,
            geo[2] / 5000.0,
        ])

        return {
            "image": image,
            "iot_sequence": torch.FloatTensor(iot_sequence),
            "geo": geo_tensor,
            "disease_label": torch.LongTensor([label]).squeeze(),
            "outbreak_target": torch.FloatTensor(outbreak_target),
            "severity": torch.LongTensor([severity]).squeeze(),
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights."""
        counts = np.zeros(self.num_classes)
        for s in self.samples:
            counts[s["disease_idx"]] += 1
        counts = np.maximum(counts, 1)  # avoid division by zero
        weights = 1.0 / counts
        weights = weights / weights.sum() * self.num_classes
        return torch.FloatTensor(weights)

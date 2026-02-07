"""Multi-modal dataset combining images, IoT time series, and geospatial data."""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger("solinfitec.multimodal_dataset")


class MultiModalDataset(Dataset):
    """Dataset that combines visual, temporal, and spatial modalities.

    Each sample contains:
        - image: (3, 224, 224) RGB image
        - iot_sequence: (30, 7) temporal features for 30 days
        - geo: (3,) normalized lat/lon/elevation
        - disease_label: int class index
        - outbreak_target: (7,) daily risk for next 7 days
        - severity: int (0=healthy, 1=initial, 2=moderate, 3=severe)

    Args:
        image_paths: List of image file paths.
        labels: List of class indices.
        iot_data: Dict mapping field_id -> DataFrame with temporal data.
        geo_data: Dict mapping field_id -> (lat, lon, elevation).
        field_assignments: List mapping each sample to a field_id.
        sequence_length: Number of past days for IoT sequence.
        forecast_days: Number of future days for outbreak prediction.
        transform: Image transform (Albumentations).
        iot_features: Column names for IoT features.
    """

    IOT_FEATURES = [
        "temperature", "humidity", "soil_moisture",
        "wind_speed", "rain_mm", "disease_prevalence", "gdd",
    ]

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        iot_data: Dict[int, pd.DataFrame],
        geo_data: Dict[int, Tuple[float, float, float]],
        field_assignments: Optional[List[int]] = None,
        sequence_length: int = 30,
        forecast_days: int = 7,
        transform: Optional[Callable] = None,
        iot_features: Optional[List[str]] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.iot_data = iot_data
        self.geo_data = geo_data
        self.sequence_length = sequence_length
        self.forecast_days = forecast_days
        self.transform = transform
        self.iot_features = iot_features or self.IOT_FEATURES

        # Assign samples to fields (round-robin if not provided)
        if field_assignments is not None:
            self.field_assignments = field_assignments
        else:
            field_ids = sorted(iot_data.keys())
            self.field_assignments = [
                field_ids[i % len(field_ids)] for i in range(len(image_paths))
            ]

        logger.info(
            f"MultiModalDataset: {len(self)} samples, "
            f"{len(iot_data)} fields, seq_len={sequence_length}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Disease label
        label = self.labels[idx]

        # IoT temporal sequence
        field_id = self.field_assignments[idx]
        iot_df = self.iot_data[field_id]

        # Select a random time window for the IoT data
        max_start = len(iot_df) - self.sequence_length - self.forecast_days
        if max_start > 0:
            start_idx = np.random.randint(0, max_start)
        else:
            start_idx = 0

        end_idx = start_idx + self.sequence_length
        iot_window = iot_df.iloc[start_idx:end_idx]

        # Extract features
        iot_features = []
        for feat in self.iot_features:
            if feat in iot_window.columns:
                iot_features.append(iot_window[feat].values)
            else:
                iot_features.append(np.zeros(self.sequence_length))

        iot_sequence = np.stack(iot_features, axis=1).astype(np.float32)

        # Pad if necessary
        if iot_sequence.shape[0] < self.sequence_length:
            pad = np.zeros(
                (self.sequence_length - iot_sequence.shape[0], iot_sequence.shape[1]),
                dtype=np.float32,
            )
            iot_sequence = np.concatenate([pad, iot_sequence], axis=0)

        # Outbreak target (next 7 days disease prevalence)
        forecast_window = iot_df.iloc[end_idx : end_idx + self.forecast_days]
        if "disease_prevalence" in forecast_window.columns and len(forecast_window) == self.forecast_days:
            outbreak_target = forecast_window["disease_prevalence"].values.astype(np.float32)
        else:
            outbreak_target = np.zeros(self.forecast_days, dtype=np.float32)

        # Severity (derived from current disease prevalence)
        current_prevalence = iot_df.iloc[end_idx - 1]["disease_prevalence"] if end_idx > 0 else 0
        severity = self._prevalence_to_severity(current_prevalence)

        # Geospatial features
        geo = self.geo_data.get(field_id, (0.0, 0.0, 0.0))
        geo_tensor = torch.FloatTensor([
            geo[0] / 90.0,    # lat normalized
            geo[1] / 180.0,   # lon normalized
            geo[2] / 5000.0,  # elevation normalized
        ])

        return {
            "image": image,
            "iot_sequence": torch.FloatTensor(iot_sequence),
            "geo": geo_tensor,
            "disease_label": torch.LongTensor([label]).squeeze(),
            "outbreak_target": torch.FloatTensor(outbreak_target),
            "severity": torch.LongTensor([severity]).squeeze(),
        }

    @staticmethod
    def _prevalence_to_severity(prevalence: float) -> int:
        """Map disease prevalence to severity stage."""
        if prevalence < 0.05:
            return 0  # healthy
        elif prevalence < 0.15:
            return 1  # initial
        elif prevalence < 0.35:
            return 2  # moderate
        else:
            return 3  # severe

    @staticmethod
    def create_from_datasets(
        plant_dataset,
        iot_data: Dict[int, pd.DataFrame],
        geo_data: Dict[int, Tuple[float, float, float]],
        transform=None,
        sequence_length: int = 30,
        forecast_days: int = 7,
    ) -> "MultiModalDataset":
        """Create MultiModalDataset from an existing PlantVillageDataset."""
        return MultiModalDataset(
            image_paths=plant_dataset.image_paths,
            labels=plant_dataset.labels,
            iot_data=iot_data,
            geo_data=geo_data,
            transform=transform,
            sequence_length=sequence_length,
            forecast_days=forecast_days,
        )

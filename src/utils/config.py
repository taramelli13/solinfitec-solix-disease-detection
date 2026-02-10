"""Configuration manager with typed dataclass access."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    external_dir: str = "data/external"
    dataset_name: str = "PlantVillage"
    img_size: List[int] = field(default_factory=lambda: [224, 224])
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    pin_memory: bool = True
    skip_nested: str = "PlantVillage/PlantVillage"
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class HeadConfig:
    hidden_dim: int = 256
    dropout: float = 0.3
    activation: str = "gelu"


@dataclass
class FreezeConfig:
    stages_frozen: List[int] = field(default_factory=lambda: [0, 1])
    unfreeze_epoch: int = 10


@dataclass
class ModelConfig:
    architecture: str = "swin_transformer"
    variant: str = "swin_tiny_patch4_window7_224"
    pretrained: bool = True
    num_classes: int = 15
    feature_dim: int = 768
    head: HeadConfig = field(default_factory=HeadConfig)
    freeze: FreezeConfig = field(default_factory=FreezeConfig)


@dataclass
class TemporalModelConfig:
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    sequence_length: int = 30
    num_features: int = 7
    output_dim: int = 256


@dataclass
class SpatialModelConfig:
    input_dim: int = 3
    hidden_dim: int = 64
    output_dim: int = 128


@dataclass
class CrossAttentionConfig:
    nhead: int = 8
    dropout: float = 0.1


@dataclass
class FusionConfig:
    visual_proj_dim: int = 256
    temporal_dim: int = 256
    spatial_dim: int = 128
    fused_dim: int = 640
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    gated_fusion: bool = True


@dataclass
class EarlyStoppingConfig:
    patience: int = 10
    monitor: str = "val_f1"
    mode: str = "max"


@dataclass
class SchedulerParamsConfig:
    T_0: int = 10
    T_mult: int = 2
    eta_min: float = 1e-6


@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-4
    optimizer: str = "AdamW"
    weight_decay: float = 0.05
    scheduler: str = "CosineAnnealingWarmRestarts"
    scheduler_params: SchedulerParamsConfig = field(default_factory=SchedulerParamsConfig)
    unfreeze_epoch: int = 10
    lr_reduction_factor: float = 0.1
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    grad_clip_norm: float = 1.0
    loss: str = "focal"
    focal_gamma: float = 2.0


@dataclass
class FusionTrainingConfig:
    epochs: int = 80
    learning_rate: float = 3e-4
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    scheduler: str = "CosineAnnealingWarmRestarts"
    scheduler_params: SchedulerParamsConfig = field(
        default_factory=lambda: SchedulerParamsConfig(T_0=15)
    )
    freeze_swin_backbone: bool = True
    early_stopping: EarlyStoppingConfig = field(
        default_factory=lambda: EarlyStoppingConfig(
            patience=15, monitor="val_total_loss", mode="min"
        )
    )


@dataclass
class AugTrainConfig:
    random_resized_crop: Dict[str, Any] = field(
        default_factory=lambda: {"size": 224, "scale": [0.7, 1.0]}
    )
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.2
    color_jitter: Dict[str, float] = field(
        default_factory=lambda: {
            "brightness": 0.3, "contrast": 0.3,
            "saturation": 0.3, "hue": 0.1,
        }
    )
    gaussian_noise: Dict[str, Any] = field(
        default_factory=lambda: {"var_limit": [10, 50], "p": 0.3}
    )
    coarse_dropout: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_holes": 8, "max_height": 32, "max_width": 32, "p": 0.3,
        }
    )
    mixup: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "alpha": 0.2}
    )
    cutmix: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "alpha": 1.0}
    )


@dataclass
class AugValConfig:
    resize: int = 256
    center_crop: int = 224


@dataclass
class AugmentationConfig:
    train: AugTrainConfig = field(default_factory=AugTrainConfig)
    minority_threshold: int = 500
    minority_multiplier: int = 3
    val: AugValConfig = field(default_factory=AugValConfig)


@dataclass
class ExportConfig:
    format: str = "onnx"
    opset_version: int = 14
    quantization: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": False, "type": "fp16"}
    )
    target_device: str = "jetson_xavier"
    validate_output: bool = True
    max_diff: float = 1e-5
    benchmark_latency: bool = True
    output_dir: str = "models/final"


@dataclass
class AlertsConfig:
    risk_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "low": 0.2, "medium": 0.5, "high": 0.75, "critical": 0.9,
        }
    )
    severity_labels: List[str] = field(
        default_factory=lambda: ["healthy", "initial", "moderate", "severe"]
    )
    actions: Dict[str, str] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = "logs/training.log"
    use_tensorboard: bool = True
    tensorboard_dir: str = "logs/tensorboard"


@dataclass
class PathsConfig:
    checkpoint_dir: str = "models/checkpoints"
    final_model_dir: str = "models/final"
    log_dir: str = "logs"
    report_dir: str = "reports"
    metrics_dir: str = "reports/metrics"


def _nested_dataclass_from_dict(cls, data: dict):
    """Recursively instantiate nested dataclasses from a dict."""
    if data is None:
        return cls()
    import dataclasses

    fieldtypes = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in data.items():
        if k in fieldtypes:
            ft = fieldtypes[k]
            # Resolve string annotations
            if isinstance(ft, str):
                ft = eval(ft)
            if dataclasses.is_dataclass(ft) and isinstance(v, dict):
                kwargs[k] = _nested_dataclass_from_dict(ft, v)
            else:
                kwargs[k] = v
    return cls(**kwargs)


class ConfigManager:
    """Loads config.yaml and provides typed dataclass access."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self._raw: Dict[str, Any] = yaml.safe_load(f)

        self.data = _nested_dataclass_from_dict(DataConfig, self._raw.get("data", {}))
        self.model = _nested_dataclass_from_dict(ModelConfig, self._raw.get("model", {}))
        self.temporal_model = _nested_dataclass_from_dict(
            TemporalModelConfig, self._raw.get("temporal_model", {})
        )
        self.spatial_model = _nested_dataclass_from_dict(
            SpatialModelConfig, self._raw.get("spatial_model", {})
        )
        self.fusion = _nested_dataclass_from_dict(FusionConfig, self._raw.get("fusion", {}))
        self.training = _nested_dataclass_from_dict(
            TrainingConfig, self._raw.get("training", {})
        )
        self.fusion_training = _nested_dataclass_from_dict(
            FusionTrainingConfig, self._raw.get("fusion_training", {})
        )
        self.augmentation = _nested_dataclass_from_dict(
            AugmentationConfig, self._raw.get("augmentation", {})
        )
        self.export = _nested_dataclass_from_dict(ExportConfig, self._raw.get("export", {}))
        self.alerts = _nested_dataclass_from_dict(AlertsConfig, self._raw.get("alerts", {}))
        self.logging = _nested_dataclass_from_dict(
            LoggingConfig, self._raw.get("logging", {})
        )
        self.paths = _nested_dataclass_from_dict(PathsConfig, self._raw.get("paths", {}))
        self.seed: int = self._raw.get("seed", 42)

    def with_overrides(self, overrides: Dict[str, Any]) -> "ConfigManager":
        """Return a new ConfigManager with overrides applied.

        Keys use dot notation: "training.learning_rate", "data.batch_size".
        """
        import copy

        raw = copy.deepcopy(self._raw)
        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            d = raw
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

        new = ConfigManager.__new__(ConfigManager)
        new.config_path = self.config_path
        new._raw = raw
        new.data = _nested_dataclass_from_dict(DataConfig, raw.get("data", {}))
        new.model = _nested_dataclass_from_dict(ModelConfig, raw.get("model", {}))
        new.temporal_model = _nested_dataclass_from_dict(
            TemporalModelConfig, raw.get("temporal_model", {})
        )
        new.spatial_model = _nested_dataclass_from_dict(
            SpatialModelConfig, raw.get("spatial_model", {})
        )
        new.fusion = _nested_dataclass_from_dict(FusionConfig, raw.get("fusion", {}))
        new.training = _nested_dataclass_from_dict(
            TrainingConfig, raw.get("training", {})
        )
        new.fusion_training = _nested_dataclass_from_dict(
            FusionTrainingConfig, raw.get("fusion_training", {})
        )
        new.augmentation = _nested_dataclass_from_dict(
            AugmentationConfig, raw.get("augmentation", {})
        )
        new.export = _nested_dataclass_from_dict(ExportConfig, raw.get("export", {}))
        new.alerts = _nested_dataclass_from_dict(AlertsConfig, raw.get("alerts", {}))
        new.logging = _nested_dataclass_from_dict(
            LoggingConfig, raw.get("logging", {})
        )
        new.paths = _nested_dataclass_from_dict(PathsConfig, raw.get("paths", {}))
        new.seed = raw.get("seed", 42)
        return new

    def get_raw(self, key: str, default: Any = None) -> Any:
        """Access raw config dict by dot-separated key."""
        keys = key.split(".")
        val = self._raw
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    def __repr__(self) -> str:
        return f"ConfigManager(path={self.config_path})"

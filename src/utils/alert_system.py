"""Alert generation system for disease detection and outbreak prediction."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("solinfitec.alert_system")


@dataclass
class Alert:
    """Structured alert for disease detection."""

    timestamp: str
    field_id: str
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    disease_name: str
    disease_confidence: float
    outbreak_risk_7d: List[float]  # Daily risk for 7 days
    severity_stage: str  # healthy, initial, moderate, severe
    severity_confidence: float
    recommended_action: str
    gradcam_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def to_dict(self) -> Dict:
        return asdict(self)


class AlertGenerator:
    """Generates structured alerts from model predictions.

    Args:
        risk_thresholds: Dict with threshold values for LOW/MEDIUM/HIGH/CRITICAL.
        severity_labels: List of severity stage labels.
        actions: Dict mapping risk levels to recommended actions.
        class_names: List of disease class names.
    """

    def __init__(
        self,
        risk_thresholds: Optional[Dict[str, float]] = None,
        severity_labels: Optional[List[str]] = None,
        actions: Optional[Dict[str, str]] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.risk_thresholds = risk_thresholds or {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.75,
            "critical": 0.9,
        }
        self.severity_labels = severity_labels or [
            "healthy", "initial", "moderate", "severe"
        ]
        self.actions = actions or {
            "LOW": "Continue regular monitoring.",
            "MEDIUM": "Increase inspection frequency. Consider preventive treatment.",
            "HIGH": "Apply targeted treatment immediately. Isolate affected area.",
            "CRITICAL": "Emergency intervention required. Full field treatment recommended.",
        }
        self.class_names = class_names or []

    def _determine_risk_level(self, max_outbreak_risk: float) -> str:
        """Determine risk level from maximum outbreak risk score."""
        if max_outbreak_risk >= self.risk_thresholds["critical"]:
            return "CRITICAL"
        elif max_outbreak_risk >= self.risk_thresholds["high"]:
            return "HIGH"
        elif max_outbreak_risk >= self.risk_thresholds["medium"]:
            return "MEDIUM"
        return "LOW"

    def generate_alert(
        self,
        field_id: str,
        disease_probs: List[float],
        outbreak_risk_7d: List[float],
        severity_probs: List[float],
        gradcam_path: Optional[str] = None,
    ) -> Alert:
        """Generate a single alert from model predictions.

        Args:
            field_id: Identifier for the field.
            disease_probs: (num_classes,) softmax probabilities.
            outbreak_risk_7d: (7,) daily risk predictions.
            severity_probs: (4,) severity stage probabilities/logits.
            gradcam_path: Path to Grad-CAM visualization.

        Returns:
            Alert object.
        """
        # Disease
        disease_idx = int(max(range(len(disease_probs)), key=lambda i: disease_probs[i]))
        disease_name = (
            self.class_names[disease_idx] if disease_idx < len(self.class_names) else f"class_{disease_idx}"
        )
        disease_confidence = disease_probs[disease_idx]

        # Outbreak risk
        max_risk = max(outbreak_risk_7d)
        risk_level = self._determine_risk_level(max_risk)

        # Severity
        severity_idx = int(max(range(len(severity_probs)), key=lambda i: severity_probs[i]))
        severity_stage = (
            self.severity_labels[severity_idx]
            if severity_idx < len(self.severity_labels)
            else f"stage_{severity_idx}"
        )
        severity_confidence = severity_probs[severity_idx]

        # Action
        action = self.actions.get(risk_level, "Monitor the situation.")

        alert = Alert(
            timestamp=datetime.now().isoformat(),
            field_id=field_id,
            risk_level=risk_level,
            disease_name=disease_name,
            disease_confidence=round(disease_confidence, 4),
            outbreak_risk_7d=[round(r, 4) for r in outbreak_risk_7d],
            severity_stage=severity_stage,
            severity_confidence=round(severity_confidence, 4),
            recommended_action=action,
            gradcam_path=gradcam_path,
            metadata={
                "max_outbreak_risk": round(max_risk, 4),
                "peak_risk_day": int(outbreak_risk_7d.index(max_risk)) + 1,
            },
        )

        logger.info(
            f"Alert [{risk_level}] Field={field_id}: {disease_name} "
            f"(conf={disease_confidence:.2f}), severity={severity_stage}, "
            f"max_risk={max_risk:.3f} on day {alert.metadata['peak_risk_day']}"
        )

        return alert

    def generate_batch_alerts(
        self,
        field_ids: List[str],
        disease_probs_batch: List[List[float]],
        outbreak_risk_batch: List[List[float]],
        severity_probs_batch: List[List[float]],
    ) -> List[Alert]:
        """Generate alerts for a batch of predictions."""
        alerts = []
        for fid, dp, orisk, sp in zip(
            field_ids, disease_probs_batch, outbreak_risk_batch, severity_probs_batch
        ):
            alert = self.generate_alert(fid, dp, orisk, sp)
            alerts.append(alert)
        return alerts

    def filter_alerts(
        self, alerts: List[Alert], min_risk_level: str = "MEDIUM"
    ) -> List[Alert]:
        """Filter alerts by minimum risk level."""
        levels = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        min_level = levels.get(min_risk_level, 0)
        return [a for a in alerts if levels.get(a.risk_level, 0) >= min_level]

    def export_alerts(self, alerts: List[Alert], filepath: str) -> None:
        """Export alerts to a JSON file."""
        data = [a.to_dict() for a in alerts]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported {len(alerts)} alerts to {filepath}")

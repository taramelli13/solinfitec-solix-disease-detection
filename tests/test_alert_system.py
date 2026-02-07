"""Tests for alert system and MQTT interface."""

import json

import pytest

from src.data.mqtt_interface import (
    AlertMessage,
    MockMQTTBroker,
    MQTTInterface,
    PredictionMessage,
    SensorMessage,
)
from src.utils.alert_system import Alert, AlertGenerator


class TestAlertGenerator:
    @pytest.fixture
    def generator(self):
        return AlertGenerator(
            class_names=["Tomato_healthy", "Tomato_Early_blight", "Potato___healthy"],
        )

    def test_generate_alert(self, generator):
        alert = generator.generate_alert(
            field_id="field_001",
            disease_probs=[0.1, 0.8, 0.1],
            outbreak_risk_7d=[0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.6],
            severity_probs=[0.1, 0.2, 0.5, 0.2],
        )
        assert alert.disease_name == "Tomato_Early_blight"
        assert alert.disease_confidence == 0.8
        assert alert.risk_level == "HIGH"
        assert alert.severity_stage == "moderate"
        assert len(alert.outbreak_risk_7d) == 7

    def test_low_risk_alert(self, generator):
        alert = generator.generate_alert(
            field_id="field_002",
            disease_probs=[0.9, 0.05, 0.05],
            outbreak_risk_7d=[0.05, 0.08, 0.1, 0.1, 0.08, 0.05, 0.03],
            severity_probs=[0.9, 0.05, 0.03, 0.02],
        )
        assert alert.risk_level == "LOW"
        assert alert.severity_stage == "healthy"

    def test_critical_alert(self, generator):
        alert = generator.generate_alert(
            field_id="field_003",
            disease_probs=[0.0, 0.95, 0.05],
            outbreak_risk_7d=[0.7, 0.8, 0.9, 0.95, 0.92, 0.88, 0.85],
            severity_probs=[0.0, 0.05, 0.15, 0.80],
        )
        assert alert.risk_level == "CRITICAL"
        assert alert.severity_stage == "severe"

    def test_alert_to_json(self, generator):
        alert = generator.generate_alert(
            field_id="field_001",
            disease_probs=[0.1, 0.8, 0.1],
            outbreak_risk_7d=[0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.6],
            severity_probs=[0.1, 0.2, 0.5, 0.2],
        )
        json_str = alert.to_json()
        data = json.loads(json_str)
        assert "risk_level" in data
        assert "disease_name" in data
        assert "outbreak_risk_7d" in data

    def test_filter_alerts(self, generator):
        alerts = [
            generator.generate_alert(
                "f1", [0.9, 0.05, 0.05], [0.1] * 7, [0.9, 0.05, 0.03, 0.02]
            ),
            generator.generate_alert(
                "f2", [0.1, 0.8, 0.1], [0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4],
                [0.1, 0.2, 0.5, 0.2],
            ),
        ]
        filtered = generator.filter_alerts(alerts, min_risk_level="HIGH")
        assert len(filtered) == 1
        assert filtered[0].field_id == "f2"

    def test_export_alerts(self, generator, tmp_path):
        alerts = [
            generator.generate_alert(
                "f1", [0.9, 0.05, 0.05], [0.1] * 7, [0.9, 0.05, 0.03, 0.02]
            ),
        ]
        filepath = str(tmp_path / "alerts.json")
        generator.export_alerts(alerts, filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["field_id"] == "f1"


class TestMQTTMessages:
    def test_sensor_message_roundtrip(self):
        msg = SensorMessage(
            field_id="field_001",
            timestamp="2024-01-01T12:00:00",
            temperature=25.5,
            humidity=70.0,
            soil_moisture=0.35,
            wind_speed=5.0,
            rain_mm=0.0,
        )
        json_str = msg.to_json()
        restored = SensorMessage.from_json(json_str)
        assert restored.field_id == "field_001"
        assert restored.temperature == 25.5

    def test_alert_message_json(self):
        msg = AlertMessage(
            field_id="field_001",
            timestamp="2024-01-01T12:00:00",
            risk_level="HIGH",
            disease_name="Tomato_Early_blight",
            confidence=0.85,
            severity="moderate",
            action="Apply treatment.",
            outbreak_risk_7d=[0.3, 0.5, 0.7, 0.8, 0.7, 0.5, 0.3],
        )
        data = json.loads(msg.to_json())
        assert data["risk_level"] == "HIGH"


class TestMockBroker:
    def test_publish_subscribe(self):
        broker = MockMQTTBroker()
        broker.connect()

        received = []
        broker.subscribe("solix/alerts/+", lambda t, p: received.append((t, p)))
        broker.publish("solix/alerts/field_001", '{"test": true}')

        assert len(received) == 1
        assert received[0][0] == "solix/alerts/field_001"

    def test_topic_matching(self):
        assert MockMQTTBroker._topic_matches("solix/+/field_001", "solix/sensors/field_001")
        assert MockMQTTBroker._topic_matches("solix/#", "solix/sensors/field_001")
        assert not MockMQTTBroker._topic_matches("solix/sensors/field_001", "solix/alerts/field_001")


class TestMQTTInterface:
    def test_publish_alert(self):
        interface = MQTTInterface(use_mock=True)
        interface.connect()

        alert = AlertMessage(
            field_id="field_001",
            timestamp="2024-01-01",
            risk_level="HIGH",
            disease_name="blight",
            confidence=0.9,
            severity="moderate",
            action="treat",
        )
        interface.publish_alert("field_001", alert)

        msgs = interface.client.get_messages()
        assert len(msgs) == 1
        assert "field_001" in msgs[0]["topic"]

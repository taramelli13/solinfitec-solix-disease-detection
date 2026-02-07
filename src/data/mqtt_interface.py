"""MQTT interface for Solix sensor communication.

Defines message schemas compatible with the Solix platform:
    - SensorMessage: incoming sensor data
    - AlertMessage: outgoing alerts
    - PredictionMessage: outgoing predictions
Includes a mock broker for testing.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("solinfitec.mqtt_interface")


@dataclass
class SensorMessage:
    """Incoming sensor data from Solix device."""

    field_id: str
    timestamp: str
    temperature: float
    humidity: float
    soil_moisture: float
    wind_speed: float
    rain_mm: float
    latitude: float = 0.0
    longitude: float = 0.0
    elevation: float = 0.0
    device_id: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, payload: str) -> "SensorMessage":
        data = json.loads(payload)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AlertMessage:
    """Outgoing alert message."""

    field_id: str
    timestamp: str
    risk_level: str
    disease_name: str
    confidence: float
    severity: str
    action: str
    outbreak_risk_7d: List[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class PredictionMessage:
    """Outgoing prediction result."""

    field_id: str
    timestamp: str
    disease_class: str
    disease_confidence: float
    outbreak_risk: List[float]
    severity_stage: str
    severity_confidence: float
    model_version: str = "1.0.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class MockMQTTBroker:
    """Mock MQTT broker for testing without external dependencies.

    Supports publish/subscribe with topic matching.
    """

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.messages: List[Dict[str, Any]] = []
        self.connected = False

    def connect(self, host: str = "localhost", port: int = 1883) -> None:
        self.connected = True
        logger.info(f"MockBroker connected to {host}:{port}")

    def disconnect(self) -> None:
        self.connected = False
        logger.info("MockBroker disconnected")

    def subscribe(self, topic: str, callback: Callable) -> None:
        self.subscribers.setdefault(topic, []).append(callback)
        logger.info(f"Subscribed to {topic}")

    def publish(self, topic: str, payload: str, qos: int = 1) -> None:
        if not self.connected:
            raise ConnectionError("Not connected to broker")

        message = {
            "topic": topic,
            "payload": payload,
            "qos": qos,
            "timestamp": time.time(),
        }
        self.messages.append(message)

        # Deliver to subscribers
        for sub_topic, callbacks in self.subscribers.items():
            if self._topic_matches(sub_topic, topic):
                for cb in callbacks:
                    cb(topic, payload)

        logger.debug(f"Published to {topic}: {payload[:100]}...")

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        """Simple MQTT topic matching with wildcards."""
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")

        for p, t in zip(pattern_parts, topic_parts):
            if p == "#":
                return True
            if p == "+":
                continue
            if p != t:
                return False

        return len(pattern_parts) == len(topic_parts)

    def get_messages(self, topic: Optional[str] = None) -> List[Dict]:
        if topic:
            return [m for m in self.messages if m["topic"] == topic]
        return self.messages

    def clear(self) -> None:
        self.messages.clear()


class MQTTInterface:
    """MQTT interface for Solix integration.

    Args:
        broker_host: MQTT broker hostname.
        broker_port: MQTT broker port.
        topics: Dict of topic templates.
        use_mock: Use MockMQTTBroker instead of real paho-mqtt.
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topics: Optional[Dict[str, str]] = None,
        use_mock: bool = True,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topics = topics or {
            "sensor_data": "solix/sensors/{field_id}",
            "alerts": "solix/alerts/{field_id}",
            "predictions": "solix/predictions/{field_id}",
        }

        if use_mock:
            self.client = MockMQTTBroker()
        else:
            try:
                import paho.mqtt.client as mqtt
                self.client = mqtt.Client()
            except ImportError:
                logger.warning("paho-mqtt not installed, using mock broker")
                self.client = MockMQTTBroker()

    def connect(self) -> None:
        self.client.connect(self.broker_host, self.broker_port)

    def disconnect(self) -> None:
        self.client.disconnect()

    def publish_alert(self, field_id: str, alert: AlertMessage) -> None:
        topic = self.topics["alerts"].format(field_id=field_id)
        self.client.publish(topic, alert.to_json())

    def publish_prediction(self, field_id: str, prediction: PredictionMessage) -> None:
        topic = self.topics["predictions"].format(field_id=field_id)
        self.client.publish(topic, prediction.to_json())

    def subscribe_sensors(self, callback: Callable) -> None:
        topic = self.topics["sensor_data"].format(field_id="+")
        self.client.subscribe(topic, callback)

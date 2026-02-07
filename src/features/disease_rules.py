"""Epidemiological knowledge base: disease-specific thresholds and SEIR parameters."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DiseaseProfile:
    """Environmental thresholds and SEIR parameters for a specific disease."""

    name: str
    favorable_temp_range: tuple  # (min, max) Celsius
    favorable_humidity_min: float  # Minimum humidity %
    incubation_days: int
    beta_base: float  # Base transmission rate
    sigma: float  # Incubation rate (1/incubation_days)
    gamma: float  # Recovery rate
    severity_rate: float  # How quickly severity progresses
    description: str = ""


# Knowledge base of plant diseases and their environmental preferences
DISEASE_PROFILES: Dict[str, DiseaseProfile] = {
    "Tomato_Early_blight": DiseaseProfile(
        name="Early Blight (Alternaria solani)",
        favorable_temp_range=(24, 29),
        favorable_humidity_min=80,
        incubation_days=3,
        beta_base=0.35,
        sigma=0.33,
        gamma=0.05,
        severity_rate=0.08,
        description="Favored by warm, humid conditions with wet/dry cycles.",
    ),
    "Tomato_Late_blight": DiseaseProfile(
        name="Late Blight (Phytophthora infestans)",
        favorable_temp_range=(15, 22),
        favorable_humidity_min=90,
        incubation_days=4,
        beta_base=0.45,
        sigma=0.25,
        gamma=0.03,
        severity_rate=0.12,
        description="Thrives in cool, wet conditions. Very destructive.",
    ),
    "Tomato_Bacterial_spot": DiseaseProfile(
        name="Bacterial Spot (Xanthomonas)",
        favorable_temp_range=(25, 30),
        favorable_humidity_min=85,
        incubation_days=5,
        beta_base=0.30,
        sigma=0.20,
        gamma=0.04,
        severity_rate=0.06,
        description="Spread by rain splash and overhead irrigation.",
    ),
    "Tomato_Leaf_Mold": DiseaseProfile(
        name="Leaf Mold (Passalora fulva)",
        favorable_temp_range=(22, 26),
        favorable_humidity_min=85,
        incubation_days=7,
        beta_base=0.25,
        sigma=0.14,
        gamma=0.06,
        severity_rate=0.05,
        description="Common in greenhouses with high humidity.",
    ),
    "Tomato_Septoria_leaf_spot": DiseaseProfile(
        name="Septoria Leaf Spot",
        favorable_temp_range=(20, 25),
        favorable_humidity_min=75,
        incubation_days=5,
        beta_base=0.28,
        sigma=0.20,
        gamma=0.05,
        severity_rate=0.07,
        description="Spread by rain splash from lower infected leaves.",
    ),
    "Tomato_Spider_mites_Two_spotted_spider_mite": DiseaseProfile(
        name="Two-Spotted Spider Mite",
        favorable_temp_range=(27, 35),
        favorable_humidity_min=30,  # Actually thrives in LOW humidity
        incubation_days=7,
        beta_base=0.40,
        sigma=0.14,
        gamma=0.03,
        severity_rate=0.10,
        description="Thrives in hot, dry conditions. Rapid reproduction.",
    ),
    "Tomato__Target_Spot": DiseaseProfile(
        name="Target Spot (Corynespora cassiicola)",
        favorable_temp_range=(20, 30),
        favorable_humidity_min=80,
        incubation_days=4,
        beta_base=0.30,
        sigma=0.25,
        gamma=0.05,
        severity_rate=0.07,
        description="Favored by warm, humid conditions.",
    ),
    "Tomato__Tomato_mosaic_virus": DiseaseProfile(
        name="Tomato Mosaic Virus (ToMV)",
        favorable_temp_range=(18, 30),
        favorable_humidity_min=50,
        incubation_days=10,
        beta_base=0.20,
        sigma=0.10,
        gamma=0.02,
        severity_rate=0.04,
        description="Mechanical transmission. Very stable virus.",
    ),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": DiseaseProfile(
        name="Tomato Yellow Leaf Curl Virus (TYLCV)",
        favorable_temp_range=(25, 32),
        favorable_humidity_min=60,
        incubation_days=14,
        beta_base=0.35,
        sigma=0.07,
        gamma=0.02,
        severity_rate=0.06,
        description="Transmitted by whiteflies (Bemisia tabaci).",
    ),
    "Potato___Early_blight": DiseaseProfile(
        name="Potato Early Blight (Alternaria solani)",
        favorable_temp_range=(24, 29),
        favorable_humidity_min=80,
        incubation_days=3,
        beta_base=0.35,
        sigma=0.33,
        gamma=0.05,
        severity_rate=0.08,
        description="Similar to tomato early blight.",
    ),
    "Potato___Late_blight": DiseaseProfile(
        name="Potato Late Blight (Phytophthora infestans)",
        favorable_temp_range=(15, 22),
        favorable_humidity_min=90,
        incubation_days=4,
        beta_base=0.50,
        sigma=0.25,
        gamma=0.03,
        severity_rate=0.15,
        description="The disease that caused the Irish Potato Famine.",
    ),
    "Pepper__bell___Bacterial_spot": DiseaseProfile(
        name="Pepper Bacterial Spot",
        favorable_temp_range=(24, 30),
        favorable_humidity_min=85,
        incubation_days=5,
        beta_base=0.30,
        sigma=0.20,
        gamma=0.04,
        severity_rate=0.06,
        description="Bacterial disease spread by rain and contaminated tools.",
    ),
}

# Healthy classes (no disease)
HEALTHY_CLASSES = [
    "Tomato_healthy",
    "Potato___healthy",
    "Pepper__bell___healthy",
]


def is_condition_favorable(
    disease_name: str, temperature: float, humidity: float
) -> bool:
    """Check if environmental conditions favor a specific disease."""
    if disease_name not in DISEASE_PROFILES:
        return False
    profile = DISEASE_PROFILES[disease_name]
    temp_ok = profile.favorable_temp_range[0] <= temperature <= profile.favorable_temp_range[1]
    hum_ok = humidity >= profile.favorable_humidity_min
    return temp_ok and hum_ok


def get_risk_score(
    disease_name: str, temperature: float, humidity: float
) -> float:
    """Compute a 0-1 risk score based on how close conditions are to optimal."""
    if disease_name not in DISEASE_PROFILES:
        return 0.0
    profile = DISEASE_PROFILES[disease_name]

    temp_min, temp_max = profile.favorable_temp_range
    temp_center = (temp_min + temp_max) / 2
    temp_range = (temp_max - temp_min) / 2
    temp_score = max(0, 1 - abs(temperature - temp_center) / (temp_range * 2))

    hum_score = max(0, min(1, (humidity - profile.favorable_humidity_min + 20) / 40))

    return temp_score * hum_score


def get_all_disease_names() -> List[str]:
    """Return list of all disease names in the knowledge base."""
    return list(DISEASE_PROFILES.keys())

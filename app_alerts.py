"""Solinfitec Solix - Disease Alert Dashboard.

Interactive Streamlit dashboard for disease detection alerts.
Supports simulation mode (default) and real inference with Grad-CAM.

Run:
    streamlit run app_alerts.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.iot_simulator import IoTSimulator
from src.features.disease_rules import DISEASE_PROFILES, HEALTHY_CLASSES, get_risk_score
from src.utils.alert_system import Alert, AlertGenerator
from src.utils.config import ConfigManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
]

RISK_COLORS = {
    "LOW": "#2ecc71",
    "MEDIUM": "#f39c12",
    "HIGH": "#e67e22",
    "CRITICAL": "#e74c3c",
}

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

FUSION_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "best_fusion_model.pth"
SWIN_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "best_swin_classifier.pth"

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Solix - Disease Alerts",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def load_config() -> ConfigManager:
    return ConfigManager(str(PROJECT_ROOT / "configs" / "config.yaml"))


@st.cache_data
def simulate_fields(num_fields: int, seed: int) -> dict:
    """Run IoT simulator and return {field_id: DataFrame}."""
    config = load_config().get_raw("iot_simulation")
    config["num_fields"] = num_fields
    config["seed"] = seed
    simulator = IoTSimulator(config, seed=seed)
    return simulator.generate_all()


def generate_sim_predictions(
    field_df: pd.DataFrame, rng: np.random.Generator
) -> dict:
    """Generate synthetic predictions from IoT data for a single field."""
    last_row = field_df.iloc[-1]
    prevalence = last_row["disease_prevalence"]

    # Disease probabilities via Dirichlet weighted by prevalence
    alpha = np.ones(NUM_CLASSES) * 0.3
    # Boost a random disease class proportional to prevalence
    dominant_idx = rng.integers(0, NUM_CLASSES)
    alpha[dominant_idx] += prevalence * 20
    disease_probs = rng.dirichlet(alpha).tolist()

    # Outbreak risk from last 7 days of disease_prevalence, scaled
    tail = field_df.tail(7)["disease_prevalence"].values
    outbreak_risk_7d = np.clip(tail * (1 + rng.uniform(0.5, 3.0)), 0, 1).tolist()

    # Severity from prevalence
    if prevalence < 0.05:
        severity_probs = [0.7, 0.2, 0.08, 0.02]
    elif prevalence < 0.15:
        severity_probs = [0.2, 0.5, 0.2, 0.1]
    elif prevalence < 0.3:
        severity_probs = [0.05, 0.15, 0.55, 0.25]
    else:
        severity_probs = [0.02, 0.08, 0.25, 0.65]

    return {
        "disease_probs": disease_probs,
        "outbreak_risk_7d": outbreak_risk_7d,
        "severity_probs": severity_probs,
    }


def generate_field_coords(num_fields: int, seed: int) -> pd.DataFrame:
    """Generate synthetic lat/lon for fields (soy belt region of Brazil)."""
    rng = np.random.default_rng(seed)
    lats = rng.uniform(-24.0, -20.0, num_fields)
    lons = rng.uniform(-52.0, -48.0, num_fields)
    return pd.DataFrame(
        {
            "field_id": list(range(num_fields)),
            "lat": lats,
            "lon": lons,
        }
    )


def build_alert_generator() -> AlertGenerator:
    cfg = load_config()
    return AlertGenerator(
        risk_thresholds=cfg.alerts.risk_thresholds,
        severity_labels=cfg.alerts.severity_labels,
        actions=cfg.alerts.actions,
        class_names=CLASS_NAMES,
    )


def postprocess_severity(alert: Alert) -> Alert:
    """If disease is not healthy, force minimum severity to 'initial'.

    Fixes the issue where the severity head returns 'healthy' for diseased plants
    because it was trained on synthetic labels.
    """
    is_healthy = any(h in alert.disease_name for h in ("healthy",))
    if not is_healthy and alert.severity_stage == "healthy":
        alert.severity_stage = "initial"
        # Adjust confidence to reflect the override
        alert.metadata["severity_overridden"] = True
    return alert


# ---------------------------------------------------------------------------
# Real inference helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def load_fusion_model():
    """Load the MultiModalFusionModel from checkpoint."""
    from src.models.fusion_model import MultiModalFusionModel

    cfg = load_config()
    model = MultiModalFusionModel(
        num_classes=NUM_CLASSES,
        swin_model_name=cfg.model.variant,
        swin_checkpoint=str(SWIN_CHECKPOINT) if SWIN_CHECKPOINT.exists() else None,
        freeze_swin=True,
        feature_dim=cfg.model.feature_dim,
        visual_proj_dim=cfg.fusion.visual_proj_dim,
        temporal_config={
            "num_features": cfg.temporal_model.num_features,
            "d_model": cfg.temporal_model.d_model,
            "nhead": cfg.temporal_model.nhead,
            "num_layers": cfg.temporal_model.num_layers,
            "dim_feedforward": cfg.temporal_model.dim_feedforward,
            "dropout": cfg.temporal_model.dropout,
            "output_dim": cfg.temporal_model.output_dim,
            "sequence_length": cfg.temporal_model.sequence_length,
        },
        spatial_config={
            "input_dim": cfg.spatial_model.input_dim,
            "hidden_dim": cfg.spatial_model.hidden_dim,
            "output_dim": cfg.spatial_model.output_dim,
        },
        fusion_config={
            "fused_dim": cfg.fusion.fused_dim,
            "cross_attention": {
                "nhead": cfg.fusion.cross_attention.nhead,
                "dropout": cfg.fusion.cross_attention.dropout,
            },
        },
        forecast_days=7,
        num_severity_levels=4,
    )

    if FUSION_CHECKPOINT.exists():
        ckpt = torch.load(str(FUSION_CHECKPOINT), map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def run_real_inference(uploaded_file, field_df: pd.DataFrame, field_coords: pd.Series):
    """Run real model inference on an uploaded image."""
    from PIL import Image

    from src.features.augmentation import get_val_transforms

    model, device = load_fusion_model()

    # Preprocess image (get_val_transforms includes Normalize + ToTensorV2)
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    transform = get_val_transforms(img_size=224, resize_size=256)
    transformed = transform(image=img_np)
    img_tensor = transformed["image"].unsqueeze(0).float()

    # IoT sequence: last 30 days
    iot_cols = [
        "temperature", "humidity", "soil_moisture",
        "wind_speed", "rain_mm", "disease_prevalence", "gdd",
    ]
    seq_df = field_df.tail(30)[iot_cols]
    if len(seq_df) < 30:
        seq_df = pd.concat([seq_df] * (30 // len(seq_df) + 1)).tail(30)
    iot_tensor = torch.tensor(seq_df.values, dtype=torch.float32).unsqueeze(0)

    # Geo features
    geo = torch.tensor(
        [[field_coords["lat"], field_coords["lon"], 500.0]],
        dtype=torch.float32,
    )

    # Forward pass
    img_tensor = img_tensor.to(device)
    iot_tensor = iot_tensor.to(device)
    geo = geo.to(device)

    with torch.no_grad():
        outputs = model(img_tensor, iot_tensor, geo)

    disease_probs = torch.softmax(outputs["disease_logits"], dim=-1)[0].cpu().numpy()
    outbreak_risk = torch.sigmoid(outputs["outbreak_risk"])[0].cpu().numpy()
    severity_probs = torch.softmax(outputs["severity_logits"], dim=-1)[0].cpu().numpy()

    # Grad-CAM
    gradcam_overlay = None
    try:
        from src.visualization.gradcam import SwinGradCAM

        import cv2

        gradcam = SwinGradCAM(model.swin, target_layer="stage3")
        img_for_cam = img_tensor.to(device)
        img_for_cam.requires_grad_(True)
        heatmap = gradcam.generate(img_for_cam, target_class=int(disease_probs.argmax()))
        # Resize original image to match heatmap (224x224)
        img_resized = cv2.resize(img_np, (heatmap.shape[1], heatmap.shape[0]))
        gradcam_overlay = gradcam.visualize(img_resized, heatmap, alpha=0.4)
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")

    return {
        "disease_probs": disease_probs.tolist(),
        "outbreak_risk_7d": outbreak_risk.tolist(),
        "severity_probs": severity_probs.tolist(),
        "gradcam_overlay": gradcam_overlay,
        "original_image": img_np,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Solix Alert Dashboard")
st.sidebar.markdown("---")

num_fields = st.sidebar.slider("Number of fields", 5, 50, 20)
sim_seed = st.sidebar.number_input("Simulator seed", value=42, min_value=0, max_value=9999)

min_risk = st.sidebar.selectbox(
    "Minimum risk level",
    RISK_ORDER,
    index=0,
)

# Real inference toggle
checkpoint_exists = FUSION_CHECKPOINT.exists()
real_inference = False
uploaded_file = None
if checkpoint_exists:
    real_inference = st.sidebar.toggle("Real Inference", value=False)
    if real_inference:
        uploaded_file = st.sidebar.file_uploader(
            "Upload leaf image", type=["jpg", "jpeg", "png"]
        )
else:
    st.sidebar.info("No fusion checkpoint found. Simulation mode only.")

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Generate data and alerts
# ---------------------------------------------------------------------------

all_fields = simulate_fields(num_fields, sim_seed)
coords_df = generate_field_coords(num_fields, sim_seed)
alert_gen = build_alert_generator()

rng = np.random.default_rng(sim_seed)
alerts: list[Alert] = []
for fid in range(num_fields):
    preds = generate_sim_predictions(all_fields[fid], rng)
    alert = alert_gen.generate_alert(
        field_id=f"field_{fid:03d}",
        disease_probs=preds["disease_probs"],
        outbreak_risk_7d=preds["outbreak_risk_7d"],
        severity_probs=preds["severity_probs"],
    )
    alert = postprocess_severity(alert)
    alerts.append(alert)

# Filter by risk
filtered_alerts = alert_gen.filter_alerts(alerts, min_risk_level=min_risk)

# Build summary dataframe
alert_rows = []
for i, alert in enumerate(alerts):
    row = coords_df[coords_df["field_id"] == i].iloc[0]
    alert_rows.append(
        {
            "field_id": alert.field_id,
            "lat": row["lat"],
            "lon": row["lon"],
            "risk_level": alert.risk_level,
            "risk_score": alert.metadata.get("max_outbreak_risk", 0.0),
            "disease": alert.disease_name,
            "confidence": alert.disease_confidence,
            "severity": alert.severity_stage,
        }
    )
alerts_df = pd.DataFrame(alert_rows)

# ---------------------------------------------------------------------------
# Section 1: Summary Metrics
# ---------------------------------------------------------------------------

st.title("Disease Alert Dashboard")

col1, col2, col3, col4 = st.columns(4)

total_alerts = len(filtered_alerts)
critical_count = sum(1 for a in alerts if a.risk_level == "CRITICAL")
avg_risk = alerts_df["risk_score"].mean() if len(alerts_df) > 0 else 0.0
worst_field = alerts_df.loc[alerts_df["risk_score"].idxmax(), "field_id"] if len(alerts_df) > 0 else "N/A"

col1.metric("Filtered Alerts", total_alerts)
col2.metric("Critical Alerts", critical_count)
col3.metric("Avg Risk Score", f"{avg_risk:.3f}")
col4.metric("Highest Risk Field", worst_field)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2: Spatial Risk Map + Alert Summary
# ---------------------------------------------------------------------------

st.subheader("Spatial Risk Map & Alert Summary")
map_col, bar_col = st.columns([2, 1])

with map_col:
    fig_map = px.scatter(
        alerts_df,
        x="lon",
        y="lat",
        color="risk_level",
        size="risk_score",
        size_max=20,
        color_discrete_map=RISK_COLORS,
        category_orders={"risk_level": RISK_ORDER},
        hover_data=["field_id", "disease", "confidence", "severity", "risk_score"],
        labels={"lon": "Longitude", "lat": "Latitude", "risk_level": "Risk Level"},
        title="Field Risk Map",
    )
    fig_map.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_map, use_container_width=True)

with bar_col:
    risk_counts = alerts_df["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0)
    fig_bar = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map=RISK_COLORS,
        labels={"x": "Risk Level", "y": "Count"},
        title="Alert Count by Risk Level",
    )
    fig_bar.update_layout(
        height=480,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3: Selected Field Details
# ---------------------------------------------------------------------------

st.subheader("Field Details")

selected_field = st.selectbox(
    "Select field",
    list(range(num_fields)),
    format_func=lambda x: f"field_{x:03d}",
)

field_df = all_fields[selected_field]
field_alert = alerts[selected_field]

detail_left, detail_right = st.columns(2)

# Timeline 7-day risk
with detail_left:
    risk_7d = field_alert.outbreak_risk_7d
    days_labels = [f"Day {i+1}" for i in range(7)]

    # Color bars by risk zone
    bar_colors = []
    thresholds = load_config().alerts.risk_thresholds
    for r in risk_7d:
        if r >= thresholds["critical"]:
            bar_colors.append(RISK_COLORS["CRITICAL"])
        elif r >= thresholds["high"]:
            bar_colors.append(RISK_COLORS["HIGH"])
        elif r >= thresholds["medium"]:
            bar_colors.append(RISK_COLORS["MEDIUM"])
        else:
            bar_colors.append(RISK_COLORS["LOW"])

    fig_timeline = go.Figure(
        go.Bar(
            x=days_labels,
            y=risk_7d,
            marker_color=bar_colors,
            text=[f"{r:.3f}" for r in risk_7d],
            textposition="outside",
        )
    )
    fig_timeline.update_layout(
        title="7-Day Outbreak Risk Forecast",
        yaxis_title="Risk Score",
        yaxis_range=[0, max(1.0, max(risk_7d) * 1.2)],
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    # Add risk zone bands
    fig_timeline.add_hrect(y0=0, y1=thresholds["low"], fillcolor=RISK_COLORS["LOW"], opacity=0.08, line_width=0)
    fig_timeline.add_hrect(y0=thresholds["low"], y1=thresholds["medium"], fillcolor=RISK_COLORS["MEDIUM"], opacity=0.08, line_width=0)
    fig_timeline.add_hrect(y0=thresholds["medium"], y1=thresholds["high"], fillcolor=RISK_COLORS["HIGH"], opacity=0.08, line_width=0)
    fig_timeline.add_hrect(y0=thresholds["high"], y1=1.0, fillcolor=RISK_COLORS["CRITICAL"], opacity=0.08, line_width=0)
    st.plotly_chart(fig_timeline, use_container_width=True)

# Temporal progression (disease + temp + humidity over the year)
with detail_right:
    fig_prog = go.Figure()

    fig_prog.add_trace(
        go.Scatter(
            x=field_df["date"],
            y=field_df["disease_prevalence"],
            name="Disease Prevalence",
            line=dict(color="#e74c3c", width=2),
            yaxis="y",
        )
    )

    fig_prog.add_trace(
        go.Scatter(
            x=field_df["date"],
            y=field_df["temperature"],
            name="Temperature (C)",
            line=dict(color="#3498db", width=1, dash="dot"),
            yaxis="y2",
        )
    )

    fig_prog.add_trace(
        go.Scatter(
            x=field_df["date"],
            y=field_df["humidity"],
            name="Humidity (%)",
            line=dict(color="#27ae60", width=1, dash="dot"),
            yaxis="y2",
        )
    )

    fig_prog.update_layout(
        title="Temporal Progression",
        yaxis=dict(title="Disease Prevalence", side="left", range=[0, max(0.5, field_df["disease_prevalence"].max() * 1.3)]),
        yaxis2=dict(
            title="Temp / Humidity",
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_prog, use_container_width=True)

# IoT data table
with st.expander("IoT Sensor Data (last 14 days)"):
    display_df = field_df.tail(14).copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4: Alert Detail
# ---------------------------------------------------------------------------

st.subheader("Alert Detail")

# Handle real inference
real_result = None
if real_inference and uploaded_file is not None:
    field_coords_row = coords_df[coords_df["field_id"] == selected_field].iloc[0]
    with st.spinner("Running model inference..."):
        real_result = run_real_inference(uploaded_file, field_df, field_coords_row)

    # Re-generate alert with real predictions
    field_alert = alert_gen.generate_alert(
        field_id=f"field_{selected_field:03d}",
        disease_probs=real_result["disease_probs"],
        outbreak_risk_7d=real_result["outbreak_risk_7d"],
        severity_probs=real_result["severity_probs"],
    )
    field_alert = postprocess_severity(field_alert)

# Alert card
alert_col, info_col = st.columns([1, 1])

with alert_col:
    risk_color = RISK_COLORS.get(field_alert.risk_level, "#95a5a6")
    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {risk_color};
            padding: 16px;
            border-radius: 4px;
            background-color: rgba(0,0,0,0.02);
            margin-bottom: 12px;
        ">
            <h3 style="margin-top:0; color: {risk_color};">{field_alert.risk_level} - {field_alert.field_id}</h3>
            <p><b>Disease:</b> {field_alert.disease_name.replace("_", " ")}</p>
            <p><b>Confidence:</b> {field_alert.disease_confidence:.2%}</p>
            <p><b>Severity:</b> {field_alert.severity_stage} ({field_alert.severity_confidence:.2%})</p>
            <p><b>Max Outbreak Risk:</b> {field_alert.metadata.get("max_outbreak_risk", 0):.4f}
               (Day {field_alert.metadata.get("peak_risk_day", "?")})</p>
            <p><b>Action:</b> {field_alert.recommended_action}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Images for real inference
    if real_result is not None:
        img_left, img_right = st.columns(2)
        with img_left:
            st.image(real_result["original_image"], caption="Uploaded Image", use_container_width=True)
        with img_right:
            if real_result["gradcam_overlay"] is not None:
                st.image(real_result["gradcam_overlay"], caption="Grad-CAM Overlay", use_container_width=True)

# Disease profile
with info_col:
    disease_key = field_alert.disease_name
    if disease_key in DISEASE_PROFILES:
        profile = DISEASE_PROFILES[disease_key]
        st.markdown("**Epidemiological Profile**")
        st.markdown(f"**Name:** {profile.name}")
        st.markdown(f"**Description:** {profile.description}")
        st.markdown(
            f"**Favorable Temp:** {profile.favorable_temp_range[0]}-{profile.favorable_temp_range[1]} C"
        )
        st.markdown(f"**Min Humidity:** {profile.favorable_humidity_min}%")
        st.markdown(f"**Incubation:** {profile.incubation_days} days")
        st.markdown(f"**Severity Rate:** {profile.severity_rate}")

        # Current field risk score
        last_row = field_df.iloc[-1]
        current_risk = get_risk_score(
            disease_key, last_row["temperature"], last_row["humidity"]
        )
        st.metric("Current Environmental Risk", f"{current_risk:.3f}")
    elif disease_key in HEALTHY_CLASSES:
        st.success("Healthy - no disease profile applicable.")
    else:
        st.info(f"No epidemiological profile for '{disease_key}'.")

# JSON expander
with st.expander("Alert JSON"):
    st.json(field_alert.to_dict())

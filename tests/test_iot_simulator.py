"""Tests for IoT simulator and temporal features."""

import numpy as np
import pandas as pd
import pytest

from src.data.iot_simulator import IoTSimulator, SEIRModel
from src.features.temporal_features import TemporalFeatureEngineer


class TestSEIRModel:
    def test_initial_conditions(self):
        seir = SEIRModel()
        assert abs(seir.S + seir.E + seir.I + seir.R - 1.0) < 1e-6

    def test_step_preserves_total(self):
        seir = SEIRModel()
        for _ in range(100):
            seir.step(25.0, 80.0)
        total = seir.S + seir.E + seir.I + seir.R
        assert abs(total - 1.0) < 1e-6

    def test_high_risk_conditions_increase_infection(self):
        seir = SEIRModel(beta_base=0.5)
        initial_I = seir.I
        for _ in range(30):
            seir.step(25.0, 85.0)  # Optimal conditions
        assert seir.I > initial_I or seir.R > 0.0

    def test_low_risk_conditions(self):
        seir = SEIRModel(beta_base=0.3)
        for _ in range(30):
            seir.step(5.0, 30.0)  # Very unfavorable conditions
        # Infection should be minimal due to low beta
        assert seir.I < 0.5


class TestIoTSimulator:
    @pytest.fixture
    def simulator(self):
        config = {
            "num_fields": 3,
            "days": 60,
            "seed": 42,
            "temperature": {"base_mean": 25, "base_amplitude": 8,
                            "diurnal_amplitude": 5, "ar1_coeff": 0.7, "noise_std": 1.5},
            "humidity": {"base_mean": 70, "anti_corr_factor": -1.5, "noise_std": 5},
            "rain": {"zero_inflation": 0.7, "gamma_shape": 0.8, "gamma_scale": 5.0},
            "soil_moisture": {"initial": 0.35, "rain_factor": 0.05, "decay_rate": 0.03},
            "wind": {"log_mean": 1.5, "log_std": 0.6},
            "disease_seir": {
                "S0": 0.95, "E0": 0.03, "I0": 0.02, "R0": 0.0,
                "beta_base": 0.3, "sigma": 0.1, "gamma": 0.05,
                "temp_optimal": 25.0, "humidity_optimal": 85.0,
            },
        }
        return IoTSimulator(config, seed=42)

    def test_generate_field_columns(self, simulator):
        df = simulator.generate_field(0)
        expected_cols = [
            "date", "field_id", "temperature", "humidity",
            "soil_moisture", "wind_speed", "rain_mm",
            "disease_prevalence", "gdd",
        ]
        for col in expected_cols:
            assert col in df.columns

    def test_generate_field_length(self, simulator):
        df = simulator.generate_field(0)
        assert len(df) == 60

    def test_value_ranges(self, simulator):
        df = simulator.generate_field(0)
        assert df["humidity"].min() >= 20
        assert df["humidity"].max() <= 100
        assert df["soil_moisture"].min() >= 0
        assert df["soil_moisture"].max() <= 1
        assert df["rain_mm"].min() >= 0
        assert df["disease_prevalence"].min() >= 0
        assert df["disease_prevalence"].max() <= 1

    def test_generate_all(self, simulator, tmp_path):
        all_data = simulator.generate_all(output_dir=str(tmp_path))
        assert len(all_data) == 3
        # Check parquet files created
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 3

    def test_different_fields_have_different_data(self, simulator):
        df0 = simulator.generate_field(0)
        df1 = simulator.generate_field(1)
        assert not np.allclose(df0["temperature"].values, df1["temperature"].values)


class TestTemporalFeatureEngineer:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=60),
            "temperature": 25 + np.random.randn(60) * 3,
            "humidity": 70 + np.random.randn(60) * 10,
            "soil_moisture": 0.3 + np.random.randn(60) * 0.05,
            "wind_speed": np.abs(np.random.randn(60) * 3),
            "rain_mm": np.maximum(np.random.randn(60) * 2, 0),
            "disease_prevalence": np.random.rand(60) * 0.1,
        })

    def test_rolling_stats_adds_columns(self, sample_df):
        eng = TemporalFeatureEngineer()
        result = eng.compute_rolling_stats(sample_df)
        assert "temperature_mean_7d" in result.columns
        assert "humidity_std_14d" in result.columns
        assert len(result) == len(sample_df)

    def test_lag_features(self, sample_df):
        eng = TemporalFeatureEngineer()
        result = eng.compute_lag_features(sample_df)
        assert "temperature_lag_1d" in result.columns
        assert "humidity_lag_7d" in result.columns

    def test_binary_indicators(self, sample_df):
        eng = TemporalFeatureEngineer()
        result = eng.compute_binary_indicators(sample_df)
        assert "favorable_fungal" in result.columns
        assert result["favorable_fungal"].dtype == int

    def test_transform_complete(self, sample_df):
        eng = TemporalFeatureEngineer()
        result = eng.transform(sample_df)
        assert len(result.columns) > len(sample_df.columns)
        assert len(result) == len(sample_df)

"""IoT sensor data simulator with realistic environmental time series.

Generates multivariate time series:
    - Temperature: seasonal base + diurnal cycle + AR(1) noise
    - Humidity: anti-correlated with temperature
    - Soil moisture: rain-driven with exponential decay
    - Wind: log-normal seasonal
    - Rain: zero-inflated gamma
    - Disease prevalence: SEIR model driven by temperature and humidity
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("solinfitec.iot_simulator")


class SEIRModel:
    """Susceptible-Exposed-Infectious-Recovered epidemiological model.

    Transmission rate beta varies with temperature and humidity.
    """

    def __init__(
        self,
        S0: float = 0.95,
        E0: float = 0.03,
        I0: float = 0.02,
        R0: float = 0.0,
        beta_base: float = 0.3,
        sigma: float = 0.1,
        gamma: float = 0.05,
        temp_optimal: float = 25.0,
        humidity_optimal: float = 85.0,
    ):
        self.S = S0
        self.E = E0
        self.I = I0
        self.R = R0
        self.beta_base = beta_base
        self.sigma = sigma
        self.gamma = gamma
        self.temp_optimal = temp_optimal
        self.humidity_optimal = humidity_optimal

    def compute_beta(self, temperature: float, humidity: float) -> float:
        """Compute transmission rate modulated by environmental conditions."""
        temp_factor = np.exp(-0.5 * ((temperature - self.temp_optimal) / 5.0) ** 2)
        hum_factor = np.exp(-0.5 * ((humidity - self.humidity_optimal) / 15.0) ** 2)
        return self.beta_base * temp_factor * hum_factor

    def step(self, temperature: float, humidity: float, dt: float = 1.0):
        """Advance SEIR by one time step (1 day)."""
        beta = self.compute_beta(temperature, humidity)

        dS = -beta * self.S * self.I * dt
        dE = (beta * self.S * self.I - self.sigma * self.E) * dt
        dI = (self.sigma * self.E - self.gamma * self.I) * dt
        dR = self.gamma * self.I * dt

        self.S = np.clip(self.S + dS, 0, 1)
        self.E = np.clip(self.E + dE, 0, 1)
        self.I = np.clip(self.I + dI, 0, 1)
        self.R = np.clip(self.R + dR, 0, 1)

        # Renormalize
        total = self.S + self.E + self.I + self.R
        if total > 0:
            self.S /= total
            self.E /= total
            self.I /= total
            self.R /= total

        return self.I


class IoTSimulator:
    """Generates realistic IoT sensor time series for agricultural fields.

    Args:
        config: Dictionary with simulation parameters (from config.yaml iot_simulation).
        seed: Random seed.
    """

    def __init__(self, config: Dict, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.num_fields = config.get("num_fields", 50)
        self.days = config.get("days", 365)

    def calibrate_from_real_data(self, csv_path: str) -> Dict[str, Dict]:
        """Calibrate simulator parameters from real sensor data (e.g. Grape Disease Dataset).

        Reads a CSV with columns: Temperature, Humidity, LW (leaf wetness).
        Computes statistics and updates internal config for more realistic simulation.

        Args:
            csv_path: Path to CSV file with real sensor readings.

        Returns:
            Dict with computed statistics for each variable.
        """
        df = pd.read_csv(csv_path)
        # Normalize column names (strip whitespace)
        df.columns = df.columns.str.strip()

        stats = {}

        if "Temperature" in df.columns:
            temp = pd.to_numeric(df["Temperature"], errors="coerce").dropna()
            stats["temperature"] = {
                "mean": float(temp.mean()),
                "std": float(temp.std()),
                "min": float(temp.min()),
                "max": float(temp.max()),
            }
            # Update config
            self.config.setdefault("temperature", {})
            self.config["temperature"]["base_mean"] = round(float(temp.mean()), 2)
            self.config["temperature"]["noise_std"] = round(float(temp.std() * 0.5), 2)
            logger.info(
                f"Calibrated temperature: mean={temp.mean():.2f}, std={temp.std():.2f}"
            )

        if "Humidity" in df.columns:
            hum = pd.to_numeric(df["Humidity"], errors="coerce").dropna()
            stats["humidity"] = {
                "mean": float(hum.mean()),
                "std": float(hum.std()),
                "min": float(hum.min()),
                "max": float(hum.max()),
            }
            self.config.setdefault("humidity", {})
            self.config["humidity"]["base_mean"] = round(float(hum.mean()), 2)
            self.config["humidity"]["noise_std"] = round(float(hum.std() * 0.5), 2)
            logger.info(
                f"Calibrated humidity: mean={hum.mean():.2f}, std={hum.std():.2f}"
            )

        if "LW" in df.columns:
            lw = pd.to_numeric(df["LW"], errors="coerce").dropna()
            stats["leaf_wetness"] = {
                "mean": float(lw.mean()),
                "std": float(lw.std()),
                "min": float(lw.min()),
                "max": float(lw.max()),
            }
            logger.info(
                f"Leaf wetness stats: mean={lw.mean():.2f}, std={lw.std():.2f}"
            )

        # Compute cross-correlation between temperature and humidity
        if "Temperature" in df.columns and "Humidity" in df.columns:
            temp = pd.to_numeric(df["Temperature"], errors="coerce")
            hum = pd.to_numeric(df["Humidity"], errors="coerce")
            valid = temp.notna() & hum.notna()
            if valid.sum() > 10:
                corr = temp[valid].corr(hum[valid])
                stats["temp_humidity_correlation"] = float(corr)
                # Update anti-correlation factor based on real data
                if corr < 0:
                    self.config["humidity"]["anti_corr_factor"] = round(float(corr * hum[valid].std() / temp[valid].std()), 2)
                logger.info(f"Temp-Humidity correlation: {corr:.3f}")

        logger.info(f"Calibration complete from {csv_path}")
        return stats

    def _generate_temperature(self) -> np.ndarray:
        """Seasonal base + diurnal cycle + AR(1) noise."""
        cfg = self.config.get("temperature", {})
        base_mean = cfg.get("base_mean", 25.0)
        base_amp = cfg.get("base_amplitude", 8.0)
        diurnal_amp = cfg.get("diurnal_amplitude", 5.0)
        ar1 = cfg.get("ar1_coeff", 0.7)
        noise_std = cfg.get("noise_std", 1.5)

        t = np.arange(self.days)
        seasonal = base_mean + base_amp * np.sin(2 * np.pi * t / 365 - np.pi / 2)
        diurnal = diurnal_amp * np.sin(2 * np.pi * t / 1 + np.pi / 4)

        # AR(1) noise
        noise = np.zeros(self.days)
        for i in range(1, self.days):
            noise[i] = ar1 * noise[i - 1] + self.rng.normal(0, noise_std)

        return seasonal + diurnal * 0.3 + noise

    def _generate_humidity(self, temperature: np.ndarray) -> np.ndarray:
        """Anti-correlated with temperature."""
        cfg = self.config.get("humidity", {})
        base_mean = cfg.get("base_mean", 70.0)
        anti_corr = cfg.get("anti_corr_factor", -1.5)
        noise_std = cfg.get("noise_std", 5.0)

        temp_deviation = temperature - temperature.mean()
        humidity = base_mean + anti_corr * temp_deviation + self.rng.normal(0, noise_std, self.days)
        return np.clip(humidity, 20, 100)

    def _generate_rain(self) -> np.ndarray:
        """Zero-inflated gamma distribution."""
        cfg = self.config.get("rain", {})
        zero_inflation = cfg.get("zero_inflation", 0.7)
        shape = cfg.get("gamma_shape", 0.8)
        scale = cfg.get("gamma_scale", 5.0)

        rain = np.zeros(self.days)
        rain_mask = self.rng.random(self.days) > zero_inflation
        rain[rain_mask] = self.rng.gamma(shape, scale, size=rain_mask.sum())
        return rain

    def _generate_soil_moisture(self, rain: np.ndarray) -> np.ndarray:
        """Rain-driven with exponential decay."""
        cfg = self.config.get("soil_moisture", {})
        initial = cfg.get("initial", 0.35)
        rain_factor = cfg.get("rain_factor", 0.05)
        decay_rate = cfg.get("decay_rate", 0.03)

        soil = np.zeros(self.days)
        soil[0] = initial
        for i in range(1, self.days):
            soil[i] = soil[i - 1] - decay_rate * soil[i - 1] + rain_factor * rain[i]
        return np.clip(soil, 0.05, 0.60)

    def _generate_wind(self) -> np.ndarray:
        """Log-normal seasonal wind speed."""
        cfg = self.config.get("wind", {})
        log_mean = cfg.get("log_mean", 1.5)
        log_std = cfg.get("log_std", 0.6)

        t = np.arange(self.days)
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * t / 365)
        wind = self.rng.lognormal(log_mean, log_std, self.days) * seasonal_factor
        return np.clip(wind, 0, 50)

    def _compute_gdd(self, temperature: np.ndarray, base_temp: float = 10.0) -> np.ndarray:
        """Accumulated growing degree days."""
        daily_gdd = np.maximum(temperature - base_temp, 0)
        return np.cumsum(daily_gdd)

    def generate_field(self, field_id: int) -> pd.DataFrame:
        """Generate complete time series for a single field."""
        # Reset RNG with field-specific seed
        self.rng = np.random.default_rng(self.config.get("seed", 42) + field_id)

        temperature = self._generate_temperature()
        humidity = self._generate_humidity(temperature)
        rain = self._generate_rain()
        soil_moisture = self._generate_soil_moisture(rain)
        wind = self._generate_wind()
        gdd = self._compute_gdd(temperature)

        # SEIR disease model
        seir_cfg = self.config.get("disease_seir", {})
        seir = SEIRModel(**seir_cfg)
        disease_prevalence = np.zeros(self.days)
        for i in range(self.days):
            disease_prevalence[i] = seir.step(temperature[i], humidity[i])

        dates = pd.date_range(start="2024-01-01", periods=self.days, freq="D")

        df = pd.DataFrame({
            "date": dates,
            "field_id": field_id,
            "temperature": np.round(temperature, 2),
            "humidity": np.round(humidity, 2),
            "soil_moisture": np.round(soil_moisture, 4),
            "wind_speed": np.round(wind, 2),
            "rain_mm": np.round(rain, 2),
            "disease_prevalence": np.round(disease_prevalence, 6),
            "gdd": np.round(gdd, 1),
        })
        return df

    def generate_all(self, output_dir: Optional[str] = None) -> Dict[int, pd.DataFrame]:
        """Generate data for all fields, optionally saving to parquet."""
        all_data = {}
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

        for field_id in range(self.num_fields):
            df = self.generate_field(field_id)
            all_data[field_id] = df

            if output_dir:
                df.to_parquet(out_path / f"field_{field_id:03d}.parquet", index=False)

        logger.info(
            f"Generated IoT data for {self.num_fields} fields, {self.days} days each"
        )
        return all_data

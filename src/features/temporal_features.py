"""Temporal feature engineering from IoT time series data."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("solinfitec.temporal_features")


class TemporalFeatureEngineer:
    """Computes temporal features from IoT sensor data.

    Features:
        - Rolling means (7d, 14d, 30d)
        - Lag features
        - Binary indicators (humidity>80 AND temp 20-30)
        - Accumulated growing degree days
    """

    def __init__(self, windows: List[int] = None):
        self.windows = windows or [7, 14, 30]

    def compute_rolling_stats(
        self, df: pd.DataFrame, columns: List[str] = None
    ) -> pd.DataFrame:
        """Compute rolling mean and std for specified columns."""
        if columns is None:
            columns = ["temperature", "humidity", "soil_moisture", "wind_speed", "rain_mm"]

        result = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            for window in self.windows:
                result[f"{col}_mean_{window}d"] = (
                    df[col].rolling(window, min_periods=1).mean()
                )
                result[f"{col}_std_{window}d"] = (
                    df[col].rolling(window, min_periods=1).std().fillna(0)
                )
        return result

    def compute_lag_features(
        self, df: pd.DataFrame, columns: List[str] = None, lags: List[int] = None
    ) -> pd.DataFrame:
        """Add lag features."""
        if columns is None:
            columns = ["temperature", "humidity", "disease_prevalence"]
        if lags is None:
            lags = [1, 3, 7]

        result = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            for lag in lags:
                result[f"{col}_lag_{lag}d"] = df[col].shift(lag).bfill()
        return result

    def compute_binary_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute disease-favorable condition indicators."""
        result = df.copy()

        # High humidity + moderate temperature = favorable for fungal diseases
        if "humidity" in df.columns and "temperature" in df.columns:
            result["favorable_fungal"] = (
                (df["humidity"] > 80) & (df["temperature"] >= 20) & (df["temperature"] <= 30)
            ).astype(int)

        # High soil moisture
        if "soil_moisture" in df.columns:
            result["high_soil_moisture"] = (df["soil_moisture"] > 0.40).astype(int)

        # Rain streak (3+ consecutive rain days)
        if "rain_mm" in df.columns:
            rain_binary = (df["rain_mm"] > 0).astype(int)
            result["rain_streak_3d"] = (
                rain_binary.rolling(3, min_periods=3).sum() == 3
            ).astype(int).fillna(0)

        # Temperature stress (>35 or <10)
        if "temperature" in df.columns:
            result["temp_stress"] = (
                (df["temperature"] > 35) | (df["temperature"] < 10)
            ).astype(int)

        return result

    def compute_gdd(
        self, df: pd.DataFrame, base_temp: float = 10.0
    ) -> pd.DataFrame:
        """Compute accumulated growing degree days."""
        result = df.copy()
        if "temperature" in df.columns:
            daily_gdd = np.maximum(df["temperature"].values - base_temp, 0)
            result["gdd_accumulated"] = np.cumsum(daily_gdd)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all temporal feature transformations."""
        result = self.compute_rolling_stats(df)
        result = self.compute_lag_features(result)
        result = self.compute_binary_indicators(result)
        result = self.compute_gdd(result)
        return result

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature column names (excluding date and ID columns)."""
        exclude = {"date", "field_id"}
        return [col for col in df.columns if col not in exclude]

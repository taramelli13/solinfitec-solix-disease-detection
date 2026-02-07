"""Weather data client using Open-Meteo API with local caching."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("solinfitec.weather_client")


class WeatherClient:
    """Fetches weather data from Open-Meteo API with cache and simulation fallback.

    Args:
        cache_dir: Directory for cached API responses.
        fallback_to_simulation: If True, generate simulated data on API failure.
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        cache_dir: str = "data/external/weather_cache",
        fallback_to_simulation: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fallback = fallback_to_simulation

    def _cache_key(self, lat: float, lon: float, start: str, end: str) -> str:
        return f"weather_{lat}_{lon}_{start}_{end}.json"

    def _load_cache(self, key: str) -> Optional[Dict]:
        cache_file = self.cache_dir / key
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, data: Dict) -> None:
        cache_file = self.cache_dir / key
        with open(cache_file, "w") as f:
            json.dump(data, f)

    def fetch(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetch weather data for a location and date range.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            variables: Weather variables to fetch.

        Returns:
            DataFrame with daily weather data.
        """
        if variables is None:
            variables = [
                "temperature_2m_max",
                "temperature_2m_min",
                "relative_humidity_2m_max",
                "precipitation_sum",
                "windspeed_10m_max",
            ]

        cache_key = self._cache_key(latitude, longitude, start_date, end_date)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info(f"Using cached weather data: {cache_key}")
            return pd.DataFrame(cached)

        try:
            import requests

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": ",".join(variables),
                "timezone": "auto",
            }
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            df = pd.DataFrame(daily)

            self._save_cache(cache_key, df.to_dict(orient="list"))
            logger.info(f"Fetched weather data: {len(df)} days")
            return df

        except Exception as e:
            logger.warning(f"Weather API failed: {e}")
            if self.fallback:
                return self._generate_fallback(start_date, end_date)
            raise

    def _generate_fallback(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate simulated weather data as fallback."""
        import numpy as np

        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        n = len(dates)
        rng = np.random.default_rng(42)

        t = np.arange(n)
        temp_max = 30 + 5 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 2, n)
        temp_min = temp_max - 8 - rng.uniform(0, 3, n)

        df = pd.DataFrame({
            "time": dates.strftime("%Y-%m-%d").tolist(),
            "temperature_2m_max": np.round(temp_max, 1),
            "temperature_2m_min": np.round(temp_min, 1),
            "relative_humidity_2m_max": np.round(
                np.clip(70 + rng.normal(0, 10, n), 30, 100), 1
            ),
            "precipitation_sum": np.round(
                np.where(rng.random(n) > 0.7, rng.gamma(0.8, 5, n), 0), 1
            ),
            "windspeed_10m_max": np.round(rng.lognormal(1.5, 0.5, n), 1),
        })

        logger.info(f"Generated fallback weather data: {n} days")
        return df

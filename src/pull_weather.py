"""Pull hourly temperature for MISO load centers from Open-Meteo and average them.

Two pulls: observed temperature (archive) and the forecast that Open-Meteo's
models made 48 hours ahead of each hour (previous runs).

Run from the repo root: python -m src.pull_weather
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.fetch import get_json

URL = "https://archive-api.open-meteo.com/v1/archive"
START = "2023-01-01"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "weather_miso.parquet"

FORECAST_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
# value as forecast 48 hours before the hour
FORECAST_VARIABLE = "temperature_2m_previous_day2"
# full coverage for these cities from here on
FORECAST_START = "2025-01-01"
FORECAST_OUT_PATH = OUT_PATH.parent / "weather_forecast_miso.parquet"

CITIES = {
    "detroit": (42.33, -83.05),
    "chicago": (41.88, -87.63),
    "minneapolis": (44.98, -93.27),
    "st_louis": (38.63, -90.20),
}


def fetch_city(lat: float, lon: float, start_date: str, end_date: str,
               url: str = URL, variable: str = "temperature_2m") -> pd.Series:
    """Hourly temperature in Fahrenheit, indexed by UTC hour."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": variable,
        "temperature_unit": "fahrenheit",
        "timezone": "GMT",
    }
    hourly = get_json(url, params)["hourly"]
    index = pd.to_datetime(hourly["time"], utc=True)
    return pd.Series(hourly[variable], index=index, dtype="float64")


def add_city_mean(temps: pd.DataFrame) -> pd.DataFrame:
    """Unweighted mean across cities; missing if any city is missing."""
    result = temps.copy()
    result["temp_f_mean"] = temps.mean(axis=1, skipna=False)
    return result


def pull(url: str, variable: str, start: str, out_path: Path) -> pd.DataFrame:
    now_hour = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
    end = now_hour.strftime("%Y-%m-%d")

    columns = {}
    for city, (lat, lon) in CITIES.items():
        print(f"Open-Meteo {variable}: {city}")
        columns[f"temp_f_{city}"] = fetch_city(lat, lon, start, end, url=url, variable=variable)

    temps = add_city_mean(pd.DataFrame(columns))
    # both endpoints fill hours after now with forecasts, drop them
    temps = temps[temps.index <= now_hour]
    temps.index.name = "timestamp_utc"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temps.to_parquet(out_path)
    print(f"Wrote {len(temps):,} hours ({temps.index.min()} to {temps.index.max()}) to {out_path}")
    return temps


def main() -> None:
    pull(URL, "temperature_2m", START, OUT_PATH)
    pull(FORECAST_URL, FORECAST_VARIABLE, FORECAST_START, FORECAST_OUT_PATH)


if __name__ == "__main__":
    main()

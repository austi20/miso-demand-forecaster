"""Pull hourly temperature for MISO load centers from Open-Meteo and average them.

Run from the repo root: python -m src.pull_weather
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.fetch import get_json

URL = "https://archive-api.open-meteo.com/v1/archive"
START = "2023-01-01"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "weather_miso.parquet"

CITIES = {
    "detroit": (42.33, -83.05),
    "chicago": (41.88, -87.63),
    "minneapolis": (44.98, -93.27),
    "st_louis": (38.63, -90.20),
}


def fetch_city(lat: float, lon: float, start_date: str, end_date: str) -> pd.Series:
    """Hourly 2m temperature in Fahrenheit, indexed by UTC hour."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "GMT",
    }
    hourly = get_json(URL, params)["hourly"]
    index = pd.to_datetime(hourly["time"], utc=True)
    return pd.Series(hourly["temperature_2m"], index=index, dtype="float64")


def add_city_mean(temps: pd.DataFrame) -> pd.DataFrame:
    """Unweighted mean across cities; missing if any city is missing."""
    result = temps.copy()
    result["temp_f_mean"] = temps.mean(axis=1, skipna=False)
    return result


def main(start: str = START, end: str | None = None) -> pd.DataFrame:
    now_hour = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
    if end is None:
        end = now_hour.strftime("%Y-%m-%d")

    columns = {}
    for city, (lat, lon) in CITIES.items():
        print(f"Open-Meteo: {city}")
        columns[f"temp_f_{city}"] = fetch_city(lat, lon, start, end)

    temps = add_city_mean(pd.DataFrame(columns))
    # archive fills hours after now with model forecast, drop them
    temps = temps[temps.index <= now_hour]
    temps.index.name = "timestamp_utc"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temps.to_parquet(OUT_PATH)
    print(f"Wrote {len(temps):,} hours ({temps.index.min()} to {temps.index.max()}) to {OUT_PATH}")
    return temps


if __name__ == "__main__":
    main()

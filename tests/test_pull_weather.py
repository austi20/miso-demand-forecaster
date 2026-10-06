"""Tests for Open-Meteo parsing and the city average."""

import math

import pandas as pd

from src import pull_weather


def test_fetch_city_returns_utc_series(monkeypatch):
    def fake_get_json(url, params):
        assert params["timezone"] == "GMT"
        return {"hourly": {"time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                           "temperature_2m": [30.5, None]}}

    monkeypatch.setattr(pull_weather, "get_json", fake_get_json)
    temps = pull_weather.fetch_city(42.3, -83.0, "2024-01-01", "2024-01-01")

    assert temps.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert temps.iloc[0] == 30.5
    assert math.isnan(temps.iloc[1])


def test_add_city_mean_leaves_gap_when_any_city_missing():
    index = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    temps = pd.DataFrame({"temp_f_a": [10.0, 20.0], "temp_f_b": [30.0, None]}, index=index)
    result = pull_weather.add_city_mean(temps)

    assert result["temp_f_mean"].iloc[0] == 20.0
    assert math.isnan(result["temp_f_mean"].iloc[1])


def test_fetch_city_reads_requested_variable(monkeypatch):
    def fake_get_json(url, params):
        assert url == pull_weather.FORECAST_URL
        assert params["hourly"] == "temperature_2m_previous_day2"
        return {"hourly": {"time": ["2025-01-01T00:00"], "temperature_2m_previous_day2": [12.5]}}

    monkeypatch.setattr(pull_weather, "get_json", fake_get_json)
    temps = pull_weather.fetch_city(42.3, -83.0, "2025-01-01", "2025-01-01",
                                    url=pull_weather.FORECAST_URL,
                                    variable="temperature_2m_previous_day2")
    assert temps.iloc[0] == 12.5

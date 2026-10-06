"""Tests for the feature table."""

import math

import pandas as pd
import pytest

from src import features


def hourly_eia(start, hours):
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    demand = [float(i) for i in range(hours)]
    return pd.DataFrame({"demand_mwh": demand, "forecast_mwh": demand}, index=index)


def flat_temps(eia):
    return pd.Series(50.0, index=eia.index)


def test_lags_are_48_and_168_hours():
    eia = hourly_eia("2024-01-01", 200)
    frame = features.build_features(eia, flat_temps(eia))

    assert math.isnan(frame["demand_lag48"].iloc[47])
    assert frame["demand_lag48"].iloc[48] == 0
    assert frame["demand_lag168"].iloc[199] == 199 - 168


def test_calendar_uses_central_time():
    # 05:00 UTC on July 4 2024 is midnight CDT
    eia = hourly_eia("2024-07-04 04:00", 2)
    frame = features.build_features(eia, flat_temps(eia))

    assert list(frame["hour"]) == [23, 0]
    assert list(frame["day_of_week"]) == [2, 3]
    assert list(frame["is_holiday"]) == [0, 1]


def holiday_flag(utc_hour):
    eia = hourly_eia(utc_hour, 1)
    return features.build_features(eia, flat_temps(eia))["is_holiday"].iloc[0]


def test_holiday_flag_covers_thanksgiving():
    assert holiday_flag("2025-11-27 17:00") == 1


def test_saturday_holiday_stays_on_saturday():
    # NERC moves Sunday holidays to Monday, never Saturday to Friday
    assert holiday_flag("2026-07-04 17:00") == 1
    assert holiday_flag("2026-07-03 17:00") == 0


def test_sunday_holiday_moves_to_monday():
    # Christmas 2022 fell on a Sunday
    assert holiday_flag("2022-12-26 17:00") == 1


def test_temperature_aligned_by_timestamp():
    eia = hourly_eia("2024-01-01", 3)
    temps = pd.Series([10.0, 20.0], index=eia.index[1:])
    frame = features.build_features(eia, temps)

    assert math.isnan(frame["temp_f"].iloc[0])
    assert list(frame["temp_f"].iloc[1:]) == [10.0, 20.0]


def test_rejects_gaps_in_hourly_grid():
    eia = hourly_eia("2024-01-01", 5).drop(pd.Timestamp("2024-01-01 02:00", tz="UTC"))
    with pytest.raises(ValueError):
        features.build_features(eia, flat_temps(eia))

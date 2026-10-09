"""Tests for the nightly forecast, scoring and drift check."""

import numpy as np
import pandas as pd
import pytest

from src import nightly
from src.features import CENTRAL, FEATURES, TARGET


def test_day_hours_cover_one_central_day_in_utc():
    hours = nightly.day_hours(pd.Timestamp("2026-10-09"))
    assert len(hours) == 24
    assert hours[0] == pd.Timestamp("2026-10-09 05:00", tz="UTC")


def test_day_hours_handle_the_daylight_saving_change():
    assert len(nightly.day_hours(pd.Timestamp("2026-11-01"))) == 25
    assert len(nightly.day_hours(pd.Timestamp("2026-03-08"))) == 23


def test_extend_to_adds_empty_hours_through_the_target_day():
    index = pd.date_range("2026-10-01", periods=48, freq="h", tz="UTC")
    eia = pd.DataFrame(1.0, index=index, columns=["demand_mwh", "forecast_mwh"])
    longer = nightly.extend_to(eia, pd.Timestamp("2026-10-05"))

    assert longer.index.max() == pd.Timestamp("2026-10-06 04:00", tz="UTC")
    assert longer["demand_mwh"].iloc[-1] != longer["demand_mwh"].iloc[-1]


def make_day(day, actual, lightgbm, eia):
    hours = nightly.day_hours(pd.Timestamp(day))
    n = len(hours)
    forecast = pd.DataFrame({"forecast_mwh": lightgbm}, index=hours)
    eia_frame = pd.DataFrame({"demand_mwh": actual, "forecast_mwh": eia}, index=hours)
    eia_frame["demand_lag168"] = 100.0
    return forecast, eia_frame, n


def test_score_day_compares_all_three_methods():
    forecast, eia, _ = make_day("2026-10-07", 100.0, 110.0, 95.0)
    row = nightly.score_day(forecast, eia)

    assert row["lightgbm_mape"] == pytest.approx(10.0)
    assert row["eia_mape"] == pytest.approx(5.0)
    assert row["naive_mape"] == pytest.approx(0.0)
    assert row["n_hours"] == 24


def test_score_day_waits_for_a_complete_day():
    forecast, eia, _ = make_day("2026-10-07", 100.0, 110.0, 95.0)
    eia.iloc[-1, eia.columns.get_loc("demand_mwh")] = np.nan
    assert nightly.score_day(forecast, eia) is None


def test_score_day_drops_hours_where_eia_has_no_forecast():
    forecast, eia, _ = make_day("2026-10-07", 100.0, 110.0, 95.0)
    eia.iloc[:4, eia.columns.get_loc("forecast_mwh")] = np.nan
    assert nightly.score_day(forecast, eia)["n_hours"] == 20


def scores_with(mapes):
    days = pd.date_range("2026-09-01", periods=len(mapes)).strftime("%Y-%m-%d")
    return pd.DataFrame({"day": days, "lightgbm_mape": mapes, "n_hours": 24})


def test_drift_waits_for_enough_days():
    state, rolling, n_days = nightly.drift_status(scores_with([3.0] * 5), baseline=3.0)
    assert state == "collecting"
    assert n_days == 5


def test_drift_warns_above_fifty_percent_over_baseline():
    state, rolling, _ = nightly.drift_status(scores_with([5.0] * 14), baseline=3.0)
    assert state == "warning"
    assert rolling == pytest.approx(5.0)


def test_drift_ok_at_baseline_and_uses_only_last_14_days():
    mapes = [20.0] * 10 + [3.0] * 14
    state, rolling, n_days = nightly.drift_status(scores_with(mapes), baseline=3.0)
    assert state == "ok"
    assert rolling == pytest.approx(3.0)
    assert n_days == 14


def test_replace_status_swaps_only_the_marked_block():
    readme = f"top\n{nightly.STATUS_START}\nold\n{nightly.STATUS_END}\nbottom\n"
    new = nightly.replace_status(readme, "fresh line")
    assert "fresh line" in new
    assert "old" not in new
    assert new.startswith("top\n") and new.endswith("bottom\n")


def test_forecast_day_returns_one_row_per_hour_of_the_target_day():
    index = pd.date_range("2026-09-01", "2026-10-09 04:00", freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    eia = pd.DataFrame({"demand_mwh": 60000 + rng.normal(0, 500, len(index)),
                        "forecast_mwh": 60000.0}, index=index)
    observed = pd.Series(60.0, index=index)
    day = pd.Timestamp("2026-10-09")
    hours = nightly.day_hours(day)
    forecast_temp = pd.Series(55.0, index=hours)
    eia.loc[hours[0] - pd.Timedelta(hours=1):, "demand_mwh"] = np.nan

    out = nightly.forecast_day(eia, observed, forecast_temp, day)

    assert list(out.index) == list(hours)
    assert out["forecast_mwh"].between(50000, 70000).all()

"""Tests for the error metrics and the walk forward folds."""

import pandas as pd
import pytest

from src import backtest
from src.features import CENTRAL, FEATURES, TARGET


def test_mape_and_mae():
    actual = pd.Series([100.0, 200.0])
    predicted = pd.Series([110.0, 190.0])
    assert backtest.mape(actual, predicted) == pytest.approx(7.5)
    assert backtest.mae(actual, predicted) == pytest.approx(10.0)


def test_fold_months_are_last_complete_central_months():
    last_hour = pd.Timestamp("2026-10-05 19:00", tz="UTC")
    starts = backtest.fold_months(last_hour, 12)

    assert len(starts) == 12
    assert starts[0] == pd.Timestamp("2025-10-01", tz=CENTRAL)
    assert starts[-1] == pd.Timestamp("2026-09-01", tz=CENTRAL)


def synthetic_frame(start, end):
    index = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    frame = pd.DataFrame(1.0, index=index, columns=FEATURES + [TARGET, "eia_forecast_mwh"])
    return frame


def test_split_fold_holds_out_a_day_and_tests_one_central_month():
    frame = synthetic_frame("2026-01-01", "2026-04-01")
    fold_start = pd.Timestamp("2026-02-01", tz=CENTRAL)
    train, test = backtest.split_fold(frame, fold_start)

    # forecast for Feb 1 is made Jan 31, before that day finishes
    assert train.index.max() < fold_start - pd.Timedelta(days=1)
    assert test.index.min() == fold_start
    assert test.index.max() == pd.Timestamp("2026-02-28 23:00", tz=CENTRAL)
    assert len(test) == 28 * 24


def test_score_table_has_overall_row_per_method():
    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    predictions = pd.DataFrame({
        "fold": ["2026-01", "2026-01", "2026-02", "2026-02"],
        "actual": [100.0, 100.0, 100.0, 100.0],
        "model": [90.0, 110.0, 100.0, 100.0],
    }, index=index)
    table = backtest.score_table(predictions, ["model"])

    overall = table[(table["fold"] == "overall") & (table["method"] == "model")]
    assert overall["mape"].iloc[0] == pytest.approx(5.0)
    assert overall["n_hours"].iloc[0] == 4
    january = table[(table["fold"] == "2026-01") & (table["method"] == "model")]
    assert january["mae"].iloc[0] == pytest.approx(10.0)


def test_predict_fold_scores_every_method_on_the_same_hours():
    frame = synthetic_frame("2026-01-01", "2026-04-01")
    fold_start = pd.Timestamp("2026-02-01", tz=CENTRAL)
    no_eia = pd.Timestamp("2026-02-10 12:00", tz=CENTRAL)
    no_weather = pd.Timestamp("2026-02-11 12:00", tz=CENTRAL)
    frame.loc[no_eia, "eia_forecast_mwh"] = float("nan")
    forecast_temp = pd.Series(1.0, index=frame.index)
    forecast_temp[no_weather] = float("nan")

    predictions = backtest.predict_fold(frame, fold_start, forecast_temp)

    assert no_eia not in predictions.index
    assert no_weather not in predictions.index
    assert len(predictions) == 28 * 24 - 2
    assert not predictions[backtest.METHODS].isna().any().any()

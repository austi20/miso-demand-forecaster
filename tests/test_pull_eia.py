"""Tests for EIA paging and the long to wide reshape."""

import math

import pandas as pd
import pytest

from src import pull_eia


def row(period, type_code, value):
    return {"period": period, "respondent": "MISO", "type": type_code, "value": value}


def test_fetch_rows_pages_until_total(monkeypatch):
    all_rows = [row(f"2024-01-01T{h:02d}", "D", "1") for h in range(7)]
    offsets = []

    def fake_get_json(url, params):
        offsets.append(params["offset"])
        page = all_rows[params["offset"]:params["offset"] + 3]
        return {"response": {"total": str(len(all_rows)), "data": page}}

    monkeypatch.setattr(pull_eia, "PAGE_SIZE", 3)
    monkeypatch.setattr(pull_eia, "get_json", fake_get_json)
    rows = pull_eia.fetch_rows("key", "2024-01-01T00", "2024-01-01T06")
    assert rows == all_rows
    assert offsets == [0, 3, 6]


def test_fetch_rows_raises_on_empty_page_before_total(monkeypatch):
    def fake_get_json(url, params):
        return {"response": {"total": "10", "data": []}}

    monkeypatch.setattr(pull_eia, "get_json", fake_get_json)
    with pytest.raises(RuntimeError):
        pull_eia.fetch_rows("key", "a", "b")


def test_rows_to_frame_pivots_to_utc_hourly_columns():
    rows = [
        row("2024-03-10T01", "DF", "70082"),
        row("2024-03-10T00", "D", "64916"),
        row("2024-03-10T00", "DF", "67894"),
        row("2024-03-10T01", "D", None),
    ]
    frame = pull_eia.rows_to_frame(rows)

    assert list(frame.columns) == ["demand_mwh", "forecast_mwh"]
    assert frame.index.name == "timestamp_utc"
    assert str(frame.index.tz) == "UTC"
    assert frame.index[0] == pd.Timestamp("2024-03-10 00:00", tz="UTC")
    assert frame.loc[frame.index[0], "demand_mwh"] == 64916
    assert math.isnan(frame.loc[frame.index[1], "demand_mwh"])


def test_rows_to_frame_keeps_both_columns_when_one_type_absent():
    frame = pull_eia.rows_to_frame([row("2024-01-01T00", "D", "5")])
    assert list(frame.columns) == ["demand_mwh", "forecast_mwh"]
    assert math.isnan(frame["forecast_mwh"].iloc[0])


def test_rows_to_frame_rejects_duplicate_hours():
    rows = [row("2024-01-01T00", "D", "5"), row("2024-01-01T00", "D", "6")]
    with pytest.raises(ValueError):
        pull_eia.rows_to_frame(rows)


def test_rows_to_frame_adds_absent_hours_as_missing():
    rows = [row("2024-01-01T00", "D", "5"), row("2024-01-01T02", "D", "7")]
    frame = pull_eia.rows_to_frame(rows)
    assert len(frame) == 3
    assert math.isnan(frame["demand_mwh"].iloc[1])


def test_rows_to_frame_rejects_empty_pull():
    with pytest.raises(ValueError):
        pull_eia.rows_to_frame([])

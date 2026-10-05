"""Pull MISO hourly demand and EIA day-ahead demand forecast (Form EIA-930).

Run from the repo root: python -m src.pull_eia
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.fetch import get_json

URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
PAGE_SIZE = 5000  # API cap on JSON rows per call
START = "2023-01-01T00"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "eia_miso.parquet"

COLUMN_NAMES = {"D": "demand_mwh", "DF": "forecast_mwh"}


def page_params(api_key: str, start: str, end: str, offset: int) -> dict:
    return {
        "api_key": api_key,
        "frequency": "hourly",  # UTC hours, "local-hourly" is the other option
        "data[0]": "value",
        "facets[respondent][]": "MISO",
        "facets[type][]": list(COLUMN_NAMES),
        "start": start,
        "end": end,
        # fixed sort so offsets never skip or repeat rows
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "sort[1][column]": "type",
        "sort[1][direction]": "asc",
        "offset": offset,
        "length": PAGE_SIZE,
    }


def fetch_rows(api_key: str, start: str, end: str) -> list[dict]:
    """Every row between start and end, one page at a time."""
    rows = []
    offset = 0
    while True:
        body = get_json(URL, page_params(api_key, start, end, offset))
        page = body["response"]["data"]
        total = int(body["response"]["total"])
        rows.extend(page)
        offset += len(page)
        print(f"EIA: {offset:,} of {total:,} rows")
        if offset >= total:
            return rows
        if len(page) == 0:
            raise RuntimeError(f"EIA returned an empty page at {offset:,} of {total:,} rows")


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """One row per UTC hour with demand_mwh and forecast_mwh columns."""
    if not rows:
        raise ValueError("EIA returned no rows")
    long = pd.DataFrame(rows, columns=["period", "type", "value"])
    long["value"] = pd.to_numeric(long["value"])

    duplicates = long.duplicated(["period", "type"]).sum()
    if duplicates:
        raise ValueError(f"{duplicates} duplicated period/type rows from EIA")

    wide = long.pivot(index="period", columns="type", values="value")
    wide = wide.reindex(columns=list(COLUMN_NAMES)).rename(columns=COLUMN_NAMES)
    wide.index = pd.to_datetime(wide.index, format="%Y-%m-%dT%H", utc=True)
    wide.index.name = "timestamp_utc"
    wide.columns.name = None
    # hours EIA skipped become NaN rows, so lags stay aligned
    return wide.sort_index().asfreq("h")


def main(start: str = START, end: str | None = None) -> pd.DataFrame:
    load_dotenv()
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise SystemExit("EIA_API_KEY is not set. Copy .env.example to .env and add your key.")
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")

    frame = rows_to_frame(fetch_rows(api_key, start, end))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH)
    print(f"Wrote {len(frame):,} hours ({frame.index.min()} to {frame.index.max()}) to {OUT_PATH}")
    return frame


if __name__ == "__main__":
    main()

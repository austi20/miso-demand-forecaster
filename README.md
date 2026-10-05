# MISO Electricity Demand Forecaster

[![tests](https://github.com/austi20/miso-demand-forecaster/actions/workflows/tests.yml/badge.svg)](https://github.com/austi20/miso-demand-forecaster/actions/workflows/tests.yml)

Can a small gradient boosting model forecast tomorrow's hourly electricity demand on the MISO grid, which covers Michigan and 14 other states, better than the grid operator's own day ahead forecast?

This repo is being built in the open. Right now it holds the data pipeline and the data audit. The model, the walk forward backtest and the nightly retrain job come next.

## The bar to beat

EIA publishes MISO's day ahead demand forecast right next to actual demand. From January 2023 through early October 2026 that forecast missed by about 2.7% per hour on average (MAPE). It also runs high: in the median hour, actual demand came in 2.3% below the forecast. That is the number this project has to beat, not just a naive "same hour last week" guess.

## The data

- **Demand and the day ahead forecast** come from the [EIA Open Data API v2](https://www.eia.gov/opendata/), route `electricity/rto/region-data`, which serves Form EIA-930 hourly data. `src/pull_eia.py` pulls types `D` (demand) and `DF` (day ahead forecast) for `respondent=MISO` from 2023-01-01 to the current hour, about 66,000 rows at 5,000 rows per call.
- **Temperature** comes from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api). `src/pull_weather.py` pulls hourly 2 m temperature for Detroit, Chicago, Minneapolis and St. Louis and averages them into one series.

Things that were easy to get wrong:

- **Time zones.** EIA offers `hourly` (UTC) and `local-hourly`. I pull UTC, so the November daylight saving hour never shows up twice. Calendar features get built later from Central time.
- **Paging.** The API returns at most 5,000 rows per call and says so only in a warning. The pull sorts on period and type so that paging by offset cannot skip or repeat rows, and it fails loudly if any hour comes back twice.
- **Future weather.** Open-Meteo's archive endpoint fills hours that have not happened yet with model forecast values. The pull drops anything after the current hour.

The [data audit notebook](notebooks/01_data_audit.ipynb) checks for missing hours, duplicated hours and outliers. No hour is missing from the UTC grid. Demand is missing for one full day (2024-07-01) and the EIA forecast for five full days, each running midnight to midnight Central Standard Time. The most extreme hours all trace back to heat waves, cold snaps and holidays, so nothing was dropped.

The parquet files are not committed. One command rebuilds both.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add a free key from https://www.eia.gov/opendata/
python -m src.build_data
pytest
```

`python -m src.build_data` writes `data/raw/eia_miso.parquet` and `data/raw/weather_miso.parquet`. It takes about a minute.

## What would break this

- EIA-930 data is preliminary and gets revised. The "actual" demand for a given hour can change after a later pull.
- Four city temperatures are a rough proxy for a footprint covering 15 states. None of the four is in MISO South (Louisiana, Arkansas, Mississippi, east Texas), and the plain average weights Minneapolis the same as Chicago.
- Any backtest that uses observed temperature instead of a forecast of it will flatter the model, since the real forecaster did not know tomorrow's weather.

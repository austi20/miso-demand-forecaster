"""Nightly job: forecast tomorrow, score past forecasts, check for drift.

Run after python -m src.build_data, from the repo root: python -m src.nightly
To seed the score log with recent days: python -m src.nightly --replay-days 14
"""

import argparse
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor

from src import pull_eia, pull_weather
from src.backtest import MODEL_PARAMS, TRAIN_GAP, mae, mape
from src.features import CENTRAL, FEATURES, TARGET, build_features

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "metrics"
FORECASTS_PATH = METRICS_DIR / "forecasts.csv"
SCORES_PATH = METRICS_DIR / "daily_scores.csv"
BACKTEST_PATH = ROOT / "results" / "backtest.csv"
README_PATH = ROOT / "README.md"

DRIFT_WINDOW_DAYS = 14
DRIFT_LIMIT = 1.5  # warn above 150% of the backtest MAPE
STATUS_START = "<!-- status:start -->"
STATUS_END = "<!-- status:end -->"


def day_hours(day: pd.Timestamp) -> pd.DatetimeIndex:
    """UTC hours of one Central time day. day is a naive date."""
    start = day.tz_localize(CENTRAL)
    end = (day + pd.Timedelta(days=1)).tz_localize(CENTRAL)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def extend_to(eia: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Add empty hours through the end of day, so lags exist for future rows."""
    end = day_hours(day)[-1]
    index = pd.date_range(eia.index.min(), max(end, eia.index.max()), freq="h")
    return eia.reindex(index)


def forecast_day(eia: pd.DataFrame, observed: pd.Series, forecast_temp: pd.Series,
                 day: pd.Timestamp) -> pd.DataFrame:
    """Retrain on data known by the morning before day, then forecast its 24 hours."""
    hours = day_hours(day)
    frame = build_features(extend_to(eia, day), observed)

    train = frame[frame.index < hours[0] - TRAIN_GAP].dropna(subset=FEATURES + [TARGET])
    model = LGBMRegressor(**MODEL_PARAMS)
    model.fit(train[FEATURES], train[TARGET])

    inputs = frame.loc[hours, FEATURES].assign(temp_f=forecast_temp.reindex(hours))
    return pd.DataFrame({"forecast_mwh": model.predict(inputs)}, index=hours)


def fetch_forecast_temps(first_day: pd.Timestamp, last_day: pd.Timestamp) -> pd.Series:
    """Four city mean of the 48 hour ahead temperature forecast."""
    start = day_hours(first_day)[0].strftime("%Y-%m-%d")
    end = day_hours(last_day)[-1].strftime("%Y-%m-%d")
    columns = {}
    for city, (lat, lon) in pull_weather.CITIES.items():
        columns[city] = pull_weather.fetch_city(
            lat, lon, start, end,
            url=pull_weather.FORECAST_URL, variable=pull_weather.FORECAST_VARIABLE)
    return pull_weather.add_city_mean(pd.DataFrame(columns))["temp_f_mean"]


def score_day(forecast: pd.DataFrame, eia: pd.DataFrame) -> dict | None:
    """Score one day against actuals, or None until every hour has demand."""
    eia = eia.reindex(forecast.index)
    if eia["demand_mwh"].isna().any():
        return None
    # same hours for every method, like the backtest
    hours = pd.DataFrame({
        "actual": eia["demand_mwh"],
        "lightgbm": forecast["forecast_mwh"],
        "eia": eia["forecast_mwh"],
        "naive": eia["demand_lag168"],
    }).dropna()
    if hours.empty:
        return None

    row = {"n_hours": len(hours)}
    for method in ["lightgbm", "eia", "naive"]:
        row[f"{method}_mape"] = mape(hours["actual"], hours[method])
        row[f"{method}_mae"] = mae(hours["actual"], hours[method])
    return row


def drift_status(scores: pd.DataFrame, baseline: float) -> tuple[str, float, int]:
    """State, rolling LightGBM MAPE and days used over the last 14 scored days."""
    recent = scores.sort_values("day").tail(DRIFT_WINDOW_DAYS)
    n_days = len(recent)
    if n_days < DRIFT_WINDOW_DAYS:
        return "collecting", float("nan"), n_days
    rolling = (recent["lightgbm_mape"] * recent["n_hours"]).sum() / recent["n_hours"].sum()
    state = "warning" if rolling > DRIFT_LIMIT * baseline else "ok"
    return state, float(rolling), n_days


def replace_status(readme: str, block: str) -> str:
    start = readme.index(STATUS_START) + len(STATUS_START)
    end = readme.index(STATUS_END)
    return readme[:start] + "\n" + block + "\n" + readme[end:]


def status_text(scores: pd.DataFrame, baseline: float) -> str:
    state, rolling, n_days = drift_status(scores, baseline)
    last = scores["day"].max()
    n_live = int((scores["source"] == "live").sum())
    tail = f"{n_live} of {len(scores)} scored days were forecast live, the rest replayed."
    if state == "collecting":
        return (f"Collecting scores: {n_days} of {DRIFT_WINDOW_DAYS} days so far, "
                f"last scored day {last}. {tail}")

    recent = scores.sort_values("day").tail(DRIFT_WINDOW_DAYS)
    eia_mape = (recent["eia_mape"] * recent["n_hours"]).sum() / recent["n_hours"].sum()
    line = (f"Last scored day {last}. 14 day MAPE: LightGBM {rolling:.2f}%, "
            f"EIA {eia_mape:.2f}%. Backtest LightGBM was {baseline:.2f}%. {tail}")
    if state == "warning":
        return f"**Drift warning.** {line} The limit is {DRIFT_LIMIT * baseline:.2f}%."
    return f"No drift. {line}"


def load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def main(replay_days: int = 0) -> None:
    eia = pd.read_parquet(pull_eia.OUT_PATH)
    observed = pd.read_parquet(pull_weather.OUT_PATH)["temp_f_mean"]

    forecasts = load_csv(FORECASTS_PATH, ["day", "timestamp_utc", "forecast_mwh", "source", "made_at_utc"])
    scores = load_csv(SCORES_PATH, ["day", "source", "n_hours"])
    known = set(forecasts["day"])

    today = pd.Timestamp.now(tz=CENTRAL).tz_localize(None).normalize()
    wanted = [(today - pd.Timedelta(days=k), "replay") for k in range(replay_days, 0, -1)]
    wanted.append((today + pd.Timedelta(days=1), "live"))
    wanted = [(day, source) for day, source in wanted if day.strftime("%Y-%m-%d") not in known]

    if wanted:
        temps = fetch_forecast_temps(wanted[0][0], wanted[-1][0])
        made_at = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ")
        new_rows = []
        for day, source in wanted:
            out = forecast_day(eia, observed, temps, day)
            out = out.assign(day=day.strftime("%Y-%m-%d"), source=source, made_at_utc=made_at)
            new_rows.append(out.rename_axis("timestamp_utc").reset_index())
            print(f"Forecast {day.date()} ({source})")
        forecasts = pd.concat([forecasts] + new_rows, ignore_index=True)

    eia = eia.assign(demand_lag168=eia["demand_mwh"].shift(168))
    scored = set(scores["day"])
    score_rows = []
    for day, group in forecasts.groupby("day"):
        if day in scored:
            continue
        forecast = group.assign(timestamp_utc=pd.to_datetime(group["timestamp_utc"], utc=True))
        row = score_day(forecast.set_index("timestamp_utc"), eia)
        if row is not None:
            score_rows.append({"day": day, "source": group["source"].iloc[0], **row})
            print(f"Scored {day}")
    if score_rows:
        scores = pd.concat([scores, pd.DataFrame(score_rows)], ignore_index=True)

    METRICS_DIR.mkdir(exist_ok=True)
    forecasts.round({"forecast_mwh": 0}).to_csv(FORECASTS_PATH, index=False)
    scores.sort_values("day").round(2).to_csv(SCORES_PATH, index=False)

    if len(scores) and README_PATH.exists():
        backtest = pd.read_csv(BACKTEST_PATH)
        baseline = float(backtest[(backtest["fold"] == "overall")
                                  & (backtest["method"] == "lightgbm")]["mape"].iloc[0])
        readme = README_PATH.read_text(encoding="utf-8")
        README_PATH.write_text(replace_status(readme, status_text(scores, baseline)),
                               encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-days", type=int, default=0)
    main(parser.parse_args().replay_days)

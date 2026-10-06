"""Walk forward backtest: LightGBM vs seasonal naive vs EIA's day ahead forecast.

For each of the last 12 complete months, train on everything before that month
and forecast every hour in it. All runs are logged to MLflow.

Run from the repo root: python -m src.backtest
"""

from pathlib import Path

import matplotlib
import mlflow
import pandas as pd
from lightgbm import LGBMRegressor

from src import pull_eia, pull_weather
from src.features import CENTRAL, FEATURES, TARGET, build_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TRACKING_URI = "sqlite:///" + (ROOT / "mlflow.db").as_posix()
EXPERIMENT = "miso-day-ahead-backtest"

N_FOLDS = 12
# forecast for day 1 is made the morning before, so that day is unfinished
TRAIN_GAP = pd.Timedelta(days=1)
MODEL_PARAMS = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31,
                "random_state": 0, "verbose": -1}

METHODS = ["lightgbm", "eia_day_ahead", "seasonal_naive", "lightgbm_observed_weather"]
METHOD_LABELS = {
    "lightgbm": "LightGBM",
    "eia_day_ahead": "EIA day ahead forecast",
    "seasonal_naive": "Same hour last week",
}


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    return float(((actual - predicted).abs() / actual).mean() * 100)


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float((actual - predicted).abs().mean())


def fold_months(last_hour: pd.Timestamp, n_folds: int) -> list[pd.Timestamp]:
    """Starts of the last n complete Central time months, oldest first."""
    local = last_hour.tz_convert(CENTRAL)
    this_month = pd.Timestamp(year=local.year, month=local.month, day=1, tz=CENTRAL)
    return [this_month - pd.DateOffset(months=k) for k in range(n_folds, 0, -1)]


def split_fold(frame: pd.DataFrame, fold_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_end = fold_start + pd.DateOffset(months=1)
    train = frame[frame.index < fold_start - TRAIN_GAP]
    test = frame[(frame.index >= fold_start) & (frame.index < fold_end)]
    return train, test


def predict_fold(frame: pd.DataFrame, fold_start: pd.Timestamp,
                 forecast_temp: pd.Series) -> pd.DataFrame:
    """Every method's forecast for each hour of one test month."""
    train, test = split_fold(frame, fold_start)
    train = train.dropna(subset=FEATURES + [TARGET])
    test = test.assign(temp_f_forecast=forecast_temp.reindex(test.index))
    # score all methods on the same hours
    test = test.dropna(subset=FEATURES + [TARGET, "eia_forecast_mwh", "temp_f_forecast"])

    model = LGBMRegressor(**MODEL_PARAMS)
    model.fit(train[FEATURES], train[TARGET])
    # trained on observed weather, scored on what was forecast 48h ahead
    forecast_inputs = test[FEATURES].assign(temp_f=test["temp_f_forecast"])

    return pd.DataFrame({
        "fold": fold_start.strftime("%Y-%m"),
        "actual": test[TARGET],
        "lightgbm": model.predict(forecast_inputs),
        "lightgbm_observed_weather": model.predict(test[FEATURES]),
        "eia_day_ahead": test["eia_forecast_mwh"],
        "seasonal_naive": test["demand_lag168"],
    }, index=test.index)


def score_table(predictions: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """MAPE and MAE per fold and method, plus an overall row per method."""
    groups = list(predictions.groupby("fold")) + [("overall", predictions)]
    rows = []
    for fold, group in groups:
        for method in methods:
            rows.append({
                "fold": fold,
                "method": method,
                "mape": mape(group["actual"], group[method]),
                "mae": mae(group["actual"], group[method]),
                "n_hours": len(group),
            })
    return pd.DataFrame(rows)


def plot_folds(table: pd.DataFrame, path: Path) -> None:
    """Monthly MAPE, one line per method."""
    colors = {"lightgbm": "#2a78d6", "eia_day_ahead": "#eb6834", "seasonal_naive": "#1baf7a"}
    monthly = table[table["fold"] != "overall"]
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for method, color in colors.items():
        rows = monthly[monthly["method"] == method]
        ax.plot(rows["fold"], rows["mape"], color=color, linewidth=2, marker="o",
                markersize=6, label=METHOD_LABELS[method])
        last = rows["mape"].iloc[-1]
        ax.annotate(f"{last:.1f}%", (len(rows) - 1, last), xytext=(8, 0),
                    textcoords="offset points", va="center", color="#52514e", fontsize=9)

    ax.set_ylabel("MAPE (%)", color="#52514e")
    ax.set_ylim(0, monthly["mape"].max() * 1.2)
    ax.set_title("Hourly forecast error by test month, MISO", loc="left", color="#0b0b0b")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.tick_params(colors="#52514e", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=9, labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def log_to_mlflow(table: pd.DataFrame, folds: list[pd.Timestamp], artifacts: list[Path]) -> None:
    """One parent run for the backtest, one nested run per method."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="walk_forward_12_months"):
        mlflow.log_params({
            "n_folds": len(folds),
            "first_test_month": folds[0].strftime("%Y-%m"),
            "last_test_month": folds[-1].strftime("%Y-%m"),
            "train_gap_hours": int(TRAIN_GAP.total_seconds() // 3600),
            "test_weather": pull_weather.FORECAST_VARIABLE,
            **MODEL_PARAMS,
        })
        mlflow.log_text("\n".join(FEATURES), "features.txt")
        for path in artifacts:
            mlflow.log_artifact(str(path))

        for method in METHODS:
            with mlflow.start_run(run_name=method, nested=True):
                mlflow.log_param("method", method)
                rows = table[table["method"] == method]
                monthly = rows[rows["fold"] != "overall"]
                for step, row in enumerate(monthly.itertuples(), start=1):
                    mlflow.log_metric("mape", row.mape, step=step)
                    mlflow.log_metric("mae", row.mae, step=step)
                overall = rows[rows["fold"] == "overall"].iloc[0]
                mlflow.log_metrics({"overall_mape": overall["mape"], "overall_mae": overall["mae"]})


def main() -> pd.DataFrame:
    eia = pd.read_parquet(pull_eia.OUT_PATH)
    observed = pd.read_parquet(pull_weather.OUT_PATH)["temp_f_mean"]
    forecast = pd.read_parquet(pull_weather.FORECAST_OUT_PATH)["temp_f_mean"]

    frame = build_features(eia, observed)
    last_hour = eia["demand_mwh"].dropna().index.max()
    folds = fold_months(last_hour, N_FOLDS)

    predictions = pd.concat([predict_fold(frame, start, forecast) for start in folds])
    table = score_table(predictions, METHODS)

    RESULTS_DIR.mkdir(exist_ok=True)
    table_path = RESULTS_DIR / "backtest.csv"
    plot_path = RESULTS_DIR / "backtest.png"
    table.round({"mape": 2, "mae": 0}).to_csv(table_path, index=False)
    plot_folds(table, plot_path)
    log_to_mlflow(table, folds, [table_path, plot_path])

    print(table.pivot(index="fold", columns="method", values="mape").round(2).to_string())
    print(table[table["fold"] == "overall"].round(2).to_string(index=False))
    return table


if __name__ == "__main__":
    main()

"""Build the hourly feature table for day ahead demand forecasting.

The forecast for Central time day D is made the morning of D-1. Every input
must be known 48 hours before the hour it predicts, which covers all 24 hours
of D from that morning.
"""

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    USLaborDay,
    USMemorialDay,
    USThanksgivingDay,
    sunday_to_monday,
)

CENTRAL = "America/Chicago"
TARGET = "demand_mwh"
FEATURES = ["demand_lag48", "demand_lag168", "hour", "day_of_week", "is_holiday", "temp_f"]


class GridHolidays(AbstractHolidayCalendar):
    """The six NERC off peak holidays. Sunday ones move to Monday."""
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMemorialDay,
        Holiday("Independence Day", month=7, day=4, observance=sunday_to_monday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=sunday_to_monday),
    ]


def build_features(eia: pd.DataFrame, temp_f: pd.Series) -> pd.DataFrame:
    """One row per UTC hour: target, EIA's forecast, and model features."""
    steps = eia.index.to_series().diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise ValueError("EIA index must be a gapless hourly grid, or the lags shift")

    frame = pd.DataFrame(index=eia.index)
    frame[TARGET] = eia["demand_mwh"]
    frame["eia_forecast_mwh"] = eia["forecast_mwh"]
    frame["demand_lag48"] = eia["demand_mwh"].shift(48)
    frame["demand_lag168"] = eia["demand_mwh"].shift(168)

    local = eia.index.tz_convert(CENTRAL)
    frame["hour"] = local.hour
    frame["day_of_week"] = local.dayofweek
    local_dates = local.tz_localize(None).normalize()
    holidays = GridHolidays().holidays(local_dates.min(), local_dates.max())
    frame["is_holiday"] = local_dates.isin(holidays).astype(int)

    frame["temp_f"] = temp_f.reindex(eia.index)
    return frame

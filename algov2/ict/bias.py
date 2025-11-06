# algov2/ict/bias.py
from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np

def _ensure_dt_index_utc(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    if getattr(df.index, "tz", None) is None:
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
    return df

def compute_bias(
    df: pd.DataFrame,
    cfg=None,
    tz: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Adds:
      - bias  ∈ {-1,0,1} based on close vs local day open
      - pd_zone ∈ {'premium','discount'} via previous day's (H+L)/2

    Mutates df in-place and returns it.
    """
    if df.empty:
        return df

    df = _ensure_dt_index_utc(df)
    tz_local = getattr(cfg, "tz", None) or tz or "US/Eastern"

    # local dating
    idx_local = df.index.tz_convert(tz_local)
    day_key = idx_local.normalize()

    # day-open (use first CLOSE of each local day; fall back to OPEN if needed)
    first_per_day = ~day_key.duplicated()
    day_open_series = pd.Series(index=df.index, dtype=float)
    # prefer close of first bar
    day_open_series[first_per_day] = df.loc[first_per_day, "close"]
    # forward fill inside day
    day_open_series = day_open_series.groupby(day_key).transform(lambda s: s.ffill().bfill())
    # if still NaN, fallback to open
    mask_nan = day_open_series.isna()
    if mask_nan.any():
        fallback = pd.Series(index=df.index, dtype=float)
        fallback[first_per_day] = df.loc[first_per_day, "open"]
        fallback = fallback.groupby(day_key).transform(lambda s: s.ffill().bfill())
        day_open_series[mask_nan] = fallback[mask_nan]

    # bias by threshold band (avoid flapping around opens)
    band_pts = 2.0  # ~2 NQ points comfort band; tune later
    delta = df["close"] - day_open_series
    bias = np.where(delta > band_pts, 1, np.where(delta < -band_pts, -1, 0))

    # previous-day premium/discount midpoint
    daily_high = df["high"].groupby(day_key).transform("max")
    daily_low  = df["low"].groupby(day_key).transform("min")
    daily_mid  = (daily_high + daily_low) / 2.0
    # shift mid by one local day to make it "previous day"
    prev_mid = daily_mid.groupby(day_key).transform("first")
    prev_mid = prev_mid.groupby(day_key).shift(1)
    # fill first day with same day's mid (harmless, used only as label)
    prev_mid = prev_mid.fillna(daily_mid)

    pd_zone = np.where(df["close"] >= prev_mid, "premium", "discount")

    df["bias"] = bias.astype(int)
    df["pd_zone"] = pd.Categorical(pd_zone, categories=["discount", "premium"])

    # convenience columns (optional consumers)
    df["day_open"] = day_open_series
    return df
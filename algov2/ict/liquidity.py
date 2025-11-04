# algov2/ict/liquidity.py
from __future__ import annotations
import pandas as pd
import numpy as np

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        for c in ("datetime", "date", "time", "timestamp"):
            if c in df.columns:
                out = df.copy()
                out.index = pd.to_datetime(out[c], utc=True, errors="coerce")
                out = out[~out.index.isna()]
                return out
        raise TypeError("liquidity.py: DataFrame must have a DatetimeIndex or a datetime-like column")
    if isinstance(df.index, pd.PeriodIndex):
        df = df.copy()
        df.index = df.index.to_timestamp()
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    return df

def mark_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    df = _ensure_dt_index(df)
    h = df["high"]; l = df["low"]
    # swing high: strictly greater than neighbors on both sides
    swing_high = pd.Series(False, index=df.index)
    swing_low  = pd.Series(False, index=df.index)
    for i in range(left, len(df)-right):
        window_h = h.iloc[i-left:i+right+1]
        window_l = l.iloc[i-left:i+right+1]
        mid_h = window_h.iloc[left]
        mid_l = window_l.iloc[left]
        if mid_h == window_h.max() and (window_h.iloc[:left] < mid_h).all() and (window_h.iloc[left+1:] < mid_h).all():
            swing_high.iloc[i] = True
        if mid_l == window_l.min() and (window_l.iloc[:left] > mid_l).all() and (window_l.iloc[left+1:] > mid_l).all():
            swing_low.iloc[i] = True

    out = df.copy()
    out["swing_high"] = swing_high.astype(int)
    out["swing_low"] = swing_low.astype(int)
    return out

def mark_equal_highs_lows(
    df: pd.DataFrame,
    tick_size: float = 0.25,
    tol_ticks: float = 2.0,
    lookback: int = 100
) -> pd.DataFrame:
    """
    Flags equal highs/lows (resting liquidity). Within tol_ticks * tick_size.
    Only compares to prior swings within lookback.
    Adds: eq_high, eq_low (bool)
    """
    df = _ensure_dt_index(df)
    out = df.copy()
    out = out if "swing_high" in out.columns else mark_swings(out)
    eq_high = pd.Series(False, index=out.index)
    eq_low  = pd.Series(False, index=out.index)

    tol = tol_ticks * tick_size
    swing_high_idxs = list(out.index[out["swing_high"] == 1])
    swing_low_idxs  = list(out.index[out["swing_low"] == 1])

    # equal highs
    last_val = None; last_idx = None
    for idx in swing_high_idxs:
        if last_val is None:
            last_val, last_idx = out.at[idx, "high"], idx
            continue
        if (idx - last_idx).components.days*1440 + (idx - last_idx).components.minutes > lookback*5:
            last_val, last_idx = out.at[idx, "high"], idx
            continue
        if abs(out.at[idx, "high"] - last_val) <= tol:
            eq_high.at[idx] = True
            eq_high.at[last_idx] = True
        last_val, last_idx = out.at[idx, "high"], idx

    # equal lows
    last_val = None; last_idx = None
    for idx in swing_low_idxs:
        if last_val is None:
            last_val, last_idx = out.at[idx, "low"], idx
            continue
        if (idx - last_idx).components.days*1440 + (idx - last_idx).components.minutes > lookback*5:
            last_val, last_idx = out.at[idx, "low"], idx
            continue
        if abs(out.at[idx, "low"] - last_val) <= tol:
            eq_low.at[idx] = True
            eq_low.at[last_idx] = True
        last_val, last_idx = out.at[idx, "low"], idx

    out["eq_high"] = eq_high.astype(int)
    out["eq_low"]  = eq_low.astype(int)
    return out

def attach_prev_day_hilo(df: pd.DataFrame, tz_local: str = "US/Eastern") -> pd.DataFrame:
    """
    Adds prev_day_high / prev_day_low forward-filled intraday.
    """
    df = _ensure_dt_index(df)
    loc = df.tz_convert(tz_local)
    daily = loc.resample("1D", label="left", closed="left").agg({"high":"max", "low":"min"})
    daily = daily.shift(1)  # previous day
    daily.columns = ["prev_day_high", "prev_day_low"]
    mapped = daily.reindex(loc.index, method="ffill")
    out = df.copy()
    out["prev_day_high"] = mapped["prev_day_high"].values
    out["prev_day_low"]  = mapped["prev_day_low"].values
    return out

def build_liquidity_map(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    tick_size: float = 0.25,
    tol_ticks: float = 2.0,
    tz_local: str = "US/Eastern"
) -> pd.DataFrame:
    """
    Combines swings, equal highs/lows and previous day Hi/Lo into a simple liquidity map.
    Adds columns:
      swing_high, swing_low, eq_high, eq_low, prev_day_high, prev_day_low
    """
    out = df.pipe(mark_swings, left=left, right=right)
    out = out.pipe(mark_equal_highs_lows, tick_size=tick_size, tol_ticks=tol_ticks)
    out = out.pipe(attach_prev_day_hilo, tz_local=tz_local)
    return out
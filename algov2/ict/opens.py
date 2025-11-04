# algov2/ict/opens.py
from __future__ import annotations
import pandas as pd
import numpy as np

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        # try common column names
        for c in ("datetime", "date", "time", "timestamp"):
            if c in df.columns:
                out = df.copy()
                out.index = pd.to_datetime(out[c], utc=True, errors="coerce")
                out = out[~out.index.isna()]
                return out
        raise TypeError("opens.py: DataFrame must have a DatetimeIndex or a datetime-like column")
    if isinstance(df.index, pd.PeriodIndex):
        df = df.copy()
        df.index = df.index.to_timestamp()
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    return df

def add_session_opens(
    df: pd.DataFrame,
    tz_local: str = "US/Eastern",
    session_time: str = "08:30",   # HH:MM local (NY equities futures open)
) -> pd.DataFrame:
    """
    Adds:
      - open_midnight: first bar's close of each local day (00:00 local)
      - open_0830:     08:30 local session open close (next valid bar if missing)
      - above_midnight / above_0830 (bool)
      - dist_midnight / dist_0830 (close minus those opens)
    """
    df = _ensure_dt_index(df)
    out = df.copy()

    # Work in local time for daily/session boundaries
    loc = out.tz_convert(tz_local)

    # Midnight open = first bar of each local calendar day
    first_idx = loc.index.normalize()
    first_of_day_mask = ~first_idx.duplicated(keep="first")
    # compute midnight open values
    mid_vals = pd.Series(np.nan, index=loc.index)
    mid_vals[first_of_day_mask] = loc["close"][first_of_day_mask]
    mid_vals = mid_vals.ffill()

    # 08:30 open (find the first bar at or after 08:30 each local day)
    t0830 = pd.to_datetime(loc.index.normalize().strftime("%Y-%m-%d ") + session_time + ":00")
    t0830 = t0830.tz_localize(tz_local)
    # for each day, find first index >= t0830
    pos = loc.index.searchsorted(t0830)
    valid_pos = pos < len(loc.index)
    open0830_vals = pd.Series(np.nan, index=loc.index)
    if valid_pos.any():
        inds = pos[valid_pos]
        open0830_vals.iloc[inds] = loc["close"].iloc[inds]
    open0830_vals = open0830_vals.ffill()

    # write back in UTC index space (same index)
    out["open_midnight"] = mid_vals.values
    out["open_0830"] = open0830_vals.values

    # Helpers
    out["above_midnight"] = (out["close"] > out["open_midnight"]).astype(int)
    out["above_0830"] = (out["close"] > out["open_0830"]).astype(int)
    out["dist_midnight"] = out["close"] - out["open_midnight"]
    out["dist_0830"] = out["close"] - out["open_0830"]
    return out
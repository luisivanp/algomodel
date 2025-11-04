# algov2/data/prep.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Optional

__all__ = [
    "PrepTypes",
    "ensure_prep",
    "prepare_market_df",
    "attach_daily_levels",
    "attach_premium_discount",
    "attach_pd_zone",
]

# ---------------------------------------------------------------------------
# Types / constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrepTypes:
    dt: str = "dt"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"

    day_open: str = "day_open"
    day_high: str = "day_high"
    day_low: str = "day_low"

    pd_mid: str = "pd_mid"         # (day_high + day_low)/2
    pd_zone: str = "pd_zone"       # "premium" if close >= mid else "discount"


PT = PrepTypes()


# ---------------------------------------------------------------------------
# Public entrypoint used by runner/pipeline
# ---------------------------------------------------------------------------

def prepare_market_df(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "5m",
    tz_local: str = "US/Eastern",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    High-level prep:
      1) fetch OHLCV with algov2.data.loader.fetch_prices
      2) clean/timezone normalize
      3) attach daily levels + premium/discount labels

    Returns a DataFrame with columns:
      ['dt','open','high','low','close','volume','date',
       'day_open','day_high','day_low','pd_mid','pd_zone', ...]
    """
    try:
        from ..loader import fetch_prices  # local import to avoid cycles
    except Exception as e:
        raise ImportError(f"prep.prepare_market_df requires data.loader.fetch_prices: {e}")

    df = fetch_prices(
        symbol=symbol,
        start=start,
        end=end,
        timeframe=timeframe,
        tz_local=tz_local,
        use_cache=use_cache,
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("prepare_market_df: provider returned empty DataFrame")

    df = ensure_prep(df, tz_local=tz_local)
    # attach daily levels / premium-discount (idempotent if already set)
    df = attach_daily_levels(df, tz_local=tz_local)
    return df


# ---------------------------------------------------------------------------
# Core prep helpers
# ---------------------------------------------------------------------------

def ensure_prep(df: pd.DataFrame, tz_local: str = "US/Eastern") -> pd.DataFrame:
    """
    Normalize/validate the market frame:
      - Ensure 'dt' exists and is tz-aware converted to tz_local
      - Ensure OHLCV columns exist and are numeric
      - Sort by dt and drop any rows with missing OHLC
      - Keep index as RangeIndex; all time in the 'dt' column
    """
    d = df.copy()

    # Ensure we have dt column in local tz
    if PT.dt not in d.columns:
        d[PT.dt] = _ensure_dt_column(d, tz_local=tz_local)
    else:
        d[PT.dt] = _to_localized_dt(d[PT.dt], tz_local=tz_local)

    # Ensure ohlcv columns exist & numeric
    _require_price_columns(d, required=[PT.open, PT.high, PT.low, PT.close, PT.volume])

    for c in (PT.open, PT.high, PT.low, PT.close, PT.volume):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Sort, drop NA rows where essential fields are missing
    d = d.sort_values(PT.dt).reset_index(drop=True)
    d = d.dropna(subset=[PT.open, PT.high, PT.low, PT.close])

    return d


# ---------------------------------------------------------------------------
# Compatibility shims expected by pipeline
# ---------------------------------------------------------------------------

def attach_daily_levels(df: pd.DataFrame, tz_local: str = "US/Eastern") -> pd.DataFrame:
    """
    Ensure daily OHLC references and premium/discount labels exist.
    Idempotent: recomputes consistently; returns a copy.
    """
    d = df.copy()

    # Ensure proper datetime column
    if PT.dt not in d.columns:
        d[PT.dt] = _ensure_dt_column(d, tz_local=tz_local)
    else:
        d[PT.dt] = _to_localized_dt(d[PT.dt], tz_local=tz_local)

    # Make sure essential columns exist
    _require_price_columns(d, required=[PT.open, PT.high, PT.low, PT.close])

    # Compute/overwrite daily features
    d = _daily_features(d, tz_local=tz_local)

    # Recompute premium/discount from daily high/low
    d[PT.pd_mid] = (d[PT.day_high] + d[PT.day_low]) / 2.0
    d[PT.pd_zone] = np.where(d[PT.close] >= d[PT.pd_mid], "premium", "discount")

    # Nice column ordering
    prefer = [
        PT.dt, "date",
        PT.open, PT.high, PT.low, PT.close, PT.volume,
        PT.day_open, PT.day_high, PT.day_low, PT.pd_mid, PT.pd_zone,
    ]
    extras = [c for c in d.columns if c not in prefer]
    d = d[prefer + extras]
    return d


def attach_premium_discount(df: pd.DataFrame, tz_local: str = "US/Eastern") -> pd.DataFrame:
    """
    Convenience wrapper that ensures daily levels exist and then (re)labels pd_zone.
    """
    d = attach_daily_levels(df, tz_local=tz_local)
    d[PT.pd_mid] = (d[PT.day_high] + d[PT.day_low]) / 2.0
    d[PT.pd_zone] = np.where(d[PT.close] >= d[PT.pd_mid], "premium", "discount")
    return d


# Alias used by older code
attach_pd_zone = attach_premium_discount


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _ensure_dt_column(df: pd.DataFrame, tz_local: str) -> pd.Series:
    """
    Build a tz-aware local 'dt' series from either:
      - an existing DatetimeIndex
      - a column named 'Datetime', 'timestamp', 'time', or 'date'
    """
    # 1) From index
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx = idx.tz_convert(tz_local)
        return pd.Series(idx, index=df.index, name=PT.dt)

    # 2) From common datetime columns
    for cand in ("dt", "Datetime", "datetime", "timestamp", "time", "date"):
        if cand in df.columns:
            ser = pd.to_datetime(df[cand], errors="coerce")
            if ser.dt.tz is None:
                ser = ser.dt.tz_localize("UTC")
            ser = ser.dt.tz_convert(tz_local)
            return ser.rename(PT.dt)

    # If nothing found, attempt to coerce the first column
    first_col = df.columns[0]
    ser = pd.to_datetime(df[first_col], errors="coerce")
    if ser.dt.tz is None:
        ser = ser.dt.tz_localize("UTC")
    ser = ser.dt.tz_convert(tz_local)
    return ser.rename(PT.dt)


def _to_localized_dt(series: pd.Series, tz_local: str) -> pd.Series:
    ser = pd.to_datetime(series, errors="coerce")
    if ser.dt.tz is None:
        ser = ser.dt.tz_localize("UTC")
    return ser.dt.tz_convert(tz_local)


def _require_price_columns(d: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in d.columns]
    if missing:
        # try a light rename from common variants (e.g., yfinance stacked cols)
        fallback_map = {}
        for c in required:
            if c in d.columns:
                continue
            # look for case/alias variants
            candidates = [
                c, c.capitalize(), c.upper(),
                f"{c}_nq=f", f"{c}_NQ=F",
                f"{c}_nqf",  f"{c}_NQF",
            ]
            found = next((x for x in candidates if x in d.columns), None)
            if found:
                fallback_map[found] = c

        if fallback_map:
            d.rename(columns=fallback_map, inplace=True)
            missing = [c for c in required if c not in d.columns]

    if missing:
        raise ValueError(f"prep: DataFrame missing required columns: {missing}")


def _daily_features(d: pd.DataFrame, tz_local: str) -> pd.DataFrame:
    """
    Compute daily open/high/low and add a 'date' column (naive date in local tz).
    Assumes d[PT.dt] is tz-aware in tz_local and OHLC columns exist.
    """
    out = d.copy()
    # Local date for grouping
    local_dt = _to_localized_dt(out[PT.dt], tz_local=tz_local)
    out["date"] = local_dt.dt.date

    # group transforms
    grp = out.groupby("date", observed=True)

    # first open per day
    out[PT.day_open] = grp[PT.open].transform("first")
    # extrema per day
    out[PT.day_high] = grp[PT.high].transform("max")
    out[PT.day_low] = grp[PT.low].transform("min")

    return out
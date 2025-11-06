# algov2/data/prep.py
from __future__ import annotations
import pandas as pd
from typing import Optional


def prepare_market_df(symbol: str,
                      start: str,
                      end: str,
                      timeframe: str,
                      tz_local: Optional[str] = None,
                      use_cache: bool = False,
                      cache_params: Optional[dict] = None) -> pd.DataFrame:
    """
    Fetch OHLCV and return a DataFrame with:
      - UTC tz-aware DatetimeIndex (strict)
      - columns: open, high, low, close, volume (float)
    No index resets here. Timestamps stay intact.
    """
    from ..loader import fetch_prices

    df = fetch_prices(symbol, start, end, timeframe, tz_local)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("prepare_market_df: fetch_prices must return a DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    need = ["open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"prepare_market_df: missing required columns: {missing}")

    df = df.astype({c: float for c in need})
    df = df[~df.index.duplicated(keep="last")].sort_index()

    try:
        print(f"[prep] bars={len(df)} first={df.index.min()} last={df.index.max()} tz={df.index.tz}")
    except Exception:
        pass

    return df


def attach_daily_levels(df: pd.DataFrame, tz_local: Optional[str] = None) -> pd.DataFrame:
    """
    Add previous-day levels onto an intraday dataframe:
      - prev_high, prev_low, prev_open (from the *previous* local day)
      - d_open (today's local midnight open, optional reference)

    Works by converting to local tz for daily boundaries, resampling to 1D,
    then shifting daily index forward one day so a backward asof-merge
    attaches "previous day" values to intraday timestamps.

    Returns a new dataframe with the same (UTC) index as input.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("attach_daily_levels: df must have a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("attach_daily_levels: df index must be timezone-aware (UTC)")

    # Work on a local-time copy to compute proper daily sessions
    df_local = df.copy()
    if tz_local:
        df_local.index = df_local.index.tz_convert(tz_local)
    else:
        # default: treat the current tz (UTC) as local
        tz_local = "UTC"

    # Daily aggregates (local days)
    d_open = df_local["open"].resample("1D").first()
    d_high = df_local["high"].resample("1D").max()
    d_low  = df_local["low"].resample("1D").min()

    # Build a daily frame with previous-day values; index shift +1 day lets asof() pick prev day
    levels = pd.DataFrame({
        "prev_high": d_high,
        "prev_low": d_low,
        "prev_open": d_open,
        "d_open": d_open,  # same-day open (we won’t shift this one for reference)
    })

    # Shift only the "previous day" levels by moving their index +1 day
    levels_prev = levels[["prev_high", "prev_low", "prev_open"]].copy()
    levels_prev.index = levels_prev.index + pd.Timedelta(days=1)

    # Keep same-day open separately at midnight of the same local day
    levels_same = levels[["d_open"]].copy()

    # Intraday frame in local tz (to align with daily midnights)
    intr_local = pd.DataFrame(index=df_local.index)
    intr_local = intr_local.reset_index().rename(columns={"index": "ts"})

    prev_local = levels_prev.reset_index().rename(columns={"index": "ts"})
    same_local = levels_same.reset_index().rename(columns={"index": "ts"})

    # As-of merge: at any time during day D, we pull prev_day levels (assigned at D 00:00 local)
    prev_join = pd.merge_asof(
        intr_local.sort_values("ts"),
        prev_local.sort_values("ts"),
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta("3D")  # generous safety window
    )
    same_join = pd.merge_asof(
        intr_local.sort_values("ts"),
        same_local.sort_values("ts"),
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta("3D")
    )

    joined = prev_join.set_index("ts")[["prev_high", "prev_low", "prev_open"]].join(
        same_join.set_index("ts")[["d_open"]],
        how="left"
    )

    # Convert back to UTC and align index to original df (which is UTC)
    joined.index = joined.index.tz_convert("UTC")
    joined = joined.reindex(df.index)

    out = df.copy()
    for col in ["prev_high", "prev_low", "prev_open", "d_open"]:
        out[col] = joined[col].astype(float)

    return out
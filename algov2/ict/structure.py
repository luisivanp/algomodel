# algov2/ict/structure.py
from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def _swing_bool(series: pd.Series, left: int, right: int, mode: str) -> pd.Series:
    """
    mode = 'high' or 'low'
    A swing high at t if value[t] > value[t-k] for k=1..left and value[t] >= value[t+k] for k=1..right
    (strict on the left, non-strict on the right to reduce ties).
    """
    s = series
    n = len(s)
    if n == 0:
        return pd.Series([], dtype=bool, index=s.index)

    is_swing = pd.Series(True, index=s.index)

    # strict vs neighbors to the left
    for k in range(1, left + 1):
        if mode == "high":
            is_swing &= s > s.shift(k)
        else:
            is_swing &= s < s.shift(k)

    # non-strict vs neighbors to the right
    for k in range(1, right + 1):
        if mode == "high":
            is_swing &= s >= s.shift(-k)
        else:
            is_swing &= s <= s.shift(-k)

    # Bars at edges cannot be swing points
    is_swing.iloc[:left] = False
    is_swing.iloc[-right:] = False
    is_swing = is_swing & s.notna()
    return is_swing


def swing_points(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """
    Adds boolean columns:
      'swing_high', 'swing_low'
    Uses high/low columns; if not present, tries close.
    """
    out = df.copy()
    high = out.get("high", out.get("close"))
    low = out.get("low", out.get("close"))
    if high is None or low is None:
        out["swing_high"] = False
        out["swing_low"] = False
        return out

    out["swing_high"] = _swing_bool(high, left, right, "high")
    out["swing_low"] = _swing_bool(low, left, right, "low")
    return out


def label_intermediate(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> pd.DataFrame:
    """
    Labels intermediate-term highs/lows (ITH/ITL) as swing points.
    (You can widen left/right to be stricter.)
    """
    out = swing_points(df, left=left, right=right).copy()
    out["ITH"] = out["swing_high"]
    out["ITL"] = out["swing_low"]
    return out


def mark_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marks simple break-of-structure against the most recent swing levels:
      'bos_up' = current high breaks last swing_high price
      'bos_dn' = current low breaks last swing_low price
    """
    out = df.copy()
    high = out.get("high", out.get("close"))
    low = out.get("low", out.get("close"))

    if "swing_high" not in out or "swing_low" not in out:
        out = swing_points(out)

    # build latest swing levels as we progress
    last_sh = np.where(out["swing_high"], high, np.nan)
    last_sl = np.where(out["swing_low"], low, np.nan)
    last_sh = pd.Series(last_sh, index=out.index).ffill()
    last_sl = pd.Series(last_sl, index=out.index).ffill()

    prev_sh = last_sh.shift(1)
    prev_sl = last_sl.shift(1)

    out["bos_up"] = (high > prev_sh) & prev_sh.notna()
    out["bos_dn"] = (low < prev_sl) & prev_sl.notna()
    return out
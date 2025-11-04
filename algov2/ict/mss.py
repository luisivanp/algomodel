# algov2/ict/mss.py
from __future__ import annotations

"""
Market Structure Shift (MSS / BOS) detection for ICT model.

Outputs:
    - mss_long  (bool): bullish market-structure shift on this bar
    - mss_short (bool): bearish market-structure shift on this bar
    - mss_dir   (int):  +1 (bull), -1 (bear), 0 (none)
    - swing_high, swing_low (bool): swing markers used internally (also exported)
    - disp_ok   (bool): displacement filter satisfied on this bar
    - disp_body_ratio (float): |body| / avg_body

Config used (cfg.mss):
    - swing_left: int
    - swing_right: int
    - require_displacement: bool
    - displacement_factor: float
"""

import numpy as np
import pandas as pd

__all__ = ["detect_mss"]


def _swing_points(df: pd.DataFrame, left: int, right: int) -> tuple[pd.Series, pd.Series]:
    """
    Vectorized swing high/low via centered rolling window.
    swing_high[i] == True if high[i] is the max over [i-left, i+right]
    swing_low[i]  == True if low[i]  is the min over [i-left, i+right]
    """
    win = int(left) + int(right) + 1
    if win < 3:
        win = 3

    roll_max = df["high"].rolling(window=win, center=True).max()
    roll_min = df["low"].rolling(window=win, center=True).min()

    swing_high = (df["high"] == roll_max) & roll_max.notna()
    swing_low = (df["low"] == roll_min) & roll_min.notna()

    # Replace edge NaNs (from centered window) with False
    swing_high = swing_high.fillna(False)
    swing_low = swing_low.fillna(False)
    return swing_high, swing_low


def _displacement_filter(df: pd.DataFrame, factor: float, lookback: int = 20) -> tuple[pd.Series, pd.Series]:
    """
    Displacement check: large-bodied impulse relative to recent average body.
    Returns:
        disp_ok (bool)
        disp_body_ratio (float)
    """
    body = (df["close"] - df["open"]).abs()
    avg_body = body.rolling(lookback, min_periods=max(5, lookback // 2)).mean()
    ratio = body / (avg_body.replace(0.0, np.nan))
    disp_ok = ratio > float(factor)
    disp_ok = disp_ok.fillna(False)
    return disp_ok, ratio.fillna(0.0)


def detect_mss(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Detect bullish/bearish market-structure shifts.
    Conditions:
        - Bull MSS: close crosses above the most recent swing high (BOS up).
        - Bear MSS: close crosses below the most recent swing low  (BOS down).
        - Optional displacement filter on the same bar (cfg.mss.require_displacement).
    """
    if df.empty:
        return df

    out = df.copy()

    # --- swings
    L = int(getattr(cfg.mss, "swing_left", 2))
    R = int(getattr(cfg.mss, "swing_right", 2))
    swing_high, swing_low = _swing_points(out, L, R)
    out["swing_high"] = swing_high
    out["swing_low"] = swing_low

    # last swing levels forward-filled
    last_swh = out["high"].where(swing_high).ffill()
    last_swl = out["low"].where(swing_low).ffill()

    # --- BOS / MSS
    above = out["close"] > last_swh
    below = out["close"] < last_swl
    above_prev = above.shift(1).fillna(False)
    below_prev = below.shift(1).fillna(False)

    bos_up = above & (~above_prev)
    bos_dn = below & (~below_prev)

    # --- displacement
    disp_required = bool(getattr(cfg.mss, "require_displacement", True))
    factor = float(getattr(cfg.mss, "displacement_factor", 1.2))
    disp_ok, disp_ratio = _displacement_filter(out, factor=factor, lookback=20)
    out["disp_ok"] = disp_ok
    out["disp_body_ratio"] = disp_ratio

    if disp_required:
        mss_long = bos_up & disp_ok
        mss_short = bos_dn & disp_ok
    else:
        mss_long = bos_up
        mss_short = bos_dn

    out["mss_long"] = mss_long
    out["mss_short"] = mss_short
    out["mss_dir"] = np.where(mss_long, 1, np.where(mss_short, -1, 0)).astype(int)

    # Make sure outputs are clean types (no pd.NA)
    for col in ["mss_long", "mss_short", "swing_high", "swing_low", "disp_ok"]:
        out[col] = out[col].fillna(False).astype(bool)

    return out
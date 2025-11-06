# algov2/ict/mss.py
from __future__ import annotations

import pandas as pd
import numpy as np

def _swings(high: pd.Series, low: pd.Series, left: int, right: int):
    """Return boolean Series for swing highs/lows."""
    sh = (high.shift(0).rolling(left+right+1, center=True)
          .apply(lambda w: w[left] == max(w), raw=True)).astype(bool)
    sl = (low.shift(0).rolling(left+right+1, center=True)
          .apply(lambda w: w[left] == min(w), raw=True)).astype(bool)
    # rolling with center requires enough rows; fill NaN False
    sh = sh.fillna(False)
    sl = sl.fillna(False)
    return sh, sl

def detect_mss(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    require_displacement: bool = True,
    disp_factor: float = 1.2,
    **kwargs,
) -> pd.DataFrame:
    """
    Market Structure Shift (simple):
      - MSS up when a bar CLOSE crosses above the last swing high
      - MSS down when CLOSE crosses below the last swing low
    Optionally require displacement relative to median body size.
    Adds columns: mss_up, mss_down (bool)
    """
    if df.empty:
        return df

    high = df["high"]; low = df["low"]; close = df["close"]; open_ = df["open"]

    sh, sl = _swings(high, low, left, right)

    last_sh = high.where(sh).ffill()
    last_sl = low.where(sl).ffill()

    body = (close - open_).abs()
    body_med = body.rolling(100, min_periods=20).median().fillna(method="bfill")
    disp_ok_up = (close - open_) >= (disp_factor * body_med)
    disp_ok_dn = (open_ - close) >= (disp_factor * body_med)

    mss_up = close > last_sh
    mss_dn = close < last_sl

    if require_displacement:
        mss_up = mss_up & disp_ok_up
        mss_dn = mss_dn & disp_ok_dn

    df["mss_up"] = mss_up.fillna(False)
    df["mss_down"] = mss_dn.fillna(False)
    return df
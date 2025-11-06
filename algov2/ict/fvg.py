# algov2/ict/fvg.py
from __future__ import annotations

import pandas as pd
import numpy as np

def scan_fvg_zones(
    df: pd.DataFrame,
    lookback: int = 100,
    allow_ifvg: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Detect classic 3-bar Fair Value Gaps:
      - Bullish FVG at bar i if low[i] > high[i-2]
      - Bearish FVG at bar i if high[i] < low[i-2]
    Adds:
      - fvg_bull, fvg_bear (event bools)
      - fvg_top, fvg_bot (current active zone bounds, forward-filled up to lookback)
    If allow_ifvg=True, keep zone active until 'inversion' fill is seen.
    """
    if df.empty:
        return df

    h = df["high"]; l = df["low"]

    # events
    bull_evt = (l > h.shift(2))
    bear_evt = (h < l.shift(2))

    top = pd.Series(index=df.index, dtype=float)
    bot = pd.Series(index=df.index, dtype=float)

    # on event, set zone
    top[bull_evt] = l[bull_evt]   # bullish zone: (high[i-2], low[i])
    bot[bull_evt] = h.shift(2)[bull_evt]

    top[bear_evt] = h.shift(2)[bear_evt]  # bearish zone: (high[i], low[i-2])
    bot[bear_evt] = l.shift(2)[bear_evt]

    # forward fill limited by lookback
    top = top.ffill(limit=lookback)
    bot = bot.ffill(limit=lookback)

    # optional basic invalidation: if price fully closes the gap, drop it
    closed_bull = df["low"] <= bot  # filled back into old high
    closed_bear = df["high"] >= top
    if not allow_ifvg:
        # when closed, clear zone
        top = top.mask(closed_bull | closed_bear)
        bot = bot.mask(closed_bull | closed_bear)

    df["fvg_bull"] = bull_evt.fillna(False)
    df["fvg_bear"] = bear_evt.fillna(False)
    df["fvg_top"] = top
    df["fvg_bot"] = bot
    return df
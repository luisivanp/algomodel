# algov2/ict/cooldown.py
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def apply_cooldown(
    signals: pd.DataFrame,
    bars_cooldown: int = 3,
    separate_by_direction: bool = True,
) -> pd.DataFrame:
    """
    Suppress new entries for N bars after a signal fires.
    If separate_by_direction, maintain independent cooldowns for long & short.

    Expected columns: 'entry','sig_long','sig_short','dir'
    Returns a new DataFrame.
    """
    sig = signals.copy()
    n = len(sig)
    if n == 0 or bars_cooldown <= 0:
        return sig

    if separate_by_direction:
        # long track
        cool = 0
        mask_allow = np.ones(n, dtype=bool)
        for i in range(n):
            if cool > 0:
                # block new long
                if sig.iloc[i]["sig_long"] > 0:
                    sig.iloc[i, sig.columns.get_loc("sig_long")] = 0.0
                cool -= 1
            if sig.iloc[i]["sig_long"] > 0:
                cool = bars_cooldown

        # short track
        cool = 0
        for i in range(n):
            if cool > 0:
                # block new short
                if sig.iloc[i]["sig_short"] > 0:
                    sig.iloc[i, sig.columns.get_loc("sig_short")] = 0.0
                cool -= 1
            if sig.iloc[i]["sig_short"] > 0:
                cool = bars_cooldown
    else:
        cool = 0
        for i in range(n):
            if cool > 0:
                # block both directions
                if sig.iloc[i]["sig_long"] > 0 or sig.iloc[i]["sig_short"] > 0:
                    sig.iloc[i, sig.columns.get_loc("sig_long")] = 0.0
                    sig.iloc[i, sig.columns.get_loc("sig_short")] = 0.0
                cool -= 1
            if sig.iloc[i]["sig_long"] > 0 or sig.iloc[i]["sig_short"] > 0:
                cool = bars_cooldown

    # recompute entry/dir
    sig["dir"] = np.where(sig["sig_long"] > 0, 1.0, np.where(sig["sig_short"] > 0, -1.0, 0.0)).astype(float)
    sig["entry"] = ((sig["sig_long"] > 0) | (sig["sig_short"] > 0)).astype(float)
    return sig
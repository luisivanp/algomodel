# algov2/ict/orderblocks.py
from __future__ import annotations
import pandas as pd
import numpy as np

def _ensure_cols(df: pd.DataFrame):
    need = {"open","high","low","close"}
    if not need.issubset(df.columns):
        raise ValueError(f"orderblocks.py: df missing {need - set(df.columns)}")

def _last_opposite_close_candle(df: pd.DataFrame, idx: int, up_move: bool, max_scan: int = 20):
    """
    For a displacement move (up_move True means displacement up),
    find the most recent opposite close candle index before idx.
    """
    start = max(0, idx - max_scan)
    sl = df.iloc[start:idx]
    if up_move:
        # bullish move: need last down-close candle
        mask = sl["close"] < sl["open"]
    else:
        # bearish move: need last up-close candle
        mask = sl["close"] > sl["open"]
    if not mask.any():
        return None
    return sl[mask].index[-1]  # last one before idx

def scan_orderblocks(
    df: pd.DataFrame,
    use_displacement_cols: bool = True,
    max_scan_back: int = 20,
    zone_mode: str = "wick_to_open"  # "body" | "wick_to_open"
) -> pd.DataFrame:
    """
    Detects simple ICT-style order blocks keyed off displacement bars.
    If use_displacement_cols=True, expects df['disp_up'] / df['disp_dn'] (ints).
    Otherwise, it will approximate displacement using large bodies vs rolling mean.
    Adds columns:
      ob_bull_low, ob_bull_high, ob_bear_low, ob_bear_high (forward-filled active zones)
    Also returns a DataFrame with per-bar zones (sparse).
    """
    _ensure_cols(df)
    out = df.copy()

    # If no displacement columns, synthesize a crude version
    if use_displacement_cols is False or "disp_up" not in out.columns or "disp_dn" not in out.columns:
        body = (out["close"] - out["open"]).abs()
        avg  = body.rolling(20, min_periods=10).mean()
        big  = body > 1.6 * avg.fillna(method="bfill").fillna(0)
        disp_up = ((out["close"] > out["open"]) & big).astype(int)
        disp_dn = ((out["close"] < out["open"]) & big).astype(int)
    else:
        disp_up = out["disp_up"].astype(int)
        disp_dn = out["disp_dn"].astype(int)

    # Sparse zone frames
    bull_low  = pd.Series(np.nan, index=out.index)
    bull_high = pd.Series(np.nan, index=out.index)
    bear_low  = pd.Series(np.nan, index=out.index)
    bear_high = pd.Series(np.nan, index=out.index)

    for i, (t, row) in enumerate(out.iterrows()):
        if disp_up.iat[i] == 1:
            anchor = _last_opposite_close_candle(out, i, up_move=True, max_scan=max_scan_back)
            if anchor is not None:
                c = out.loc[anchor]
                if zone_mode == "body":
                    low, high = min(c["open"], c["close"]), max(c["open"], c["close"])
                else:  # wick_to_open (common ICT mapping)
                    low, high = c["low"], c["open"]
                bull_low.iat[i] = low
                bull_high.iat[i] = high

        if disp_dn.iat[i] == 1:
            anchor = _last_opposite_close_candle(out, i, up_move=False, max_scan=max_scan_back)
            if anchor is not None:
                c = out.loc[anchor]
                if zone_mode == "body":
                    low, high = min(c["open"], c["close"]), max(c["open"], c["close"])
                else:  # wick_to_open
                    low, high = c["open"], c["high"]
                bear_low.iat[i] = low
                bear_high.iat[i] = high

    # Forward-fill “active” zones until they’re invalidated (optional logic)
    # For simplicity we just carry last known zone. Your pipeline can add invalidation rules.
    out["ob_bull_low"]  = bull_low.ffill()
    out["ob_bull_high"] = bull_high.ffill()
    out["ob_bear_low"]  = bear_low.ffill()
    out["ob_bear_high"] = bear_high.ffill()
    return out
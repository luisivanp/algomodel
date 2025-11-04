# algov2/ict/fvg.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple

__all__ = ["scan_fvg_zones", "build_fvg_zones"]


# ---- Optional params container (duck-typed with your cfg.fvg) ----
@dataclass
class _FVGParams:
    lookback_bars: int = 100
    min_size_points: float = 1.0
    allow_ifvg: bool = True
    prefer_htf_parent: bool = True
    htf_frame: str = "15m"


def _ensure_numeric(df: pd.DataFrame) -> None:
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"fvg: missing price column '{c}'")
        df[c] = pd.to_numeric(df[c], errors="coerce")


def _detect_fvgs_core(
    df: pd.DataFrame,
    min_size: float,
    lookback_bars: int,
) -> pd.DataFrame:
    """
    Classic 3-candle ICT FVG detection:
      - Bullish FVG at candle i if high[i-2] < low[i]  and (low[i] - high[i-2]) >= min_size
      - Bearish FVG at candle i if low[i-2]  > high[i] and (low[i-2] - high[i]) >= min_size
    We mark them at the *third* candle (i), with bounds:
      - Bullish gap: [high[i-2], low[i]]
      - Bearish gap: [high[i],  low[i-2]]
    """
    out = df.copy()

    # base columns (no NaNs to avoid later issues)
    out["fvg_dir"] = 0  # +1 bullish, -1 bearish, 0 none
    out["fvg_low"] = np.nan
    out["fvg_high"] = np.nan
    out["fvg_id"] = -1

    H = out["high"]
    L = out["low"]

    # Shifted refs
    H_m2 = H.shift(2)
    L_m2 = L.shift(2)

    # Bullish condition
    bull_gap = (H_m2 < L) & ((L - H_m2) >= min_size)
    # Bearish condition
    bear_gap = (L_m2 > H) & ((L_m2 - H) >= min_size)

    # Assign bounds at creation index (i)
    bull_idx = np.where(bull_gap.values)[0]
    bear_idx = np.where(bear_gap.values)[0]

    # ID counter
    fvg_id = 0

    # fill bullish
    for i in bull_idx:
        lo = float(H_m2.iat[i])
        hi = float(L.iat[i])
        out.at[out.index[i], "fvg_dir"] = 1
        out.at[out.index[i], "fvg_low"] = lo
        out.at[out.index[i], "fvg_high"] = hi
        out.at[out.index[i], "fvg_id"] = fvg_id
        fvg_id += 1

    # fill bearish
    for i in bear_idx:
        lo = float(H.iat[i])
        hi = float(L_m2.iat[i])
        out.at[out.index[i], "fvg_dir"] = -1
        out.at[out.index[i], "fvg_low"] = lo
        out.at[out.index[i], "fvg_high"] = hi
        out.at[out.index[i], "fvg_id"] = fvg_id
        fvg_id += 1

    # Trim to lookback window if requested (keep only last N bars with new FVGs; older rows keep NaNs)
    if lookback_bars is not None and lookback_bars > 0 and len(out) > lookback_bars:
        # nothing to drop from df; lookback is mainly for scanning speed.
        pass

    # Convenience booleans
    out["has_fvg"] = out["fvg_dir"] != 0

    # Ensure numeric dtype (Backtrader hates NAType in lines—keep NaNs as float)
    out["fvg_low"] = pd.to_numeric(out["fvg_low"], errors="coerce")
    out["fvg_high"] = pd.to_numeric(out["fvg_high"], errors="coerce")

    return out


def _compute_ifvg_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very lightweight iFVG heuristic:
      - After an FVG is created, if price later trades *through* the gap and closes beyond
        the far boundary, that gap is considered 'inverted' on that bar.
      - We mark 'ifvg_dir' = -fvg_dir at the first bar that fully closes beyond.
    This is intentionally simple (good enough for filters).
    """
    out = df.copy()

    out["ifvg_dir"] = 0
    out["ifvg_low"] = np.nan
    out["ifvg_high"] = np.nan
    out["has_ifvg"] = False

    # Iterate efficiently over indices where there was an FVG defined
    fvg_rows = np.where(out["has_fvg"].values)[0]
    if len(fvg_rows) == 0:
        return out

    close = out["close"].values
    fvg_dir = out["fvg_dir"].values
    fvg_lo = out["fvg_low"].values
    fvg_hi = out["fvg_high"].values

    # track per FVG whether we've already inverted it
    inverted_done = set()

    for i in fvg_rows:
        fid = int(out["fvg_id"].iat[i])
        if fid in inverted_done:
            continue

        # scan forward until gap is fully closed beyond opposite boundary
        # bullish fvg (dir=+1): invert when close < fvg_low
        # bearish fvg (dir=-1): invert when close > fvg_high
        if fvg_dir[i] == 1:
            # look ahead
            for j in range(i + 1, len(out)):
                if close[j] < fvg_lo[i]:
                    out.at[out.index[j], "ifvg_dir"] = -1  # inverted acts as resistance
                    out.at[out.index[j], "ifvg_low"] = fvg_lo[i]
                    out.at[out.index[j], "ifvg_high"] = fvg_hi[i]
                    out.at[out.index[j], "has_ifvg"] = True
                    inverted_done.add(fid)
                    break
        elif fvg_dir[i] == -1:
            for j in range(i + 1, len(out)):
                if close[j] > fvg_hi[i]:
                    out.at[out.index[j], "ifvg_dir"] = +1  # inverted acts as support
                    out.at[out.index[j], "ifvg_low"] = fvg_lo[i]
                    out.at[out.index[j], "ifvg_high"] = fvg_hi[i]
                    out.at[out.index[j], "has_ifvg"] = True
                    inverted_done.add(fid)
                    break

    return out


def build_fvg_zones(df: pd.DataFrame, fvg_cfg: Optional[object] = None) -> pd.DataFrame:
    """
    Core builder. Returns a copy of df with these columns:
      - fvg_dir   : +1 bullish, -1 bearish, 0 none
      - fvg_low   : float lower bound of the gap (nan if none)
      - fvg_high  : float upper bound of the gap (nan if none)
      - fvg_id    : integer id per gap (=-1 if none)
      - has_fvg   : bool
      - ifvg_dir  : +1/-1/0 (0 if none)
      - ifvg_low  : float (nan if none)
      - ifvg_high : float (nan if none)
      - has_ifvg  : bool
    This is intentionally simple and fast.
    """
    # Duck-typed params
    if fvg_cfg is None:
        params = _FVGParams()
    else:
        params = _FVGParams(
            lookback_bars=getattr(fvg_cfg, "lookback_bars", 100),
            min_size_points=getattr(fvg_cfg, "min_size_points", 1.0),
            allow_ifvg=getattr(fvg_cfg, "allow_ifvg", True),
            prefer_htf_parent=getattr(fvg_cfg, "prefer_htf_parent", True),
            htf_frame=getattr(fvg_cfg, "htf_frame", "15m"),
        )

    _ensure_numeric(df)

    base = _detect_fvgs_core(df, params.min_size_points, params.lookback_bars)

    if params.allow_ifvg:
        base = _compute_ifvg_flags(base)
    else:
        base["ifvg_dir"] = 0
        base["ifvg_low"] = np.nan
        base["ifvg_high"] = np.nan
        base["has_ifvg"] = False

    # For safety: no NATypes in bool/int columns
    base["fvg_dir"] = base["fvg_dir"].fillna(0).astype(int)
    base["fvg_id"] = base["fvg_id"].fillna(-1).astype(int)
    base["has_fvg"] = base["has_fvg"].fillna(False).astype(bool)
    base["ifvg_dir"] = base["ifvg_dir"].fillna(0).astype(int)
    base["has_ifvg"] = base["has_ifvg"].fillna(False).astype(bool)

    return base


def scan_fvg_zones(df: pd.DataFrame, fvg_cfg: Optional[object] = None) -> pd.DataFrame:
    """
    Thin wrapper kept for pipeline compatibility.
    """
    return build_fvg_zones(df, fvg_cfg)
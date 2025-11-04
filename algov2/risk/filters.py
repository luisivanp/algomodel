# algov2/ict/filters.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class FiltersParams:
    allow_sessions: Tuple[str, ...] = ("LONDON", "NYAM", "NYPM")
    require_bias_align: bool = True
    premium_discount_col: str = "pd_zone"     # expected values: {"premium","discount","mid"}
    enforce_pd_match: bool = False            # LONG in discount / SHORT in premium
    one_signal_per_bar: bool = True


def _normalize_params(cfg_like) -> FiltersParams:
    if isinstance(cfg_like, FiltersParams):
        return cfg_like
    if isinstance(cfg_like, dict):
        return FiltersParams(
            allow_sessions=tuple(cfg_like.get("allow_sessions", ("LONDON", "NYAM", "NYPM"))),
            require_bias_align=bool(cfg_like.get("require_bias_align", True)),
            premium_discount_col=str(cfg_like.get("premium_discount_col", "pd_zone")),
            enforce_pd_match=bool(cfg_like.get("enforce_pd_match", False)),
            one_signal_per_bar=bool(cfg_like.get("one_signal_per_bar", True)),
        )
    # best effort
    return FiltersParams()


def _session_mask(df: pd.DataFrame, allow: Iterable[str]) -> pd.Series:
    allow_set = set(s.upper() for s in allow)
    if "session" not in df.columns:
        # If sessions not marked, allow all to avoid accidental suppression
        return pd.Series(True, index=df.index)
    return df["session"].astype(str).str.upper().isin(allow_set)


def _bias_align_mask(df: pd.DataFrame) -> pd.Series:
    """
    If df has 'bias' (1 bullish, -1 bearish) and direction columns, use that.
    Otherwise return True.
    """
    if "bias" not in df.columns:
        return pd.Series(True, index=df.index)

    # Direction inferred from candidate signal columns when we later enforce
    # (here we just say bias info exists and is valid)
    return pd.Series(True, index=df.index)


def _pd_match_mask(df: pd.DataFrame, col: str, want_long_in_discount=True) -> pd.Series:
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    z = df[col].astype(str).str.lower()
    # True = PD info available and valid; the per-direction enforcement happens later
    return z.isin(["premium", "discount", "mid"])


def compute_filter_mask(df: pd.DataFrame, params: FiltersParams) -> pd.Series:
    """
    Returns a base boolean mask (per-row allowed) *independent of signal direction*.
    Direction-specific checks (bias align & premium/discount match) are applied in enforce_filters().
    """
    P = _normalize_params(params)
    m_session = _session_mask(df, P.allow_sessions)
    m_biasinfo = _bias_align_mask(df)
    m_pdinfo = _pd_match_mask(df, P.premium_discount_col)
    return (m_session & m_biasinfo & m_pdinfo).astype(bool)


def enforce_filters(
    signals: pd.DataFrame,
    df: pd.DataFrame,
    params: FiltersParams,
) -> pd.DataFrame:
    """
    Applies session/bias/premium-discount and one-per-bar gating to a signal frame.

    Expected signal columns: ['entry','dir','sig_long','sig_short','sl','tp']
    df may contain: 'session','bias' (+1/-1), 'pd_zone' ('premium'/'discount'/'mid')
    """
    P = _normalize_params(params)
    sig = signals.copy()

    # Ensure required columns exist
    for c in ["entry", "dir", "sig_long", "sig_short", "sl", "tp"]:
        if c not in sig.columns:
            sig[c] = 0.0 if c not in ("sl", "tp") else np.nan

    base_mask = compute_filter_mask(df, P)

    # Start by zeroing entries not in allowed base mask
    sig.loc[~base_mask, ["entry", "sig_long", "sig_short", "dir"]] = 0.0

    # Bias alignment (only if present and required)
    if P.require_bias_align and "bias" in df.columns:
        bias = df["bias"].fillna(0).astype(int)
        # allow LONG only if bias >= 1, SHORT only if bias <= -1
        long_ok = bias >= 1
        short_ok = bias <= -1

        # kill longs when not allowed
        kill_long = (sig["sig_long"] > 0) & (~long_ok)
        sig.loc[kill_long, ["entry", "sig_long"]] = 0.0

        # kill shorts when not allowed
        kill_short = (sig["sig_short"] > 0) & (~short_ok)
        sig.loc[kill_short, ["entry", "sig_short"]] = 0.0

    # Premium/Discount matching (if requested and pd_zone present)
    if P.enforce_pd_match and P.premium_discount_col in df.columns:
        zone = df[P.premium_discount_col].astype(str).str.lower()

        # LONG only in discount; SHORT only in premium
        kill_long = (sig["sig_long"] > 0) & (~zone.eq("discount"))
        kill_short = (sig["sig_short"] > 0) & (~zone.eq("premium"))

        sig.loc[kill_long, ["entry", "sig_long"]] = 0.0
        sig.loc[kill_short, ["entry", "sig_short"]] = 0.0

    # Recompute dir after gating
    sig["dir"] = np.where(sig["sig_long"] > 0, 1.0, np.where(sig["sig_short"] > 0, -1.0, 0.0)).astype(float)
    sig["entry"] = ((sig["sig_long"] > 0) | (sig["sig_short"] > 0)).astype(float)

    # One-signal-per-bar: ensure only one direction is active; prefer 'dir' already set
    if P.one_signal_per_bar:
        both = (sig["sig_long"] > 0) & (sig["sig_short"] > 0)
        # default preference: keep 'dir' if set, otherwise keep long
        keep_long = (sig["dir"] >= 0) | (sig["dir"].abs() == 0)
        sig.loc[both & keep_long, "sig_short"] = 0.0
        sig.loc[both & (~keep_long), "sig_long"] = 0.0
        sig["dir"] = np.where(sig["sig_long"] > 0, 1.0, np.where(sig["sig_short"] > 0, -1.0, 0.0)).astype(float)
        sig["entry"] = ((sig["sig_long"] > 0) | (sig["sig_short"] > 0)).astype(float)

    return sig
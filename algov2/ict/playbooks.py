# algov2/ict/playbooks.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

__all__ = ["PlaybookConfig", "scan_open_playbook"]


# Minimal config used for typing / defaults in this module.
# (You may also have a similarly named dataclass in algov2.config; that’s fine—
# they live in different modules. This one just prevents import errors.)
@dataclass
class PlaybookConfig:
    min_confs: int = 2               # minimum confluences required
    use_ma9_filter: bool = True      # require price vs MA alignment
    min_rr: float = 1.5              # target R:R for TP projection
    prefer_tp: str = "best"          # "tp1" | "tp2" | "best"
    partials: bool = True            # allow partial exits at TP1 then trail/TP2


def _ensure_col(df: pd.DataFrame, name: str, default):
    if name not in df.columns:
        if isinstance(default, bool):
            df[name] = False
        else:
            df[name] = default
    return df


def scan_open_playbook(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Build open-model entries from ICT confluences:
      session_active ∧ MSS ∧ (FVG or iFVG) ∧ (optional MA filter) ∧ (optional bias align)

    Outputs/updates:
      - sig_long, sig_short  (bool)
      - dir                  ("LONG"/"SHORT" where a signal prints)
      - entry, sl, tp        (floats; NaN when no signal on bar)
    """
    if df.empty:
        base = df.copy()
        for c in ["sig_long", "sig_short", "dir", "entry", "sl", "tp"]:
            base[c] = np.nan if c in ("entry", "sl", "tp") else False
        base["dir"] = base["dir"].astype("object")
        return base

    out = df.copy()

    # --- config knobs with safe fallbacks
    rr = float(getattr(getattr(cfg, "playbook", object), "min_rr", 1.5))
    use_ma9 = bool(getattr(getattr(cfg, "playbook", object), "use_ma9_filter", True))
    align_bias = bool(getattr(getattr(cfg, "bias", object), "require_align_for_trades", True))
    allow_ifvg = bool(getattr(getattr(cfg, "fvg", object), "allow_ifvg", True))
    min_stop = float(getattr(getattr(cfg, "risk", object), "min_stop_points", 1.0))

    # --- ensure presence of the columns we rely on (graceful defaults)
    for bcol in [
        "session_active", "mss_long", "mss_short",
        "fvg_bull", "fvg_bear", "ifvg_bull", "ifvg_bear"
    ]:
        out = _ensure_col(out, bcol, False)

    if "bias_dir" not in out.columns:
        out["bias_dir"] = None

    for ncol in ["fvg_bull_lo", "fvg_bull_hi", "fvg_bull_mt", "fvg_bear_lo", "fvg_bear_hi", "fvg_bear_mt"]:
        out = _ensure_col(out, ncol, np.nan)

    # Optional MA filter
    has_ma = False
    if "ema9" in out.columns:
        has_ma = True
        ma = out["ema9"]
    elif "ma9" in out.columns:
        has_ma = True
        ma = out["ma9"]

    # --- base masks
    session_ok = out["session_active"].astype(bool)

    fvg_long_ok = out["fvg_bull"].astype(bool) | (allow_ifvg and out["ifvg_bull"].astype(bool))
    fvg_short_ok = out["fvg_bear"].astype(bool) | (allow_ifvg and out["ifvg_bear"].astype(bool))

    mss_long = out["mss_long"].astype(bool)
    mss_short = out["mss_short"].astype(bool)

    long_mask = session_ok & fvg_long_ok & mss_long
    short_mask = session_ok & fvg_short_ok & mss_short

    if align_bias:
        long_mask &= (out["bias_dir"] == "LONG")
        short_mask &= (out["bias_dir"] == "SHORT")

    if use_ma9 and has_ma:
        long_mask &= (out["close"] >= ma)
        short_mask &= (out["close"] <= ma)

    # --- outputs
    out["sig_long"] = False
    out["sig_short"] = False
    out["dir"] = None
    out["entry"] = np.nan
    out["sl"] = np.nan
    out["tp"] = np.nan

    # Long geometry
    entry_long = np.where(
        long_mask,
        np.where(~out["fvg_bull_mt"].isna(), out["fvg_bull_mt"], out["close"]),
        np.nan,
    ).astype(float)

    sl_long = np.where(
        long_mask,
        np.where(~out["fvg_bull_lo"].isna(), out["fvg_bull_lo"], out["low"]),
        np.nan,
    ).astype(float)

    sl_long = np.where(long_mask & ((entry_long - sl_long) < min_stop), entry_long - min_stop, sl_long)
    tp_long = np.where(long_mask, entry_long + rr * (entry_long - sl_long), np.nan)

    # Short geometry
    entry_short = np.where(
        short_mask,
        np.where(~out["fvg_bear_mt"].isna(), out["fvg_bear_mt"], out["close"]),
        np.nan,
    ).astype(float)

    sl_short = np.where(
        short_mask,
        np.where(~out["fvg_bear_hi"].isna(), out["fvg_bear_hi"], out["high"]),
        np.nan,
    ).astype(float)

    sl_short = np.where(short_mask & ((sl_short - entry_short) < min_stop), entry_short + min_stop, sl_short)
    tp_short = np.where(short_mask, entry_short - rr * (sl_short - entry_short), np.nan)

    # write back
    out.loc[long_mask, "sig_long"] = True
    out.loc[short_mask, "sig_short"] = True
    out.loc[long_mask, "dir"] = "LONG"
    out.loc[short_mask, "dir"] = "SHORT"

    out["entry"] = np.where(~np.isnan(entry_long), entry_long, out["entry"])
    out["entry"] = np.where(~np.isnan(entry_short), entry_short, out["entry"])

    out["sl"] = np.where(~np.isnan(sl_long), sl_long, out["sl"])
    out["sl"] = np.where(~np.isnan(sl_short), sl_short, out["sl"])

    out["tp"] = np.where(~np.isnan(tp_long), tp_long, out["tp"])
    out["tp"] = np.where(~np.isnan(tp_short), tp_short, out["tp"])

    out["sig_long"] = out["sig_long"].astype(bool)
    out["sig_short"] = out["sig_short"].astype(bool)
    out["dir"] = out["dir"].astype("object")

    return out
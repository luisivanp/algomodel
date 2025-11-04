# algov2/exec/signals.py
from __future__ import annotations

import numpy as np
import pandas as pd
from .types import LONG, SHORT

REASON_COL = "signals"

def add_reason(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    out = df.copy()
    if REASON_COL not in out.columns:
        out[REASON_COL] = ""
    r = out[REASON_COL].fillna("").astype(str)
    r = np.where(r == "", reason, r + "," + reason)
    out[REASON_COL] = r
    return out

def attach_types_and_reasons(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "type" not in out.columns:
        out["type"] = ""
    # derive dir if missing
    if "dir" not in out.columns:
        out["dir"] = 0
    if "sig_long" in out.columns:
        out["dir"] = np.where(out["sig_long"].fillna(0).astype(int) > 0, 1, out["dir"])
    if "sig_short" in out.columns:
        out["dir"] = np.where(out["sig_short"].fillna(0).astype(int) > 0, -1, out["dir"])
    # default type label
    out["type"] = np.where(out.get("sig_long", 0).astype(int) > 0, "LONG",
                   np.where(out.get("sig_short", 0).astype(int) > 0, "SHORT", out["type"]))
    # ensure reason col exists
    if REASON_COL not in out.columns:
        out[REASON_COL] = ""
    return out

def pulse_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn always-on flags into pulses (1 on the first bar of the signal).
    Keeps intraday timestamps exactly as-is.
    """
    out = df.copy()
    for c in ("sig_long", "sig_short", "entry"):
        if c not in out.columns:
            out[c] = 0
        out[c] = out[c].fillna(0).astype(int)

    # pulse on the transition 0->1
    e = out["entry"].astype(int)
    pulse = (e.eq(1) & e.shift(1, fill_value=0).ne(1)).astype(int)
    out["entry"] = pulse

    # mask dir with pulses too (only keep dir on the pulse bar)
    if "dir" in out.columns:
        out["dir"] = np.where(out["entry"] == 1, out["dir"], 0)

    # sl/tp only matter on the pulse row; clear others
    if "sl" in out.columns:
        out["sl"] = np.where(out["entry"] == 1, out["sl"], np.nan)
    if "tp" in out.columns:
        out["tp"] = np.where(out["entry"] == 1, out["tp"], np.nan)

    return out

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["entry", "dir", "sig_long", "sig_short", "sl", "tp", "type", REASON_COL]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0 if c in ("entry", "dir", "sig_long", "sig_short") else ("" if c in ("type", REASON_COL) else np.nan)
    return out

def finalize_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final pass:
      - ensure columns
      - attach type/reasons
      - convert to pulses so we get a single trade bar per signal
      - DO NOT normalize the index (keep intraday bar times)
    """
    out = _ensure_cols(df)
    out = attach_types_and_reasons(out)
    out = pulse_entries(out)

    # keep only bars that actually fire an entry
    out = out.loc[out["entry"] == 1, ["entry", "dir", "sig_long", "sig_short", "sl", "tp", "type", REASON_COL]]

    # guarantee DatetimeIndex if possible (no tz mangling here)
    if not isinstance(out.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        # leave as-is; strategy will align by equality on provided index
        pass

    return out.sort_index()
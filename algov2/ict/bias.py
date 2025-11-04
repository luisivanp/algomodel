# algov2/ict/bias.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Optional
import numpy as np
import pandas as pd


def _as_simple_cfg(obj: Any) -> dict:
    """Return a plain dict of useful attributes; never calls len(obj)."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    # Generic object -> pick public attrs only
    out = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        # keep only simple values
        if isinstance(v, (int, float, str, bool, tuple)):
            out[k] = v
    return out


def _daily_levels(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Attach previous day high/low and midnight open; index must be DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("bias._daily_levels: index must be DatetimeIndex")
    x = df.copy()

    # Work in provided tz for daily splits
    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    x = x.tz_convert(tz)

    # Daily OHLC
    d = pd.DataFrame({
        "d_high": x["high"].resample("1D").max(),
        "d_low":  x["low"].resample("1D").min(),
    })

    # Midnight open = first close after 00:00 local
    mid_open = x["open"].resample("1D").first().rename("mid_open")

    d = d.join(mid_open)
    d["prev_high"] = d["d_high"].shift(1)
    d["prev_low"]  = d["d_low"].shift(1)
    d["prev_mid_open"] = d["mid_open"].shift(1)
    d["midline"] = (d["d_high"] + d["d_low"]) / 2.0

    # Broadcast back to intraday
    out = x.join(d, on=x.index.normalize())
    # Return to original tz of incoming df
    out = out.tz_convert("UTC")
    out.index = out.index.tz_convert("UTC")
    return out


def compute_bias(df: pd.DataFrame, bias_cfg: Optional[Any] = None, tz: str = "US/Eastern") -> pd.DataFrame:
    """
    Attach:
      - bias_dir: +1/-1/0 using daily levels and (optionally) daily open
      - pd_zone: 'premium'/'discount'/'eq' vs daily midline
    Accepts BiasConfig, full Config, or dict-like. Never uses len(cfg).
    """
    cfgd = _as_simple_cfg(bias_cfg)
    use_daily_open = bool(cfgd.get("use_daily_open", True))

    x = df.copy()
    x = _daily_levels(x, tz=tz)

    # Basic premium/discount vs daily midline
    x["pd_zone"] = np.where(
        x["close"] > x["midline"], "premium",
        np.where(x["close"] < x["midline"], "discount", "eq")
    )

    # Bias rule:
    #  - if use_daily_open: compare to previous midnight open when available
    #  - otherwise: compare to daily midline
    ref = x["prev_mid_open"] if use_daily_open else x["midline"]
    x["bias_dir"] = np.where(
        x["close"] > ref, 1,
        np.where(x["close"] < ref, -1, 0)
    )

    # Clean up any initial NaNs (first day) -> neutral
    x["bias_dir"] = x["bias_dir"].fillna(0).astype(int)
    x["pd_zone"] = x["pd_zone"].fillna("eq")

    # Drop helper daily columns we don't need downstream
    keep = [c for c in x.columns]
    return x[keep]
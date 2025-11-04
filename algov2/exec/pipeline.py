# algov2/exec/pipeline.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Any

from ..config import Config
from .signals import finalize_signals, add_reason


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _session_mask(idx_utc: pd.DatetimeIndex, cfg: Config) -> pd.Series:
    """OR of allowed session windows in cfg.tz, returned aligned to UTC index."""
    if not isinstance(idx_utc, pd.DatetimeIndex):
        return pd.Series(False, index=idx_utc)

    idx_local = idx_utc.tz_convert(cfg.tz) if idx_utc.tz is not None else idx_utc.tz_localize(cfg.tz)
    minutes = idx_local.hour * 60 + idx_local.minute

    allow = set(cfg.sessions.allow_sessions)
    winmap = cfg.session_windows()  # {name: (HH:MM, HH:MM)}

    mask = np.zeros(len(idx_utc), dtype=bool)
    for name, (start, end) in winmap.items():
        if name not in allow:
            continue
        smin, emin = _minutes(start), _minutes(end)
        mask |= (minutes >= smin) & (minutes < emin)
    return pd.Series(mask, index=idx_utc)


def _daily_prev_hilo(df: pd.DataFrame, tz_for_day: str) -> pd.DataFrame:
    """
    Previous day's high/low aligned to every intraday bar.
    Day boundaries use tz_for_day (so your 'day' matches your trading session locale).
    """
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    # define the 'day' in the chosen timezone, then map back
    idx_local = idx.tz_convert(tz_for_day)
    day_key = idx_local.normalize()

    daily_high = df["high"].groupby(day_key).max()
    daily_low = df["low"].groupby(day_key).min()
    prev_high_by_day = daily_high.shift(1)
    prev_low_by_day = daily_low.shift(1)

    out = pd.DataFrame(index=df.index)
    out["prev_high"] = prev_high_by_day.reindex(day_key).to_numpy()
    out["prev_low"] = prev_low_by_day.reindex(day_key).to_numpy()
    return out


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _call_ict(func: Callable[[pd.DataFrame, Any], pd.DataFrame], df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Call an ICT helper with dataclass cfg; if it raises due to expecting a dict
    (e.g., tries cfg.copy()), retry with cfg.asdict().
    """
    try:
        return func(df, cfg)
    except Exception:
        # Retry with dict-like config for legacy helpers
        cfg_like = cfg.asdict() if hasattr(cfg, "asdict") else dict(vars(cfg))
        return func(df, cfg_like)


def build_signals(market_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Returns only pulse rows with the schema expected by bt_strategy:
      ['entry','dir','sig_long','sig_short','sl','tp','type','signals']
    Tries ICT pipeline first; if unavailable, uses an *intraday* breakout fallback.
    """
    print(f"[pipeline] using: {__file__}")
    df = market_df.copy()

    # Ensure tz-aware UTC index
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    # ---------- Prefer ICT pipeline if present ----------
    try:
        # NOTE: prep lives under data.*, others under ict.*
        from ..data.prep import attach_daily_levels
        from ..ict.bias import compute_bias
        from ..ict.sessions import mark_sessions
        from ..ict.fvg import scan_fvg_zones
        from ..ict.mss import detect_mss
        from ..ict.playbooks import scan_open_playbook

        work = _call_ict(attach_daily_levels, df, cfg)
        work = _call_ict(compute_bias, work, cfg)
        work = _call_ict(mark_sessions, work, cfg)
        work = _call_ict(scan_fvg_zones, work, cfg)
        work = _call_ict(detect_mss, work, cfg)
        sigs = _call_ict(scan_open_playbook, work, cfg)

        if isinstance(sigs, pd.DataFrame) and len(sigs):
            if "signals" not in sigs.columns:
                sigs = add_reason(sigs, "ICT:OpenPlay")
            sigs = finalize_signals(sigs)
            print(f"[pipeline] ICT produced {len(sigs)} signals")
            return sigs

        # If ICT returned empty, fall through to fallback
        print("[pipeline] ICT produced 0 signals; using fallback.")
    except Exception as e:
        # Swallow any ICT issues and continue to fallback
        print(f"[pipeline] ICT modules not ready, falling back. Reason: {e}")

    # ---------- Intraday fallback: prev-day H/L breakout within sessions ----------
    # case-insensitive OHLC mapping
    lower_map = {c.lower(): c for c in df.columns}
    needed = ("open", "high", "low", "close")
    missing = [k for k in needed if k not in lower_map]
    if missing:
        raise ValueError(f"fallback needs OHLC; missing {missing}")
    o, h, l, c = (df[lower_map[k]] for k in needed)

    levels = _daily_prev_hilo(df, cfg.tz)
    prev_high, prev_low = levels["prev_high"], levels["prev_low"]

    sess = _session_mask(df.index, cfg)

    long_hit = (h >= prev_high) & prev_high.notna() & sess
    short_hit = (l <= prev_low) & prev_low.notna() & sess

    # one-bar pulses (rising edge)
    long_pulse = long_hit & (~long_hit.shift(1, fill_value=False))
    short_pulse = short_hit & (~short_hit.shift(1, fill_value=False))
    both = long_pulse & short_pulse
    short_pulse = short_pulse & (~both)

    entry = (long_pulse | short_pulse).astype(int)
    direction = np.where(long_pulse, 1, np.where(short_pulse, -1, 0))

    atr = _atr(df, n=14)
    atr_mult_sl = 3.0
    rr = max(getattr(cfg.risk, "min_rr", 1.5), 1.0)

    sl = np.where(direction == 1, c - atr_mult_sl * atr,
         np.where(direction == -1, c + atr_mult_sl * atr, np.nan))
    tp = np.where(direction == 1, c + rr * (c - sl),
         np.where(direction == -1, c - rr * (sl - c), np.nan))

    sigs = pd.DataFrame({
        "entry": entry,
        "dir": direction,
        "sig_long": (direction == 1).astype(int),
        "sig_short": (direction == -1).astype(int),
        "sl": sl,
        "tp": tp,
        "type": np.where(direction == 1, "LONG",
                 np.where(direction == -1, "SHORT", "")),
        "signals": np.where(long_pulse | short_pulse, "FallbackBreakout", ""),
    }, index=df.index)

    sigs = sigs.loc[sigs["entry"] == 1]
    sigs = finalize_signals(sigs)

    print(f"[pipeline] fallback breakout produced {len(sigs)} intraday entries")
    return sigs
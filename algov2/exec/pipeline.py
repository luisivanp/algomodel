# algov2/exec/pipeline.py
from __future__ import annotations
import pandas as pd
import numpy as np

# ---- helpers ---------------------------------------------------------------

def _series_bool(df: pd.DataFrame, val, default: bool = False) -> pd.Series:
    """Return a boolean Series aligned to df.index from val (Series/ndarray/scalar/None)."""
    if isinstance(val, pd.Series):
        return val.reindex(df.index).fillna(False).astype(bool)
    if isinstance(val, (np.ndarray, list, tuple)):
        return pd.Series(val, index=df.index).fillna(False).astype(bool)
    if val is None:
        return pd.Series(default, index=df.index)
    # scalar bool/num/str → bool broadcast
    return pd.Series(bool(val), index=df.index)

def _safe_bias(df: pd.DataFrame, cfg) -> pd.DataFrame:
    try:
        from ..ict.bias import compute_bias
        out = compute_bias(df.copy(), cfg=cfg)
        for col in ("bias", "pd_zone"):
            if col in out.columns:
                df[col] = out[col]
        print("[pipeline] bias computed via ict.bias")
    except Exception as e:
        print(f"[pipeline] bias/ict failed: {e}")
        if "bias" not in df.columns:
            df["bias"] = 0
    return df

def _safe_sessions(df: pd.DataFrame, cfg) -> pd.DataFrame:
    try:
        from ..ict.sessions import mark_sessions
        df = mark_sessions(df, tz=getattr(cfg, "tz", "US/Eastern"),
                           windows=cfg.session_windows())
        print("[pipeline] sessions marked via ict.sessions")
    except Exception as e:
        print(f"[pipeline] sessions/ict failed: {e}")
        for nm in getattr(cfg.sessions, "windows", {}).keys():
            col = f"in_{nm}"
            if col not in df.columns:
                df[col] = False
    return df

def _safe_mss(df: pd.DataFrame, cfg) -> pd.DataFrame:
    try:
        from ..ict.mss import detect_mss
        out = detect_mss(
            df.copy(),
            left=getattr(cfg.mss, "swing_left", 2),
            right=getattr(cfg.mss, "swing_right", 2),
            require_displacement=getattr(cfg.mss, "require_displacement", True),
            displacement_factor=getattr(cfg.mss, "displacement_factor", 1.2),
        )
        for col in ("mss_up", "mss_down", "disp_up", "disp_down"):
            if col in out.columns:
                df[col] = _series_bool(df, out[col])
        print("[pipeline] mss via ict.mss")
    except Exception as e:
        print(f"[pipeline] mss/ict failed: {e}")
        for col in ("mss_up", "mss_down", "disp_up", "disp_down"):
            if col not in df.columns:
                df[col] = False
    return df

def _safe_fvg(df: pd.DataFrame, cfg) -> pd.DataFrame:
    try:
        from ..ict.fvg import scan_fvg_zones
        out = scan_fvg_zones(
            df.copy(),
            lookback=getattr(cfg.fvg, "lookback_bars", 100),
            min_size=getattr(cfg.fvg, "min_size_points", 1.0),
            allow_ifvg=getattr(cfg.fvg, "allow_ifvg", True),
            prefer_htf_parent=getattr(cfg.fvg, "prefer_htf_parent", True),
        )
        for name in ("fvg_top", "fvg_bot", "ifvg_top", "ifvg_bot", "fvg_long", "fvg_short"):
            if name in out.columns:
                df[name] = out[name]
        print("[pipeline] fvg via ict.fvg")
    except Exception as e:
        print(f"[pipeline] fvg/ict failed: {e}")
        for name in ("fvg_top", "fvg_bot", "ifvg_top", "ifvg_bot", "fvg_long", "fvg_short"):
            if name not in df.columns:
                df[name] = np.nan
    return df

def _safe_timing(df: pd.DataFrame, cfg) -> pd.DataFrame:
    try:
        from ..ict.timing import timing_gate
        df = timing_gate(
            df,
            cfg=cfg,
            tz=getattr(cfg, "tz", "US/Eastern"),
            allow_sessions=getattr(cfg.sessions, "allow_sessions", None),
            windows=cfg.session_windows(),
        )
        print("[pipeline] timing gate applied")
    except Exception as e:
        print(f"[pipeline] timing gate skipped: {e}")
        if "time_ok" not in df.columns:
            df["time_ok"] = True
    return df

def _safe_cooldown(df: pd.DataFrame, cfg) -> pd.DataFrame:
    try:
        from ..ict.cooldown import apply_cooldown
        df = apply_cooldown(df, minutes=0, column="entry")
        print("[pipeline] cooldown applied")
    except Exception as e:
        print(f"[pipeline] cooldown skipped: {e}")
        if "cool_ok" not in df.columns:
            df["cool_ok"] = True
    return df

def _has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def _fvg_presence(df: pd.DataFrame, primary: str, fallback: str | None = None) -> pd.Series:
    if _has_col(df, primary):
        s = df[primary]
        # if it's already boolean, keep; otherwise treat not-null as presence
        return s if s.dtype == bool else s.notna()
    if fallback and _has_col(df, fallback):
        s = df[fallback]
        return s if s.dtype == bool else s.notna()
    return pd.Series(False, index=df.index)

# ---- main ------------------------------------------------------------------

def build_signals(df_in: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Build signal frame with columns:
      ['entry','dir','sig_long','sig_short','sl','tp','type','signals']
    Confluence ≥ cfg.playbook.min_confs (default 2/3) decides entries.
    """
    df = df_in.copy()

    # core features
    df = _safe_bias(df, cfg)
    df = _safe_sessions(df, cfg)
    df = _safe_mss(df, cfg)
    df = _safe_fvg(df, cfg)
    df = _safe_timing(df, cfg)
    df = _safe_cooldown(df, cfg)

    # Ensure OHLCV
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"pipeline requires column '{col}'")

    # bias series
    bias = df.get("bias")
    if isinstance(bias, pd.Series):
        bias = bias.reindex(df.index).fillna(0)
    else:
        bias = pd.Series(0, index=df.index)
    bias_up = bias > 0
    bias_dn = bias < 0

    # timing / cooldown
    in_time = _series_bool(df, df.get("time_ok"), True)
    cool_ok = _series_bool(df, df.get("cool_ok"), True)

    # sessions
    in_london = _series_bool(df, df.get("in_LONDON"), False)
    in_nyam   = _series_bool(df, df.get("in_NYAM"), False)
    in_nypm   = _series_bool(df, df.get("in_NYPM"), False)
    in_any_session = in_london | in_nyam | in_nypm

    # mss / displacement (robust → Series)
    mss_up   = _series_bool(df, df.get("mss_up"), False)
    mss_down = _series_bool(df, df.get("mss_down"), False)
    disp_up  = _series_bool(df, df.get("disp_up"), False)
    disp_dn  = _series_bool(df, df.get("disp_down"), False)

    # fvg presence
    fvg_support = _fvg_presence(df, "fvg_bot", "fvg_long")
    fvg_resist  = _fvg_presence(df, "fvg_top", "fvg_short")

    # premium/discount enforcement
    enforce_pd = bool(getattr(cfg, "filters_kwargs", lambda: {})().get("enforce_pd_match", False))
    pd_zone = df.get("pd_zone")
    if enforce_pd and isinstance(pd_zone, pd.Series):
        pd_ok_long  = pd_zone.reindex(df.index).eq("discount").fillna(False)
        pd_ok_short = pd_zone.reindex(df.index).eq("premium").fillna(False)
    else:
        pd_ok_long  = pd.Series(True, index=df.index)
        pd_ok_short = pd.Series(True, index=df.index)

    # confluence stacks
    long_terms = pd.concat([
        bias_up.astype(bool),
        in_time, cool_ok, in_any_session,
        (mss_up | disp_up),
        fvg_support.astype(bool),
        pd_ok_long.astype(bool),
    ], axis=1)

    short_terms = pd.concat([
        bias_dn.astype(bool),
        in_time, cool_ok, in_any_session,
        (mss_down | disp_dn),
        fvg_resist.astype(bool),
        pd_ok_short.astype(bool),
    ], axis=1)

    long_confs  = long_terms.sum(axis=1).to_numpy()
    short_confs = short_terms.sum(axis=1).to_numpy()

    min_confs = int(getattr(cfg.playbook, "min_confs", 3))
    mask_long  = long_confs  >= min_confs
    mask_short = short_confs >= min_confs
    # hard-require trading during allowed time windows
    must_time = True
    if must_time:
        mask_long  = mask_long  & in_time.values
        mask_short = mask_short & in_time.values

    # resolve both-true bars by bias
    both = mask_long & mask_short
    if both.any():
        prefer_long = (bias >= 0).to_numpy()
        mask_long  = np.where(both, prefer_long, mask_long)
        mask_short = np.where(both, ~prefer_long, mask_short)

    # entry & dir arrays
    entry = np.zeros(len(df), dtype=int)
    dirv  = np.zeros(len(df), dtype=int)
    entry[mask_long]  = 1; dirv[mask_long]  = +1
    entry[mask_short] = 1; dirv[mask_short] = -1

    # SL/TP (swing-based)
    rr = float(getattr(cfg.playbook, "min_rr", 1.5))
    min_stop = float(getattr(cfg.risk, "min_stop_points", 1.0))
    close = df["close"].to_numpy()

    low_look  = pd.Series(df["low"]).rolling(10, min_periods=1).min().shift(1).to_numpy()
    high_look = pd.Series(df["high"]).rolling(10, min_periods=1).max().shift(1).to_numpy()

    sl = np.full(len(df), np.nan); tp = np.full(len(df), np.nan)

    li = np.where(mask_long)[0]
    if li.size:
        sl_long = np.minimum(low_look[li], close[li] - min_stop)
        too_tight = (close[li] - sl_long) < min_stop
        sl_long[too_tight] = close[li][too_tight] - min_stop
        tp_long = close[li] + rr * (close[li] - sl_long)
        sl[li] = sl_long; tp[li] = tp_long

    si = np.where(mask_short)[0]
    if si.size:
        sl_short = np.maximum(high_look[si], close[si] + min_stop)
        too_tight = (sl_short - close[si]) < min_stop
        sl_short[too_tight] = close[si][too_tight] + min_stop
        tp_short = close[si] - rr * (sl_short - close[si])
        sl[si] = sl_short; tp[si] = tp_short

    # assemble output
    out = pd.DataFrame(index=df.index)
    out["entry"]     = entry
    out["dir"]       = dirv
    out["sig_long"]  = (dirv == +1).astype(int)
    out["sig_short"] = (dirv == -1).astype(int)
    out["sl"]        = sl
    out["tp"]        = tp
    out["type"]      = "sl+tp"

    # reasons/tags
    reasons = []
    for i in range(len(df)):
        if entry[i] == 0:
            reasons.append("")
            continue
        tags = []
        if dirv[i] == +1:
            if bias_up.iloc[i]: tags.append("BiasUp")
            if (mss_up | disp_up).iloc[i]: tags.append("MSS/DispUp")
            if fvg_support.iloc[i]: tags.append("FVGsup")
        else:
            if bias_dn.iloc[i]: tags.append("BiasDn")
            if (mss_down | disp_dn).iloc[i]: tags.append("MSS/DispDn")
            if fvg_resist.iloc[i]: tags.append("FVGres")
        if in_london.iloc[i]: tags.append("London")
        if in_nyam.iloc[i]: tags.append("NYAM")
        if in_nypm.iloc[i]: tags.append("NYPM")
        reasons.append(",".join(tags))
    out["signals"] = reasons

    # debug prints
    def _ct(s): return int(s.sum()) if isinstance(s, pd.Series) else int(np.sum(s))
    print("[pipeline] bars:", len(df))
    print("[pipeline] bias +1/-1:", _ct(bias_up), "/", _ct(bias_dn))
    print("[pipeline] sessions LDN/NYAM/NYPM:", _ct(in_london), _ct(in_nyam), _ct(in_nypm))
    print("[pipeline] mss up/down:", _ct(mss_up), "/", _ct(mss_down))
    print("[pipeline] disp up/down:", _ct(disp_up), "/", _ct(disp_dn))
    print("[pipeline] fvg sup/res:", _ct(fvg_support), "/", _ct(fvg_resist))
    print("[pipeline] time_ok:", _ct(in_time), "cool_ok:", _ct(cool_ok))
    print(f"[pipeline] confluence threshold: {min_confs}")
    print("[pipeline] entries long/short:", int(mask_long.sum()), "/", int(mask_short.sum()))
    if int((out["entry"] == 1).sum()) == 0:
        print("[pipeline] NOTE: No entries. Consider lowering cfg.playbook.min_confs, "
              "loosening filters, or verify FVG/MSS outputs.")

    return out
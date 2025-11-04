# algov2/ict/sessions.py
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import time
from typing import Dict, Tuple, Iterable, Optional

__all__ = ["mark_sessions"]


def _parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def _in_window(t: time, start: time, end: time) -> bool:
    """Return True if time t is inside [start, end) with midnight wrap support."""
    if start <= end:
        return start <= t < end
    # window crosses midnight (e.g., 22:00 -> 02:00)
    return t >= start or t < end


def mark_sessions(
    df: pd.DataFrame,
    windows: Dict[str, Tuple[str, str]],
    tz_local: str = "US/Eastern",
    allowed: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Add session labels/flags to a time-series DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain a datetime column named 'dt'.
        'dt' may be tz-aware (any tz) or naive. We convert/localize to tz_local.
    windows : dict[str, (start_hhmm, end_hhmm)]
        Session name -> ("HH:MM", "HH:MM") in local time (tz_local).
        Example: {"LONDON": ("03:00","07:30"), "NYAM": ("08:30","11:30"), "NYPM": ("13:30","16:00")}
    tz_local : str
        IANA timezone string used to evaluate session windows.
    allowed : iterable[str] | None
        If provided, marks 'allowed_session' True only when session is in this iterable.

    Returns
    -------
    DataFrame
        Same df with new columns:
          - 'session' (str): session name or "OFF"
          - 'in_session' (bool)
          - 'allowed_session' (bool)
          - One boolean column per session, e.g. 'is_LONDON', 'is_NYAM', ...
    """
    if "dt" not in df.columns:
        raise ValueError("mark_sessions: DataFrame must contain a 'dt' column")

    out = df.copy()

    # Normalize datetime to tz_local
    dts = out["dt"]
    if getattr(dts.dt, "tz", None) is None:
        # naive -> localize as tz_local
        local_dt = dts.dt.tz_localize(tz_local)
    else:
        # aware -> convert to tz_local
        local_dt = dts.dt.tz_convert(tz_local)

    local_time = local_dt.dt.time
    n = len(out)

    # Prepare containers
    sess_names = list(windows.keys())
    session = np.array(["OFF"] * n, dtype=object)
    in_session = np.zeros(n, dtype=bool)
    masks_by_name = {}

    # Compute masks per session in given order (priority = order provided)
    for name in sess_names:
        start_s, end_s = windows[name]
        t_start = _parse_hhmm(start_s)
        t_end = _parse_hhmm(end_s)
        mask = local_time.map(lambda t: _in_window(t, t_start, t_end)).to_numpy()
        masks_by_name[name] = mask

    # Assign first matching session per row
    # If overlapping windows exist, the first one in dict order wins.
    remaining = np.ones(n, dtype=bool)
    for name in sess_names:
        mask = masks_by_name[name] & remaining
        session[mask] = name
        in_session[mask] = True
        remaining &= ~mask

    # Per-session boolean columns
    for name in sess_names:
        out[f"is_{name}"] = masks_by_name[name]

    out["session"] = session
    out["in_session"] = in_session

    if allowed is None:
        # If not provided, consider any named session as allowed
        allowed_set = set(sess_names)
    else:
        allowed_set = set(allowed)

    out["allowed_session"] = out["session"].isin(allowed_set)

    return out
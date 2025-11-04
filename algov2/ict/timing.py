# algov2/ict/timing.py
from __future__ import annotations

from typing import Dict, Iterable, Tuple, Optional
import pandas as pd


def session_mask(
    idx: pd.DatetimeIndex,
    windows: Dict[str, Tuple[str, str]],
    allow: Iterable[str],
) -> pd.Series:
    """Boolean mask True when index time is inside any allowed session window."""
    allow_set = set(a.upper() for a in allow)
    out = pd.Series(False, index=idx)
    if not windows:
        return pd.Series(True, index=idx)

    for name, (start_s, end_s) in windows.items():
        if name.upper() not in allow_set:
            continue
        st = pd.to_datetime(start_s).time()
        en = pd.to_datetime(end_s).time()
        t = idx.time
        if st <= en:
            m = (t >= st) & (t <= en)
        else:  # overnight
            m = (t >= st) | (t <= en)
        out = out | pd.Series(m, index=idx)
    return out


def lunch_block_mask(
    idx: pd.DatetimeIndex,
    enabled: bool = True,
    start: str = "12:00",
    end: str = "13:00",
) -> pd.Series:
    """True when NOT in lunch (if enabled)."""
    if not enabled:
        return pd.Series(True, index=idx)
    st = pd.to_datetime(start).time()
    en = pd.to_datetime(end).time()
    t = idx.time
    if st <= en:
        allow = ~((t >= st) & (t <= en))
    else:
        allow = ~((t >= st) | (t <= en))
    return pd.Series(allow, index=idx)


def timing_gate(
    idx: pd.DatetimeIndex,
    windows: Dict[str, Tuple[str, str]],
    allow: Iterable[str],
    block_lunch: bool = True,
    lunch_start: str = "12:00",
    lunch_end: str = "13:00",
) -> pd.Series:
    """Combined allowed-time mask: sessions AND (not-lunch if enabled)."""
    m_sess = session_mask(idx, windows, allow)
    m_lunch = lunch_block_mask(idx, block_lunch, lunch_start, lunch_end)
    return (m_sess & m_lunch).astype(bool)
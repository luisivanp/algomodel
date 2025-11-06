# algov2/ict/sessions.py
from __future__ import annotations
import pandas as pd
from datetime import time

def _local_index(idx: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("mark_sessions requires a DatetimeIndex")
    if idx.tz is None:
        # assume incoming is UTC if naive, then convert to local tz
        return idx.tz_localize("UTC").tz_convert(tz)
    return idx.tz_convert(tz)

def mark_sessions(df: pd.DataFrame, tz: str, windows: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """
    Adds boolean columns in_{SESSION} for each window in `windows`.
    windows: {"LONDON":("03:00","07:30"), "NYAM":("08:30","11:30"), "NYPM":("13:30","16:00")}
    Times are interpreted in `tz` (e.g., "US/Eastern").
    """
    out = df.copy()
    lidx = _local_index(out.index, tz)
    lt = lidx.time  # vector of datetime.time

    for name, (start, end) in windows.items():
        sh, sm = map(int, start.split(":"))
        eh, em = map(int, end.split(":"))
        s = time(sh, sm)
        e = time(eh, em)

        if s <= e:
            mask = [(t >= s and t <= e) for t in lt]
        else:  # overnight window (e.g., 22:00 -> 02:00)
            mask = [(t >= s or t <= e) for t in lt]

        out[f"in_{name}"] = pd.Series(mask, index=out.index, dtype=bool)

    return out
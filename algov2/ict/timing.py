# algov2/ict/timing.py
from __future__ import annotations
import pandas as pd

def timing_gate(
    df: pd.DataFrame,
    cfg,
    tz: str,
    allow_sessions: tuple[str, ...] | None = None,
    windows: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Sets df['time_ok'] True if row falls in any allowed session.
    If session flags are missing, defaults to True (no gate).
    """
    out = df.copy()
    if allow_sessions is None:
        allow_sessions = tuple(windows.keys()) if windows else ()

    flags = []
    for name in allow_sessions:
        col = f"in_{name}"
        if col in out.columns:
            flags.append(out[col].astype(bool))

    if flags:
        ok = flags[0].copy()
        for s in flags[1:]:
            ok |= s
    else:
        ok = pd.Series(True, index=out.index)

    out["time_ok"] = ok
    return out
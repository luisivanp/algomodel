# algov2/ict/cooldown.py
from __future__ import annotations
import pandas as pd

def apply_cooldown(
    df: pd.DataFrame,
    minutes: int = 0,
    column: str = "entry",
    **kwargs,
) -> pd.DataFrame:
    """
    No-op cooldown by default. If you later want a real cooldown,
    convert 'entry' pulses into a series that blanks out the next N minutes.
    """
    if column not in df.columns:
        df[column] = 0
    df["cool_ok"] = True
    return df
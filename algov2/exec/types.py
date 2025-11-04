# algov2/exec/types.py
from __future__ import annotations
from typing import Optional

LONG = "LONG"
SHORT = "SHORT"
FLAT = "FLAT"

DIR_LONG = 1
DIR_SHORT = -1
DIR_FLAT = 0

def dir_to_str(d: Optional[float]) -> str:
    if d is None:
        return FLAT
    try:
        di = int(d)
    except Exception:
        return FLAT
    return LONG if di > 0 else SHORT if di < 0 else FLAT

def str_to_dir(s: Optional[str]) -> int:
    if not s:
        return DIR_FLAT
    s_up = str(s).upper()
    if s_up == LONG:
        return DIR_LONG
    if s_up == SHORT:
        return DIR_SHORT
    return DIR_FLAT

__all__ = [
    "LONG", "SHORT", "FLAT",
    "DIR_LONG", "DIR_SHORT", "DIR_FLAT",
    "dir_to_str", "str_to_dir",
]
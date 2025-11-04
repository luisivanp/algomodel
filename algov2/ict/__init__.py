# algov2/__init__.py
from __future__ import annotations

"""
algov2 — ICT-inspired intraday backtesting toolkit (Backtrader-based)

This package exposes a minimal public API at import time to avoid circular
imports with Backtrader strategies. Heavier modules are imported lazily.

Public exports:
- __version__
- get_config()          -> algov2.config.get_config
- about()               -> quick environment diagnostics
- main()                -> algov2.runner.main  (lazy)
- prepare_market_df()   -> algov2.loader.prepare_market_df  (lazy)
"""

from typing import Any
import importlib

__version__ = "0.6.0"

# --- lightweight, safe import (no BT dependencies)
try:
    from ..config import get_config  # noqa: F401
except Exception:  # pragma: no cover
    # Keep import-time failures non-fatal; users can still call about()
    def get_config() -> Any:
        raise RuntimeError("get_config is unavailable: config module failed to import.")


def about() -> str:
    """
    Return a short string with environment diagnostics and optional deps.
    Useful when users hit parquet/excel issues.
    """
    import sys
    import platform

    lines = []
    lines.append(f"algov2 {__version__}")
    lines.append(f"python  {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    # Optional libs
    for mod in ("pyarrow", "fastparquet", "openpyxl", "backtrader", "pandas", "numpy"):
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "OK")
            lines.append(f"{mod:<12} {ver}")
        except Exception:
            lines.append(f"{mod:<12} MISSING")
    info = "\n".join(lines)
    print(info)
    return info


# --- lazy attribute access for heavier imports to avoid circulars
def __getattr__(name: str):
    """
    Lazy-load heavy or optional attributes on first access.

    - main               -> algov2.runner.main
    - prepare_market_df  -> algov2.loader.prepare_market_df
    - SignalExecutor     -> algov2.exec.bt_strategy.SignalExecutor
    """
    if name == "main":
        return importlib.import_module(".runner", __name__).main
    if name == "prepare_market_df":
        return importlib.import_module(".loader", __name__).prepare_market_df
    if name == "SignalExecutor":
        return importlib.import_module(".exec.bt_strategy", __name__).SignalExecutor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "__version__",
    "get_config",
    "about",
    "main",
    "prepare_market_df",
    "SignalExecutor",
]
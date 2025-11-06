# algov2/exec/__init__.py
"""
Execution-layer package (Backtrader strategy, pipeline, manager).
Do NOT import heavy modules here to avoid circular imports.
Import directly from submodules in callers, e.g.:
    from algov2.exec.pipeline import build_signals
    from algov2.exec.bt_strategy import SignalExecutor
"""
from __future__ import annotations

__all__ = []  # keep empty; import from submodules explicitly
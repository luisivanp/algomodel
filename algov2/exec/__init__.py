# algov2/exec/__init__.py
from __future__ import annotations

# Re-export both names so old code importing ICTStrategy keeps working.
from .bt_strategy import SignalExecutor as ICTStrategy, SignalExecutor

__all__ = ["SignalExecutor", "ICTStrategy"]
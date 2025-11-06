# algov2/ict/__init__.py
"""
ICT/SMC feature detectors (bias, sessions, MSS, FVG, etc.).
Keep this import-light; callers should import the specific tools they need:
    from algov2.ict.bias import compute_bias
    from algov2.ict.sessions import mark_sessions
    from algov2.ict.mss import detect_mss
    from algov2.ict.fvg import scan_fvg_zones
"""
from __future__ import annotations

__all__ = []  # avoid eager imports; prevents cycles
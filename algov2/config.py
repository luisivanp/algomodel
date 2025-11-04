# algov2/config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import os


# ------------------------------
# Session windows (interpreted in cfg.tz)
# Times are "HH:MM" 24h for the local timezone configured below.
# ------------------------------
DEFAULT_SESSIONS: Dict[str, Tuple[str, str]] = {
    # London futures flow (pre-NY): keep it focused to avoid chop
    "LONDON": ("03:00", "07:30"),
    # NY AM index/futures session (CPI/NFP at 08:30 often matters)
    "NYAM":   ("08:30", "11:30"),
    # NY PM continuation/mean-revert window
    "NYPM":   ("13:30", "16:00"),
}


# ------------------------------
# Nested configs
# ------------------------------

@dataclass
class BiasConfig:
    enabled: bool = True
    daily_lookback_days: int = 20         # lookback to detect prior highs/lows
    use_daily_open: bool = True           # include daily open for PD calc / bias
    require_align_for_trades: bool = True # drop counter-bias signals


@dataclass
class FVGConfig:
    enabled: bool = True
    lookback_bars: int = 150              # scan depth on working TF
    min_size_points: float = 1.0          # min size of the imbalance (price pts)
    allow_ifvg: bool = True               # include inversion FVGs (iFVG)
    prefer_htf_parent: bool = True        # prioritize HTF/parent FVG if present
    htf_frame: str = "15m"                # HTF used by prep/pipeline scanners


@dataclass
class MSSConfig:
    enabled: bool = True
    swing_left: int = 2                   # swing pivot definition
    swing_right: int = 2
    require_displacement: bool = True     # large impulse required
    displacement_factor: float = 1.3      # vs avg body/ATR threshold


@dataclass
class SessionConfig:
    allow_sessions: Tuple[str, ...] = ("LONDON", "NYAM", "NYPM")
    windows: Dict[str, Tuple[str, str]] = field(default_factory=lambda: DEFAULT_SESSIONS.copy())


@dataclass
class PlaybookConfig:
    min_confs: int = 2                    # minimum confluences to allow a setup
    use_ma9_filter: bool = True           # basic orderflow gating
    min_rr: float = 1.5                   # sanity check at strategy level
    prefer_tp: str = "best"               # "tp1" | "tp2" | "best"
    partials: bool = True                 # take partial at TP1 then trail/TP2


@dataclass
class RiskConfig:
    # execution sizing/constraints (consumed by exec.manager)
    risk_mode: str = "fixed_dollars"      # "fixed_contracts" | "fixed_dollars" | "percent_equity"
    fixed_size: int = 1
    risk_dollars: float = 300.0           # ~$300 per trade risk maps well to NQ
    risk_pct: float = 0.005               # 0.5% if percent_equity mode
    start_cash: float = 100_000.0
    max_contracts: int = 3
    min_rr: float = 1.5
    min_stop_points: float = 2.0          # avoid ultra-tight stops (< 0.5 NQ pt tick=0.25)
    point_value: float = 20.0             # NQ ≈ $20/point
    prefer_tp: str = "best"


@dataclass
class BrokerConfig:
    start_cash: float = 100_000.0
    commission_per_contract: float = 2.00  # per side, per contract
    slippage_points: float = 0.25          # ~1 tick on NQ
    slip_randomize: bool = False


@dataclass
class IOConfig:
    data_dir: str = "data"
    cache_dir: str = "cache"
    report_dir: str = "reports"
    use_cache: bool = True
    parquet_engine: str = "auto"          # "auto" | "pyarrow" | "fastparquet"
    export_excel: bool = True


# ------------------------------
# Main Config
# ------------------------------

@dataclass
class Config:
    # instrument & time
    symbol: str = "NQ=F"
    start: Optional[str] = None           # ISO "YYYY-MM-DD"; filled in get_config() if None
    end: Optional[str] = None
    timeframe: str = "5m"
    tz: str = "US/Eastern"                # all session windows interpreted in this tz

    # feature toggles / parameters
    bias: BiasConfig = field(default_factory=BiasConfig)
    fvg: FVGConfig = field(default_factory=FVGConfig)
    mss: MSSConfig = field(default_factory=MSSConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    playbook: PlaybookConfig = field(default_factory=PlaybookConfig)

    # execution / broker / io
    risk: RiskConfig = field(default_factory=RiskConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    io: IOConfig = field(default_factory=IOConfig)

    # misc
    debug: bool = False
    seed: Optional[int] = 42

    # ----- helpers -----
    def asdict(self) -> dict:
        return asdict(self)

    def session_windows(self) -> Dict[str, Tuple[str, str]]:
        return self.sessions.windows

    def risk_kwargs(self) -> dict:
        """Translate to exec.manager.RiskParams kwargs (no import here)."""
        return {
            "risk_mode": self.risk.risk_mode,
            "fixed_size": self.risk.fixed_size,
            "risk_dollars": self.risk.risk_dollars,
            "risk_pct": self.risk.risk_pct,
            "start_cash": self.broker.start_cash,
            "max_contracts": self.risk.max_contracts,
            "min_rr": self.risk.min_rr,
            "min_stop_ticks": self.risk.min_stop_points,  # naming kept for downstream
            "point_value": self.risk.point_value,
            "prefer_tp": self.risk.prefer_tp,
        }

    def filters_kwargs(self) -> dict:
        """Translate to exec.manager.Filters kwargs."""
        return {
            "allow_sessions": self.sessions.allow_sessions,
            "require_bias_align": self.bias.require_align_for_trades,
            "premium_discount_col": "pd_zone",
            "enforce_pd_match": False,      # set True later if you want LONG=discount / SHORT=premium
            "one_signal_per_bar": True,
        }


# ------------------------------
# Backward-compat properties expected by existing code
# ------------------------------

def _add_compat_aliases():
    def interval(self) -> str:  # old name for timeframe
        return self.timeframe

    def timezone(self) -> str:  # old name for tz
        return self.tz

    def report_dir(self) -> str:
        return self.io.report_dir

    def commission(self) -> float:
        return self.broker.commission_per_contract

    def slippage(self) -> float:
        return self.broker.slippage_points

    def point_value(self) -> float:
        return self.risk.point_value

    def cash(self) -> float:
        return self.broker.start_cash

    setattr(Config, "interval", property(interval))
    setattr(Config, "timezone", property(timezone))
    setattr(Config, "report_dir", property(report_dir))
    setattr(Config, "commission", property(commission))
    setattr(Config, "slippage", property(slippage))
    setattr(Config, "point_value", property(point_value))
    setattr(Config, "cash", property(cash))


_add_compat_aliases()


# ------------------------------
# Factory
# ------------------------------

def _iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def get_config() -> Config:
    """
    Main entrypoint used by runner.
    - Reads optional env overrides
      ALGOV2_SYMBOL, ALGOV2_START, ALGOV2_END, ALGOV2_TZ, ALGOV2_TIMEFRAME
    - If start/end not provided, defaults to last 30 calendar days up to yesterday.
    """
    cfg = Config()

    # env overrides (nice for quick experiments without touching code)
    sym = os.getenv("ALGOV2_SYMBOL")
    if sym:
        cfg.symbol = sym

    tf = os.getenv("ALGOV2_TIMEFRAME")
    if tf:
        cfg.timeframe = tf

    tz = os.getenv("ALGOV2_TZ")
    if tz:
        cfg.tz = tz

    env_start = os.getenv("ALGOV2_START")
    env_end = os.getenv("ALGOV2_END")

    if env_start:
        cfg.start = env_start
    if env_end:
        cfg.end = env_end

    # sensible rolling defaults if not provided
    if cfg.start is None or cfg.end is None:
        today = datetime.utcnow().date()
        end = today - timedelta(days=1)
        start = end - timedelta(days=30)
        cfg.start = _iso(datetime.combine(start, datetime.min.time()))
        cfg.end = _iso(datetime.combine(end, datetime.min.time()))

    return cfg
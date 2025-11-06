# algov2/loader.py
from __future__ import annotations
from typing import Optional, Dict
import pandas as pd

# Optional yfinance; you can swap this out for your own provider later.
try:
    import yfinance as yf
except Exception:
    yf = None


_INTERVAL_MAP: Dict[str, str] = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
    "90m": "90m",
    "1h": "60m",
    "1d": "1d",
}


def _flatten_and_normalize_cols(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Make column names simple: open, high, low, close, volume
    Handles MultiIndex from yfinance and suffixes like '_NQ=F' or '_nq_f'.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([c for c in tup if c not in (None, "")]) for tup in df.columns]

    # lower-case + underscores
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # strip symbol suffix variants
    sym_lower = symbol.lower()
    sym_sanit = sym_lower.replace("=", "_").replace("^", "")
    candidates = {f"_{sym_lower}", f"_{sym_sanit}"}

    def strip_sym_suffix(name: str) -> str:
        for suf in candidates:
            if name.endswith(suf):
                return name[: -len(suf)]
        return name

    df.columns = [strip_sym_suffix(c) for c in df.columns]

    # unify common yahoo names
    rename_map = {
        "adj_close": "adj_close",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    # If yahoo used capitalized names and survived lowercasing, these are already handled.

    # Sometimes yahoo uses 'close' but volume is 'volume'; just ensure we keep required ones
    df = df.rename(columns=rename_map)

    return df


def _ensure_dt_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure index is tz-aware UTC DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    # drop any completely invalid rows
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _yf_download(symbol: str, start: str, end: str, timeframe: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not available. Install with `pip install yfinance`.")

    interval = _INTERVAL_MAP.get(timeframe, timeframe)
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"Empty data for {symbol} {start}->{end} {timeframe}")

    df = _flatten_and_normalize_cols(df, symbol)
    df = _ensure_dt_utc_index(df)

    # Expect these five
    need = ["open", "high", "low", "close", "volume"]
    # Some providers return 'adj_close' too; we ignore it for OHLCV
    missing = [c for c in need if c not in df.columns]
    if missing:
        # Try common alternative names (rare)
        alts = {
            "open": ["opening_price"],
            "high": ["max_price"],
            "low": ["min_price"],
            "close": ["closing_price", "price"],
            "volume": ["vol"],
        }
        for k in list(missing):
            if any(a in df.columns for a in alts.get(k, [])):
                # map the first alt that exists
                for a in alts[k]:
                    if a in df.columns:
                        df[k] = df[a]
                        break
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(
                f"Downloaded data missing columns: {missing} | got={list(df.columns)}"
            )

    df = df.astype({c: float for c in need})
    df = df.dropna(subset=["close"])
    return df[need]


def fetch_prices(symbol: str,
                 start: str,
                 end: str,
                 timeframe: str,
                 tz_local: Optional[str] = None) -> pd.DataFrame:
    """
    Primary entry point used by data.prep.prepare_market_df.
    Returns OHLCV with UTC tz-aware index and columns: open, high, low, close, volume.
    """
    df = _yf_download(symbol, start, end, timeframe)

    # If caller wants data *also* in a local tz later, they can convert; we keep UTC here.
    # (prepare_market_df will attach daily levels using tz_local for daily boundaries if needed)
    return df
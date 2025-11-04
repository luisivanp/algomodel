# algov2/loader.py
from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


def _tag(symbol: str) -> str:
    return re.sub(r"\W+", "", symbol).upper()


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns to lower-case: open, high, low, close, volume.
    Handle possible MultiIndex columns from yfinance.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Collapse MultiIndex to level=0 names like 'Open','High',...
        df.columns = [c[0] for c in df.columns]

    # Standard yfinance names -> ours
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    # Case-insensitive rename
    norm = {c: rename_map.get(c, c).lower() for c in df.columns}
    df = df.rename(columns=norm)

    # Require OHLCV
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data missing columns: {missing}")

    # Reorder with required first
    other = [c for c in df.columns if c not in required]
    return df[required + other]


def _localize_index(idx: pd.Index, tz_local: str) -> pd.DatetimeIndex:
    dt = pd.to_datetime(idx)
    if getattr(dt, "tz", None) is None:
        # yfinance intraday often returns tz-naive -> assume UTC then convert
        dt = dt.tz_localize("UTC")
    return dt.tz_convert(tz_local)


def _read_cache(path: str, tz_local: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = _localize_index(df.index, tz_local)
    return _ensure_cols(df)


def _write_cache(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write with ISO index
    df_out = df.copy()
    # Store as naive ISO to be robust across tz differences on read
    if getattr(df_out.index, "tz", None) is not None:
        df_out.index = df_out.index.tz_convert("UTC").tz_localize(None)
    df_out.to_csv(path)


def fetch_prices(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "5m",
    tz_local: str = "US/Eastern",
    use_cache: bool = True,
    data_dir: str = "data",
    cache_dir: str = "cache",
) -> pd.DataFrame:
    """
    Return OHLCV DataFrame indexed by localized timestamps with columns:
    open, high, low, close, volume (lowercase). Uses yfinance and caches to CSV.

    Parameters mirror what prep.prepare_market_df expects.
    """
    if timeframe.lower() not in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
        raise ValueError(f"Unsupported intraday timeframe for yfinance: {timeframe}")

    tag = _tag(symbol)
    cache_path = os.path.join(cache_dir, f"{tag}_{start}_{end}_{timeframe}.csv")

    # Try cache first
    if use_cache:
        cached = _read_cache(cache_path, tz_local)
        if cached is not None and not cached.empty:
            return cached

    if yf is None:
        raise ImportError("yfinance is required: pip install yfinance")

    print(f"Downloading {timeframe} data from {start} to {end} for {symbol}...")

    # yfinance can be picky with intraday; we keep it simple and robust
    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=timeframe,
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
        group_by="column",  # avoid MultiIndex if possible
    )

    if df is None or len(df) == 0:
        raise RuntimeError("No data returned from yfinance")

    # Normalize columns and index
    df = _ensure_cols(df)
    df.index = _localize_index(df.index, tz_local)

    # Clean/sanity
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop bars without prices; fill volume NaN with 0
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)

    # Remove obvious duplicates / sort
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Cache
    if use_cache:
        try:
            _write_cache(df, cache_path)
        except Exception:
            pass  # cache write failure shouldn't stop the run

    return df
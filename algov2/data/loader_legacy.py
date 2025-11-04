# algov2/data/loader_legacy.py
from __future__ import annotations

import os
from typing import Optional
import pandas as pd

# yfinance is the only dependency here
try:
    import yfinance as yf
except Exception as e:
    yf = None


def _map_interval(tf: str) -> str:
    """Map our timeframe tokens to yfinance intervals."""
    tf = (tf or "5m").lower()
    return {
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "60m",
        "1h": "60m",
        "1d": "1d",
    }.get(tf, "5m")


def _flatten_yf_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance’s MultiIndex columns and standardize names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]  # ('Open','NQ=F') -> 'Open'
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    # Keep only the fields we need and order them
    keep = ["open", "high", "low", "close", "volume"]
    # Some feeds miss Volume on 5m — fill with 0 if absent
    for k in keep:
        if k not in df.columns:
            df[k] = 0.0 if k == "volume" else pd.NA
    return df[keep]


def fetch_prices(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "5m",
    tz_local: str = "US/Eastern",
    use_cache: bool = False,
    cache_dir: str = "cache",
) -> pd.DataFrame:
    """
    Legacy interface expected by older pipeline code.
    Returns a DataFrame indexed by *naive* local-time datetimes with columns:
    ['open','high','low','close','volume'] (float).
    """
    if yf is None:
        raise RuntimeError("yfinance not installed. pip install yfinance")

    os.makedirs(cache_dir, exist_ok=True)
    cache_name = f"{symbol.replace('=','')}_{start}_{end}_{timeframe}.csv"
    cache_path = os.path.join(cache_dir, cache_name)

    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=[0], index_col=0)
    else:
        interval = _map_interval(timeframe)
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            prepost=False,
            progress=False,
            auto_adjust=False,
            threads=True,
        )
        if df is None or len(df) == 0:
            raise RuntimeError("No data returned from yfinance")

        df = _flatten_yf_cols(df)

        # Normalize index -> local naive (Backtrader & our prep prefer naive)
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        idx = idx.tz_convert(tz_local).tz_localize(None)
        df.index = idx

        # Clean up any bad rows
        df = df.dropna(subset=["open", "high", "low", "close"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])

        if use_cache:
            df.to_csv(cache_path)

    # Final sanity: enforce column order/types
    df = df.astype(
        {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
        errors="ignore",
    )
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = None
    return df
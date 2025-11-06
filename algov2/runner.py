# algov2/runner.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import backtrader as bt

# ---- Local imports (lazy where helpful to avoid cycles) ----
from .config import get_config, Config
from .exec.bt_strategy import SignalData, SignalExecutor  # feed + strategy


# ----------------- Helpers -----------------

def ensure_tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Make index tz-naive (Excel-safe). Assumes index is datetime-like or convertible."""
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        # try to coerce
        idx = pd.to_datetime(df.index, utc=True, errors='coerce')
        df = df.copy()
        df.index = idx
    # now tz-normalize -> naive
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        else:
            # already naive; keep as-is
            pass
    elif isinstance(df.index, pd.PeriodIndex):
        # convert to timestamp then drop tz
        df.index = df.index.to_timestamp().tz_localize(None)
    # name it consistently for Backtrader/Excel niceness
    if df.index.name is None:
        df.index.name = "Datetime"
    return df


def export_to_excel(report_dir: str, symbol_tag: str, start: str, end: str,
                    market_df: pd.DataFrame, signals_df: pd.DataFrame) -> str:
    """Write two sheets: prices & signals, with tz-naive indices."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    outfile = os.path.join(
        report_dir,
        f"trades_{symbol_tag}_{start.replace('-','')}_{end.replace('-','')}.xlsx"
    )

    # select standard OHLCV if present
    price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in market_df.columns]
    md = ensure_tz_naive(market_df[price_cols].copy()) if price_cols else ensure_tz_naive(market_df.copy())
    sg = ensure_tz_naive(signals_df.copy()) if not signals_df.empty else pd.DataFrame()

    with pd.ExcelWriter(outfile, engine="openpyxl") as xw:
        md.to_excel(xw, sheet_name="prices")
        if not sg.empty:
            sg.to_excel(xw, sheet_name="signals")

    return outfile


def _safe_numeric(s: pd.Series, fallback: float | int = 0) -> pd.Series:
    """Coerce to numeric; replace non-numeric with fallback."""
    return pd.to_numeric(s, errors="coerce").fillna(fallback)


def merge_price_and_signals(market_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join prices and signals on the index, and keep the columns Backtrader needs.
    Backtrader requires: open, high, low, close, volume (lowercase).
    Strategy feed uses: entry, dir, sl, tp
    """
    df = market_df.copy()
    if not signals_df.empty:
        df = df.join(signals_df[["entry", "dir", "sl", "tp"]], how="left")

    # Ensure mandatory columns exist
    for col in ["entry", "dir", "sl", "tp"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Clean dtypes for the four signal columns
    df["entry"] = _safe_numeric(df["entry"], 0).clip(lower=0, upper=1)
    df["dir"]   = _safe_numeric(df["dir"], 0).apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    df["sl"]    = pd.to_numeric(df["sl"], errors="coerce")  # NaN allowed -> strategy falls back
    df["tp"]    = pd.to_numeric(df["tp"], errors="coerce")  # NaN allowed -> strategy falls back

    # Backtrader requires tz-naive index
    df = ensure_tz_naive(df)

    # A quick sanity: sort by index to ensure monotonic increasing time
    df = df.sort_index()

    return df


def load_prices(cfg: Config) -> pd.DataFrame:
    """Call the prep routine to fetch and prepare OHLCV over the requested range."""
    try:
        from .data.prep import prepare_market_df  # local import
    except Exception as e:
        print(f"[runner] fatal: could not import data.prep.prepare_market_df: {e}", file=sys.stderr)
        raise

    symbol = cfg.symbol
    start  = cfg.start
    end    = cfg.end
    tf     = cfg.timeframe
    tzloc  = cfg.tz

    df = prepare_market_df(symbol, start, end, tf, tzloc)  # must return ohlcv with datetime index
    if not all(c in df.columns for c in ["open", "high", "low", "close", "volume"]):
        got = list(df.columns)
        raise ValueError(f"Downloaded data missing columns: ['open','high','low','close','volume'] | got={got}")
    return df


def build_pipeline_signals(market_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Run exec.pipeline.build_signals(df, cfg) and return a signals-only DataFrame."""
    try:
        from .exec import pipeline as pipeline_mod  # prints its path for debug
    except Exception as e:
        print(f"[runner] fatal: cannot import exec.pipeline: {e}", file=sys.stderr)
        return pd.DataFrame(columns=["entry", "dir", "sl", "tp"])

    try:
        signals_df = pipeline_mod.build_signals(market_df, cfg)
    except Exception as e:
        print(f"[runner] pipeline/build_signals error: {e}", file=sys.stderr)
        return pd.DataFrame(columns=["entry", "dir", "sl", "tp"])

    # Keep only the columns the executor expects
    keep = [c for c in ["entry", "dir", "sl", "tp", "type", "signals", "time_ok", "cool_ok"] if c in signals_df.columns]
    return signals_df[keep].copy() if keep else pd.DataFrame(columns=["entry", "dir", "sl", "tp"])


def print_signal_snapshot(df: pd.DataFrame, max_rows: int = 10) -> None:
    cols = [c for c in ["entry", "dir", "sl", "tp", "type", "signals", "time_ok", "cool_ok"] if c in df.columns]
    if not cols:
        print("[runner] (no signal columns to preview)")
        return
    nonzero = df.loc[df["entry"].fillna(0) > 0]
    print(f"[runner] signal columns present: {cols}")
    print(f"[runner] found {len(nonzero)} signal rows")
    if len(nonzero) > 0:
        print(nonzero[cols].head(max_rows))


# ----------------- Main execution -----------------

def run() -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = get_config()  # your chosen best-defaults

    # 1) Load OHLCV
    market_df = load_prices(cfg)
    print(f"[runner] market bars: {len(market_df)} | first: {market_df.index.min()} | last: {market_df.index.max()}")

    # 2) Build signals via pipeline
    signals_df = build_pipeline_signals(market_df, cfg)
    print_signal_snapshot(signals_df)

    # 3) Export report (prices + raw signals)
    symbol_tag = cfg.symbol.replace('=F', 'F').replace('=f', 'F').replace('^', '')
    xls_path = export_to_excel(cfg.io.report_dir, symbol_tag, cfg.start, cfg.end, market_df, signals_df)
    print(f"Saved Excel report → {xls_path}")

    # If there are no entries, still stop cleanly after export
    if signals_df.empty or (signals_df["entry"].fillna(0).sum() == 0):
        print("Initial Portfolio Value: {:.2f}".format(cfg.broker.start_cash))
        print("Final Portfolio Value:   {:.2f}".format(cfg.broker.start_cash))
        print("Trades 0 | Wins 0 | Losses 0 | WinRate 0.00%")
        print("Max Drawdown: 0.00%")
        return market_df, signals_df

    # 4) Merge price + signals for Backtrader feed
    df_bt = merge_price_and_signals(market_df, signals_df)

    # 5) Backtrader: set up engine
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cfg.broker.start_cash)
    # commission per contract side (futures-like)
    cerebro.broker.setcommission(commission=cfg.broker.commission_per_contract)

    # Data feed (mapping done by SignalData params)
    data_feed = SignalData(
        dataname=df_bt,
        timeframe=bt.TimeFrame.Minutes,
        compression=int(cfg.timeframe.strip('m')) if cfg.timeframe.endswith('m') else 1,
    )
    cerebro.adddata(data_feed)

    # Strategy (no kwargs—SignalExecutor reads signal lines from feed)
    cerebro.addstrategy(SignalExecutor)

    print("Initial Portfolio Value: {:.2f}".format(cerebro.broker.getvalue()))
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print("Final Portfolio Value:   {:.2f}".format(final_value))

    # 6) Done
    return market_df, signals_df


def main():
    try:
        run()
    except Exception as e:
        print(f"[runner] fatal: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
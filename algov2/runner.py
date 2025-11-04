# algov2/runner.py
from __future__ import annotations

import os
import re
import inspect
from pathlib import Path
from typing import Optional, Iterable, Set

import numpy as np
import pandas as pd
import backtrader as bt

from .config import get_config
from .data.prep import prepare_market_df


# -----------------------------
# Robust datetime index helper
# -----------------------------
def ensure_dt_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("ensure_dt_index: empty DataFrame")

    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif isinstance(df.index, pd.PeriodIndex):
        idx = df.index.to_timestamp()
    else:
        # detect a datetime-like column
        import re as _re
        col_candidates = [c for c in df.columns if _re.search(r"(datetime|timestamp|^ts$|date|time)", str(c), _re.I)]
        preferred = ["datetime", "timestamp", "ts", "date", "time"]

        ordered: list = []
        for p in preferred:
            for c in col_candidates:
                if str(c).lower() == p:
                    ordered.append(c)
        for c in col_candidates:
            if c not in ordered:
                ordered.append(c)

        found_idx = None
        used_col = None
        for c in ordered:
            try:
                ser = pd.to_datetime(df[c], errors="coerce", utc=True)
                if ser.notna().any():
                    found_idx = pd.DatetimeIndex(ser)
                    used_col = c
                    break
            except Exception:
                continue

        if found_idx is None:
            # last resort: parse index
            try:
                parsed = pd.to_datetime(df.index, errors="raise", utc=True)
                found_idx = pd.DatetimeIndex(parsed)
            except Exception as e:
                sample_cols = list(map(str, df.columns[:10]))
                raise TypeError(
                    f"ensure_dt_index: could not find/parse a datetime column or index. "
                    f"Columns sample={sample_cols}"
                ) from e

        idx = found_idx

    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(tz)

    out = df.copy()
    out.index = idx

    try:
        if 'used_col' in locals() and used_col in out.columns:
            out = out.drop(columns=[used_col])
    except Exception:
        pass

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def sanitize_symbol(symbol: str) -> str:
    return re.sub(r"\W+", "", symbol).upper()


def export_to_excel(report_dir: str,
                    symbol: str,
                    start: str,
                    end: str,
                    trades_df: Optional[pd.DataFrame],
                    tx_df: Optional[pd.DataFrame]) -> str:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    sym = sanitize_symbol(symbol)
    xls_path = os.path.join(report_dir, f"trades_{sym}_{start}_{end}.xlsx")

    with pd.ExcelWriter(xls_path, engine="openpyxl") as xw:
        if trades_df is not None and len(trades_df):
            trades_df.to_excel(xw, sheet_name="trades", index=False)
        else:
            pd.DataFrame({"notice": ["no trades"]}).to_excel(xw, sheet_name="trades", index=False)

        if tx_df is not None and len(tx_df):
            tx_df.to_excel(xw, sheet_name="transactions", index=False)
        else:
            pd.DataFrame({"notice": ["no transactions"]}).to_excel(xw, sheet_name="transactions", index=False)

    return xls_path


# -----------------------------
# Fallback minimal strategy
# -----------------------------
class MinimalSignalStrategy(bt.Strategy):
    params = dict(
        entry=None,       # np.ndarray
        sig_long=None,    # np.ndarray
        sig_short=None,   # np.ndarray
        sl=None,          # np.ndarray
        tp=None,          # np.ndarray
        size=1,
    )

    def __init__(self):
        self.trades_log = []
        self.tx_log = []

    def notify_order(self, order):
        if order.status in [order.Completed]:
            otype = "BUY" if order.isbuy() else "SELL"
            dt = bt.num2date(order.executed.dt)
            self.tx_log.append(dict(
                datetime=dt,
                type=otype,
                price=order.executed.price,
                size=order.executed.size,
                value=order.executed.value,
                comm=order.executed.comm,
            ))
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            try:
                dt = bt.num2date(order.created.dt)
            except Exception:
                dt = None
            self.tx_log.append(dict(
                datetime=dt,
                type="CANCELLED",
                price=float(getattr(getattr(order, "created", object()), "price", np.nan) or np.nan),
                size=getattr(getattr(order, "created", object()), "size", np.nan),
                value=np.nan,
                comm=np.nan,
            ))

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        dtopen = bt.num2date(trade.dtopen)
        dtclose = bt.num2date(trade.dtclose)
        self.trades_log.append(dict(
            open_datetime=dtopen,
            close_datetime=dtclose,
            size=trade.size,
            price_open=trade.price,
            price_close=trade.price,
            pnl=trade.pnl,
            pnlcomm=trade.pnlcomm,
        ))

    def next(self):
        i = len(self.data) - 1
        try:
            entry = int(self.p.entry[i]) if self.p.entry is not None else 0
            go_long = int(self.p.sig_long[i]) if self.p.sig_long is not None else 0
            go_short = int(self.p.sig_short[i]) if self.p.sig_short is not None else 0
            sl = float(self.p.sl[i]) if self.p.sl is not None else np.nan
            tp = float(self.p.tp[i]) if self.p.tp is not None else np.nan
        except Exception:
            return

        if not entry or (go_long == go_short):
            return
        if self.position:
            return

        size = int(self.p.size) if self.p.size else 1

        if go_long == 1:
            parent = self.buy(exectype=bt.Order.Market, size=size)
            if np.isfinite(sl):
                self.sell(exectype=bt.Order.Stop, price=sl, size=size, parent=parent)
            if np.isfinite(tp):
                self.sell(exectype=bt.Order.Limit, price=tp, size=size, parent=parent)
        elif go_short == 1:
            parent = self.sell(exectype=bt.Order.Market, size=size)
            if np.isfinite(sl):
                self.buy(exectype=bt.Order.Stop, price=sl, size=size, parent=parent)
            if np.isfinite(tp):
                self.buy(exectype=bt.Order.Limit, price=tp, size=size, parent=parent)


# -----------------------------
# Strategy param introspection
# -----------------------------
def bt_param_names(cls) -> Set[str]:
    """
    Extract backtrader Strategy params names as strings.
    Supports tuple/list of tuples and dict forms.
    """
    names: Set[str] = set()
    ps = getattr(cls, 'params', ())
    if isinstance(ps, dict):
        names = set(map(str, ps.keys()))
    else:
        # ps can be a tuple/list like: (('param', default), 'param2', ...)
        for p in ps if isinstance(ps, Iterable) else ():
            if isinstance(p, tuple) and p:
                names.add(str(p[0]))
            else:
                names.add(str(p))
    # backtrader also collects params from bases; try to read _getparams if present
    try:
        for p in getattr(cls, '_getparams', lambda: [])():
            names.add(str(p[0]))
    except Exception:
        pass
    return names


# -----------------------------
# Signals
# -----------------------------
def build_signals_safe(cfg, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calls exec.pipeline.build_signals with `cfg` (dataclass). If the pipeline expects
    a dict and tries cfg.copy(), we retry with cfg.asdict().
    """
    try:
        from .exec.pipeline import build_signals  # (cfg, df) -> df_signals
        try:
            return build_signals(cfg, market_df)
        except Exception as e1:
            # Retry with asdict if available
            try:
                cfgd = cfg.asdict()
                return build_signals(cfgd, market_df)
            except Exception as e2:
                print(f"[runner] pipeline/build_signals error: {e1}")
                return pd.DataFrame(index=market_df.index, data={
                    "entry": 0.0, "dir": 0.0, "sig_long": 0.0, "sig_short": 0.0, "sl": np.nan, "tp": np.nan
                })
    except Exception as e:
        print(f"[runner] pipeline/build_signals error: {e}")
        return pd.DataFrame(index=market_df.index, data={
            "entry": 0.0, "dir": 0.0, "sig_long": 0.0, "sig_short": 0.0, "sl": np.nan, "tp": np.nan
        })


# -----------------------------
# Backtrader feed
# -----------------------------
class PandasOHLC(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
    )


# -----------------------------
# Main
# -----------------------------
def run():
    cfg = get_config()
    sym = cfg.symbol
    print(f"Downloading {cfg.timeframe} data from {cfg.start} to {cfg.end} for {sym}...")

    # Filter kwargs to whatever prepare_market_df currently accepts
    full_kwargs = dict(
        symbol=cfg.symbol,
        start=cfg.start,
        end=cfg.end,
        timeframe=cfg.timeframe,
        tz_local=cfg.tz,
        use_cache=getattr(cfg.io, "use_cache", True),
        debug=getattr(cfg, "debug", False),
    )
    sig = inspect.signature(prepare_market_df)
    filtered_kwargs = {k: v for k, v in full_kwargs.items() if k in sig.parameters}

    # 1) Prepare OHLCV
    market_df = prepare_market_df(**filtered_kwargs)
    market_df = ensure_dt_index(market_df, cfg.tz)

    market_df = market_df.rename(columns={c: str(c).lower() for c in market_df.columns})
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in market_df.columns]
    if missing:
        raise ValueError(f"Prepared data missing columns: {missing}")
    market_df = market_df[needed].copy()

    # 2) Signals (ICT pipeline)
    df_sig = build_signals_safe(cfg, market_df)
    present = list(df_sig.columns)
    print(f"[runner] signal columns present: {present}")
    nonzero = df_sig[(df_sig.get("entry", 0) > 0) &
                     ((df_sig.get("sig_long", 0) > 0) | (df_sig.get("sig_short", 0) > 0))]
    print(f"[runner] found {len(nonzero)} signal rows")
    if len(nonzero) > 0:
        try:
            print(nonzero.head(10)[["entry", "dir", "sig_long", "sig_short", "sl", "tp"]])
        except Exception:
            pass

    # Align signals to market index
    align_cols = ["entry", "sig_long", "sig_short", "sl", "tp"]
    for c in align_cols:
        if c not in df_sig.columns:
            df_sig[c] = 0.0 if c not in ("sl", "tp") else np.nan
    df_sig = df_sig.reindex(market_df.index)
    df_sig[["entry", "sig_long", "sig_short"]] = df_sig[["entry", "sig_long", "sig_short"]].fillna(0.0)
    df_sig = df_sig[align_cols].copy()

    # 3) Backtrader
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cfg.broker.start_cash)
    cerebro.broker.setcommission(commission=cfg.broker.commission_per_contract, percabs=True)

    data = PandasOHLC(dataname=market_df)
    cerebro.adddata(data)

    used_fallback = False
    try:
        from .exec.bt_strategy import ICTStrategy  # your strategy (if present)
        # Build kwargs only with params ICTStrategy actually supports
        param_names = bt_param_names(ICTStrategy)

        candidate = {
            "signals_df": df_sig,
            "entry": df_sig["entry"].to_numpy(),
            "sig_long": df_sig["sig_long"].to_numpy(),
            "sig_short": df_sig["sig_short"].to_numpy(),
            "sl": df_sig["sl"].to_numpy(),
            "tp": df_sig["tp"].to_numpy(),
            "size": max(1, int(getattr(cfg.risk, "fixed_size", 1))),
        }
        kwargs = {k: v for k, v in candidate.items() if k in param_names}

        if kwargs:
            cerebro.addstrategy(ICTStrategy, **kwargs)
        else:
            # If we can't pass anything meaningful, fallback
            used_fallback = True
    except Exception:
        used_fallback = True

    if used_fallback:
        cerebro.addstrategy(
            MinimalSignalStrategy,
            entry=df_sig["entry"].to_numpy(),
            sig_long=df_sig["sig_long"].to_numpy(),
            sig_short=df_sig["sig_short"].to_numpy(),
            sl=df_sig["sl"].to_numpy(),
            tp=df_sig["tp"].to_numpy(),
            size=max(1, int(getattr(cfg.risk, "fixed_size", 1))),
        )

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')

    print(f"Initial Portfolio Value: {cerebro.broker.getvalue():,.2f}")
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    print(f"Final Portfolio Value: {final_value:,.2f}")

    # 4) Export logs (fallback strat)
    trades_df, tx_df = None, None
    if isinstance(strat, MinimalSignalStrategy):
        trades_df = pd.DataFrame(strat.trades_log)
        tx_df = pd.DataFrame(strat.tx_log)

    xls_path = export_to_excel(
        cfg.io.report_dir,
        cfg.symbol,
        cfg.start,
        cfg.end,
        trades_df,
        tx_df
    )
    print(f"Saved Excel report → {xls_path}")

    # 5) Stats
    ta = strat.analyzers.ta.get_analysis() if hasattr(strat, 'analyzers') else {}
    dd = strat.analyzers.dd.get_analysis() if hasattr(strat, 'analyzers') else {}

    total_trades = ta.get('total', {}).get('total', 0) if ta else (0 if trades_df is None else len(trades_df))
    won = ta.get('won', {}).get('total', 0) if ta else np.nan
    lost = ta.get('lost', {}).get('total', 0) if ta else np.nan
    winrate = (won / total_trades * 100.0) if total_trades else 0.0
    maxdd_pct = dd.get('max', {}).get('drawdown', 0.0) if dd else 0.0

    print(f"Trades {total_trades} | Wins {won} | Losses {lost} | WinRate {winrate:.2f}%")
    print(f"Max Drawdown: {maxdd_pct:.2f}%")


def main():
    try:
        run()
    except Exception as e:
        print(f"[runner] fatal: {e}")
        raise


if __name__ == "__main__":
    main()
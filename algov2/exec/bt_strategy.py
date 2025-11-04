# algov2/exec/bt_strategy.py
from __future__ import annotations

import backtrader as bt
import pandas as pd
from datetime import datetime, timezone

class SignalExecutor(bt.Strategy):
    params = dict(
        signals_df=None,   # DataFrame with index == bar datetimes
        size=1,            # contracts/shares per trade
    )

    def __init__(self):
        if self.p.signals_df is None or len(self.p.signals_df) == 0:
            self.signals = pd.DataFrame(columns=["entry","dir","sl","tp"])
        else:
            sigs = self.p.signals_df.copy()
            # safety: ensure required cols
            for c in ["entry", "dir", "sl", "tp"]:
                if c not in sigs.columns:
                    sigs[c] = 0 if c in ("entry","dir") else float("nan")

            # normalize index tz to UTC to match bt datetimes
            if isinstance(sigs.index, pd.DatetimeIndex):
                if sigs.index.tz is None:
                    sigs.index = sigs.index.tz_localize("UTC")
                else:
                    sigs.index = sigs.index.tz_convert("UTC")
            self.signals = sigs[sigs["entry"] == 1]

        self._open_orders = []
        self._fired_idx = set()

    def log(self, txt):
        dt = bt.num2date(self.data.datetime[0], tz=timezone.utc)
        print(f"{dt:%Y-%m-%d %H:%M:%S} - {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            side = "BUY" if order.isbuy() else "SELL"
            self.log(f"ORDER FILLED {side} @ {order.executed.price:.2f} | Size {order.executed.size:+g}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("ORDER Canceled/Rejected")
        # keep track
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            if order in self._open_orders:
                self._open_orders.remove(order)

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl = trade.pnlcomm
            self.log(f"TRADE CLOSED | Net {pnl:.2f}")

    def _dt_utc(self):
        return bt.num2date(self.data.datetime[0], tz=timezone.utc).replace(tzinfo=timezone.utc)

    def next(self):
        # skip if already in a position or have open orders
        if self.position.size != 0 or len(self._open_orders) > 0:
            return

        dt = self._dt_utc()

        # exact match on bar timestamp
        if not isinstance(self.signals.index, pd.DatetimeIndex):
            return
        if dt not in self.signals.index:
            return
        if dt in self._fired_idx:
            return

        row = self.signals.loc[dt]
        if isinstance(row, pd.DataFrame):
            # multiple rows same timestamp -> take first
            row = row.iloc[0]

        direction = int(row.get("dir", 0))
        if direction not in (1, -1):
            return

        sl = row.get("sl", float("nan"))
        tp = row.get("tp", float("nan"))
        size = int(self.p.size)

        # place bracket
        if direction == 1:
            o = self.buy()  # market
            self._open_orders.append(o)
            if pd.notna(sl):
                so = self.sell(exectype=bt.Order.Stop, price=float(sl), size=size)
                self._open_orders.append(so)
            if pd.notna(tp):
                lo = self.sell(exectype=bt.Order.Limit, price=float(tp), size=size)
                self._open_orders.append(lo)
        else:
            o = self.sell()  # market
            self._open_orders.append(o)
            if pd.notna(sl):
                so = self.buy(exectype=bt.Order.Stop, price=float(sl), size=size)
                self._open_orders.append(so)
            if pd.notna(tp):
                lo = self.buy(exectype=bt.Order.Limit, price=float(tp), size=size)
                self._open_orders.append(lo)

        self._fired_idx.add(dt)
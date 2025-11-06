# algov2/exec/bt_strategy.py
from __future__ import annotations
import math
import backtrader as bt

class SignalData(bt.feeds.PandasData):
    """
    Extends PandasData to expose strategy signal columns produced by the pipeline:
      - entry: 1 when we want to act on this bar, else 0/NaN
      - dir: +1 for long, -1 for short
      - sl: absolute stop-loss price for this signal
      - tp: absolute take-profit price for this signal
    """
    lines = ('entry', 'dir', 'sl', 'tp',)
    params = (
        ('datetime', None),
        ('open',    'open'),
        ('high',    'high'),
        ('low',     'low'),
        ('close',   'close'),
        ('volume',  'volume'),
        ('openinterest', None),

        # map custom lines
        ('entry', 'entry'),
        ('dir',   'dir'),
        ('sl',    'sl'),
        ('tp',    'tp'),
    )


class SignalExecutor(bt.Strategy):
    """
    Minimal executor:
      - On bars where data.entry == 1, and no position/orders -> submit bracket
      - Uses data.dir to choose long/short
      - Uses data.sl / data.tp for stop/limit (falls back to default offsets if NaN)
      - Ensures one active trade at a time (per data)
    """
    params = dict(
        size=1,                # contracts per trade
        fallback_sl_points=20, # if sl is NaN: 20 points
        fallback_tp_points=30, # if tp is NaN: 30 points
        log=True,
    )

    def __init__(self):
        self.o_main = None
        self.o_sl   = None
        self.o_tp   = None

    def log(self, txt):
        if self.p.log:
            dt = self.datas[0].datetime.datetime(0)
            print(f'{dt:%Y-%m-%d %H:%M:%S} - {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'ORDER FILLED BUY @ {order.executed.price:.2f} | Size {order.executed.size:+g}')
            else:
                self.log(f'ORDER FILLED SELL @ {order.executed.price:.2f} | Size {order.executed.size:+g}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('ORDER Canceled/Margin/Rejected')

        # if the main is done (filled or canceled), reset pointers when all siblings are resolved
        if order is self.o_main and order.status in [order.Completed, order.Canceled, order.Rejected]:
            pass  # siblings may still be alive; we reset in notify_trade when flat again

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        pnl = trade.pnlcomm
        self.log(f'TRADE CLOSED | Net {pnl:.2f}')
        # reset order refs when flat
        self.o_main, self.o_sl, self.o_tp = None, None, None

    def next(self):
        d = self.datas[0]

        # Avoid duplicate submissions while orders are alive
        if any(o for o in (self.o_main, self.o_sl, self.o_tp) if o and o.status in [o.Submitted, o.Accepted]):
            return

        # Only trade when flat
        if self.position.size != 0:
            return

        # Read signal columns
        entry = d.entry[0]
        dirv  = d.dir[0]
        slval = d.sl[0]
        tpval = d.tp[0]

        # Ensure they are usable numbers
        def _isnum(x):
            try:
                return (x is not None) and (not math.isnan(x))
            except TypeError:
                return False

        if not _isnum(entry) or entry < 0.5:
            return
        if not _isnum(dirv) or (dirv not in (-1.0, 1.0)):
            return

        close = d.close[0]

        # Fallback SL/TP if missing
        if not _isnum(slval):
            slval = close - self.p.fallback_sl_points if dirv > 0 else close + self.p.fallback_sl_points
        if not _isnum(tpval):
            tpval = close + self.p.fallback_tp_points if dirv > 0 else close - self.p.fallback_tp_points

        size = self.p.size

        if dirv > 0:
            # LONG
            self.o_main, self.o_sl, self.o_tp = self.buy_bracket(
                size=size,
                exectype=bt.Order.Market,
                stopprice=slval,
                limitprice=tpval,
            )
            self.log(f'SIGNAL LONG -> bracket @mkt | SL={slval:.2f} TP={tpval:.2f}')
        else:
            # SHORT
            self.o_main, self.o_sl, self.o_tp = self.sell_bracket(
                size=size,
                exectype=bt.Order.Market,
                stopprice=slval,
                limitprice=tpval,
            )
            self.log(f'SIGNAL SHORT -> bracket @mkt | SL={slval:.2f} TP={tpval:.2f}')
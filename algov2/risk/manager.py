# algov2/exec/manager.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class RiskParams:
    """
    Risk configuration used to turn candidate signals into executable sizes.

    risk_mode:
        - "fixed_contracts": always use fixed_size
        - "fixed_dollars":   allocate up to risk_dollars per trade
        - "percent_equity":  allocate up to risk_pct * start_cash per trade (static pre-sizing)

    Notes:
      * For true dynamic sizing by evolving equity, do it *inside* the broker/strategy.
        Here we pre-size statically to keep pipeline deterministic.
    """
    risk_mode: str = "fixed_contracts"   # "fixed_contracts" | "fixed_dollars" | "percent_equity"
    fixed_size: int = 1                  # used when risk_mode == fixed_contracts
    risk_dollars: float = 0.0            # used when risk_mode == fixed_dollars
    risk_pct: float = 0.0                # used when risk_mode == percent_equity (e.g., 0.005 for 0.5%)
    start_cash: float = 100_000.0        # base for percent_equity
    max_contracts: int = 10              # hard cap
    min_rr: float = 1.0                  # drop signals with RR below this
    min_stop_ticks: float = 0.25         # prevent ultra-tight stops (in *price* units if you don’t use ticks)
    point_value: float = 20.0            # $ per index point (NQ ~ $20/point)
    prefer_tp: str = "tp1"               # which TP to use when checking RR ("tp1"|"tp2"|"best")


@dataclass
class Filters:
    """
    Optional filters beyond RR/risk:
    """
    allow_sessions: Optional[Sequence[str]] = None     # e.g., ["LONDON", "NYAM", "NYPM"]
    require_bias_align: bool = False                   # drop trades that conflict with 'bias' column if present
    premium_discount_col: Optional[str] = None         # ex: "pd_zone" with values {"premium","discount",None}
    enforce_pd_match: bool = False                     # if True, LONG must be "discount", SHORT must be "premium"
    one_signal_per_bar: bool = True                    # if True, keep best RR per timestamp


class SignalManager:
    """
    Orchestrates the *final mile* from candidate signals -> execution-ready signals.

    Expected input schema (minimum):
        columns: ['open_dt','direction','entry','sl','tp1','tp2']
        optional: ['session','confs','comment','bias','pd_zone','rr_hint', ...]

    Output:
        input columns + ['size','rr1','rr2','rr_pick'] filtered & deduped.
    """

    def __init__(self, risk: RiskParams, flt: Optional[Filters] = None):
        self.risk = risk
        self.flt = flt or Filters()

    # ------------- public API -------------

    def finalize(
        self,
        signals: pd.DataFrame,
        data_ref: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        - sanitize columns/types
        - compute RR (rr1/rr2)
        - filter by RR + sessions + bias + premium/discount
        - compute size per risk params
        - drop untradeable (size <= 0 or NaN)
        - dedupe per bar if requested
        """
        if signals is None or len(signals) == 0:
            return self._empty_like(signals)

        df = signals.copy()

        # normalize time and required fields
        df = self._normalize(df)

        # RR calc
        df = self._attach_rr(df)

        # RR filter
        df = df[df["rr_pick"] >= self.risk.min_rr].copy()

        # session filter (optional)
        if self.flt.allow_sessions:
            df = df[df["session"].isin(self.flt.allow_sessions)].copy()

        # bias alignment (optional)
        if self.flt.require_bias_align and "bias" in df.columns:
            df = df[self._bias_ok(df)].copy()

        # premium/discount enforcement (optional)
        if self.flt.enforce_pd_match and (self.flt.premium_discount_col or "pd_zone" in df.columns):
            pd_col = self.flt.premium_discount_col or "pd_zone"
            if pd_col in df.columns:
                df = df[self._pd_ok(df, pd_col)].copy()

        if df.empty:
            return self._empty_like(signals)

        # risk-based sizing
        df["size"] = self._compute_size_vectorized(df)

        # drop zeros or NaNs
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["size"])
        df = df[df["size"] > 0].copy()

        if df.empty:
            return self._empty_like(signals)

        # enforce minimal stop width (price units)
        stop_w = (df["entry"] - df["sl"]).abs()
        df = df[stop_w >= max(self.risk.min_stop_ticks, 1e-8)].copy()

        if df.empty:
            return self._empty_like(signals)

        # pick one per bar (keep best RR)
        if self.flt.one_signal_per_bar:
            df = self._dedupe_per_bar_keep_best_rr(df)

        # nice ordering
        keep_cols = self._order_columns(df.columns)
        df = df[keep_cols].sort_values("open_dt").reset_index(drop=True)
        return df

    # ------------- internals -------------

    @staticmethod
    def _empty_like(signals: Optional[pd.DataFrame]) -> pd.DataFrame:
        cols = [
            "open_dt", "direction", "size", "entry", "sl", "tp1", "tp2",
            "session", "confs", "comment", "rr1", "rr2", "rr_pick",
        ]
        if signals is not None:
            # preserve any extra columns (bias, pd_zone, etc.) even if empty
            for c in signals.columns:
                if c not in cols:
                    cols.append(c)
        return pd.DataFrame(columns=cols)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # time column
        if "open_dt" in df.columns:
            df["open_dt"] = pd.to_datetime(df["open_dt"]).dt.tz_localize(None)
        else:
            df["open_dt"] = pd.to_datetime(df.index).tz_localize(None)

        # required numeric fields
        for c in ("entry", "sl"):
            if c not in df.columns:
                raise ValueError(f"signals missing required column '{c}'")
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # optional tps
        for c in ("tp1", "tp2"):
            if c not in df.columns:
                df[c] = np.nan
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # direction/session/confs/comment
        df["direction"] = df["direction"].astype(str).str.upper()
        if "session" not in df.columns:
            df["session"] = None
        if "confs" not in df.columns:
            df["confs"] = None
        if "comment" not in df.columns:
            df["comment"] = None

        return df

    def _attach_rr(self, df: pd.DataFrame) -> pd.DataFrame:
        # distance to stop in points
        stop = (df["entry"] - df["sl"]).abs().replace(0.0, np.nan)

        # RR for tp1/tp2 depending on direction
        # LONG: (tp - entry) / (entry - sl)
        # SHORT: (entry - tp) / (sl - entry)
        is_long = df["direction"].eq("LONG")

        rr1 = np.where(
            is_long,
            (df["tp1"] - df["entry"]) / (df["entry"] - df["sl"]),
            (df["entry"] - df["tp1"]) / (df["sl"] - df["entry"]),
        )
        rr2 = np.where(
            is_long,
            (df["tp2"] - df["entry"]) / (df["entry"] - df["sl"]),
            (df["entry"] - df["tp2"]) / (df["sl"] - df["entry"]),
        )

        # mask out where tp is nan
        rr1 = np.where(np.isnan(df["tp1"]), np.nan, rr1)
        rr2 = np.where(np.isnan(df["tp2"]), np.nan, rr2)

        # avoid divide-by-zero from earlier step
        rr1 = np.where(stop.isna(), np.nan, rr1)
        rr2 = np.where(stop.isna(), np.nan, rr2)

        df = df.copy()
        df["rr1"] = rr1
        df["rr2"] = rr2
        df["rr_pick"] = self._pick_rr(df)
        return df

    def _pick_rr(self, df: pd.DataFrame) -> np.ndarray:
        if self.risk.prefer_tp == "tp1":
            return df["rr1"].fillna(df["rr2"]).values
        if self.risk.prefer_tp == "tp2":
            return df["rr2"].fillna(df["rr1"]).values
        # "best"
        rr = np.vstack([df["rr1"].fillna(-np.inf).values,
                        df["rr2"].fillna(-np.inf).values])
        return np.nanmax(rr, axis=0)

    def _bias_ok(self, df: pd.DataFrame) -> pd.Series:
        # expecting 'bias' as 'BULL'|'BEAR' or 'LONG'|'SHORT'
        bias = df["bias"].astype(str).str.upper()
        want_long = bias.isin(["BULL", "LONG"])
        want_short = bias.isin(["BEAR", "SHORT"])
        is_long = df["direction"].eq("LONG")
        return (is_long & want_long) | (~is_long & want_short)

    def _pd_ok(self, df: pd.DataFrame, pd_col: str) -> pd.Series:
        # premium/discount: LONG -> discount, SHORT -> premium
        zone = df[pd_col].astype("category")
        is_long = df["direction"].eq("LONG")
        return (is_long & zone.eq("discount")) | (~is_long & zone.eq("premium"))

    def _compute_size_vectorized(self, df: pd.DataFrame) -> pd.Series:
        mode = self.risk.risk_mode
        max_ct = max(int(self.risk.max_contracts), 1)
        pv = float(self.risk.point_value)

        # price risk per contract
        risk_pts = (df["entry"] - df["sl"]).abs().replace(0.0, np.nan)
        risk_usd = risk_pts * pv

        if mode == "fixed_contracts":
            size = pd.Series(self.risk.fixed_size, index=df.index, dtype="float64")
        elif mode == "fixed_dollars":
            risk_d = max(float(self.risk.risk_dollars), 0.0)
            size = np.floor(risk_d / risk_usd)
        elif mode == "percent_equity":
            eff_d = max(float(self.risk.start_cash) * float(self.risk.risk_pct), 0.0)
            size = np.floor(eff_d / risk_usd)
        else:
            raise ValueError(f"Unknown risk_mode: {mode}")

        # sanitize
        size = size.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        size = size.clip(lower=0.0, upper=float(max_ct))
        # convert to int, enforce at least 1 if >0
        size = size.astype(int)
        size[size < 1] = 0
        return size

    @staticmethod
    def _dedupe_per_bar_keep_best_rr(df: pd.DataFrame) -> pd.DataFrame:
        # in case multiple signals share the exact same open_dt, keep the one with highest rr_pick
        df = df.sort_values(["open_dt", "rr_pick"], ascending=[True, False]).copy()
        return df.drop_duplicates(subset=["open_dt"], keep="first")

    @staticmethod
    def _order_columns(cols: Iterable[str]) -> list[str]:
        front = [
            "open_dt", "session", "direction", "size",
            "entry", "sl", "tp1", "tp2",
            "rr1", "rr2", "rr_pick",
            "confs", "comment",
        ]
        tail = [c for c in cols if c not in front]
        return front + tail
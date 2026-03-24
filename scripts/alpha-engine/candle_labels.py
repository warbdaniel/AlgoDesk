"""
Historical 5M Candle ML Pipeline - Label Engine
================================================

Generates supervised ML labels from historical 5-minute OHLCV candles.
All labels are forward-looking (computed from future candles relative
to the observation point).

Label types:
  - Forward returns:  raw return at multiple horizons (1, 3, 6, 12, 24 candles)
  - Direction:        ternary classification (LONG / FLAT / SHORT)
  - Triple barrier:   first barrier hit (TP / SL / TIME_EXPIRY)
  - Max drawdown:     worst adverse move within horizon
  - Volatility label: forward realised vol (useful for vol-targeting)

Design:
  - Batch-oriented: operates on a full candle list (no streaming)
  - No look-ahead bias: labels at index i only use candles[i+1:]
  - Returns None for candles near the end where horizon is incomplete
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, fields
from enum import IntEnum

from candle_config import CandleLabelConfig, CANDLE_SYMBOLS

logger = logging.getLogger("candle_labels")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class CandleDirection(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class CandleBarrierHit(IntEnum):
    STOP_LOSS = -1
    TIME_EXPIRY = 0
    TAKE_PROFIT = 1


# ---------------------------------------------------------------------------
# Label dataclass
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CandleLabel:
    """Complete label set for a single 5M candle observation."""

    symbol: str = ""
    ts: float = 0.0                  # candle open_time (matches feature ts)

    # Forward returns (close-to-close, in price units)
    fwd_ret_1: float = 0.0          # 1 candle  = 5 min
    fwd_ret_3: float = 0.0          # 3 candles = 15 min
    fwd_ret_6: float = 0.0          # 6 candles = 30 min
    fwd_ret_12: float = 0.0         # 12 candles = 1 hour
    fwd_ret_24: float = 0.0         # 24 candles = 2 hours

    # Forward log returns
    fwd_logret_1: float = 0.0
    fwd_logret_3: float = 0.0
    fwd_logret_6: float = 0.0
    fwd_logret_12: float = 0.0
    fwd_logret_24: float = 0.0

    # Ternary direction labels
    dir_1: float = 0.0              # CandleDirection value
    dir_3: float = 0.0
    dir_6: float = 0.0
    dir_12: float = 0.0
    dir_24: float = 0.0

    # Triple-barrier labels
    barrier_label: float = 0.0      # CandleBarrierHit value
    barrier_return: float = 0.0     # return at barrier hit
    barrier_candles: float = 0.0    # candles until barrier hit

    # Risk labels
    max_adverse_excursion: float = 0.0   # worst drawdown in horizon
    max_favorable_excursion: float = 0.0 # best run-up in horizon
    fwd_volatility: float = 0.0          # forward realised vol

    def to_dict(self) -> dict[str, float]:
        """Return all label values (exclude symbol/ts)."""
        result = {}
        for f in fields(self):
            if f.name in ("symbol", "ts"):
                continue
            result[f.name] = getattr(self, f.name)
        return result


# ---------------------------------------------------------------------------
# Label engine
# ---------------------------------------------------------------------------
class CandleLabelEngine:
    """Generates labels from a list of historical 5M candles.

    Each candle dict must have keys: open, high, low, close, open_time.
    The list must be sorted by open_time ascending.
    """

    def __init__(self, config: CandleLabelConfig | None = None):
        self._cfg = config or CandleLabelConfig()

    def label_all(
        self,
        candles: list[dict],
        symbol: str = "",
    ) -> list[CandleLabel]:
        """Generate labels for every candle where sufficient future data exists.

        Candles near the end of the list (within max horizon) are skipped.
        """
        n = len(candles)
        max_horizon = max(
            max(self._cfg.forward_horizons),
            self._cfg.barrier_max_candles,
            self._cfg.max_drawdown_horizon,
        )

        if n < max_horizon + 1:
            logger.warning(
                "Need > %d candles for labels, got %d", max_horizon, n
            )
            return []

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        timestamps = [c.get("open_time", 0.0) for c in candles]

        sym = symbol or candles[0].get("symbol", "")
        pip_size = CANDLE_SYMBOLS.get(sym, {}).get("pip_size", 0.0001)

        long_thresh = self._cfg.long_threshold_pips * pip_size
        short_thresh = self._cfg.short_threshold_pips * pip_size
        tp_dist = self._cfg.barrier_tp_pips * pip_size
        sl_dist = self._cfg.barrier_sl_pips * pip_size

        labels: list[CandleLabel] = []

        for i in range(n - max_horizon):
            entry_close = closes[i]
            if entry_close <= 0:
                continue

            lbl = CandleLabel(symbol=sym, ts=timestamps[i])

            # ── Forward returns ───────────────────────────────
            for h in self._cfg.forward_horizons:
                if i + h >= n:
                    continue
                fwd_close = closes[i + h]
                raw_ret = fwd_close - entry_close
                log_ret = math.log(fwd_close / entry_close) if fwd_close > 0 else 0.0

                setattr(lbl, f"fwd_ret_{h}", raw_ret)
                setattr(lbl, f"fwd_logret_{h}", log_ret)

                # Ternary direction
                if raw_ret > long_thresh:
                    direction = CandleDirection.LONG
                elif raw_ret < -short_thresh:
                    direction = CandleDirection.SHORT
                else:
                    direction = CandleDirection.FLAT
                setattr(lbl, f"dir_{h}", float(direction))

            # ── Triple barrier ────────────────────────────────
            barrier_label, barrier_ret, barrier_dur = self._triple_barrier(
                closes, highs, lows, i, entry_close, tp_dist, sl_dist,
            )
            lbl.barrier_label = float(barrier_label)
            lbl.barrier_return = barrier_ret
            lbl.barrier_candles = float(barrier_dur)

            # ── Risk labels ───────────────────────────────────
            dd_horizon = min(self._cfg.max_drawdown_horizon, n - i - 1)
            if dd_horizon > 0:
                future_highs = highs[i + 1: i + 1 + dd_horizon]
                future_lows = lows[i + 1: i + 1 + dd_horizon]
                lbl.max_favorable_excursion = max(future_highs) - entry_close
                lbl.max_adverse_excursion = entry_close - min(future_lows)

                # Forward realised volatility
                future_closes = closes[i + 1: i + 1 + dd_horizon]
                if len(future_closes) >= 2:
                    log_rets = []
                    prev = entry_close
                    for fc in future_closes:
                        if fc > 0 and prev > 0:
                            log_rets.append(math.log(fc / prev))
                        prev = fc
                    if log_rets:
                        mean_r = sum(log_rets) / len(log_rets)
                        var_r = sum((r - mean_r) ** 2 for r in log_rets) / len(log_rets)
                        lbl.fwd_volatility = math.sqrt(var_r)

            labels.append(lbl)

        return labels

    def _triple_barrier(
        self,
        closes: list[float],
        highs: list[float],
        lows: list[float],
        idx: int,
        entry_price: float,
        tp_dist: float,
        sl_dist: float,
    ) -> tuple[int, float, int]:
        """Evaluate triple barrier from candle idx.

        Returns (barrier_hit, return_at_hit, candles_to_hit).
        Uses high/low within each candle to check intra-candle barrier hits.
        """
        max_candles = self._cfg.barrier_max_candles
        n = len(closes)

        for j in range(1, max_candles + 1):
            ci = idx + j
            if ci >= n:
                break

            # Check TP first (long-biased default)
            if highs[ci] - entry_price >= tp_dist:
                return (
                    int(CandleBarrierHit.TAKE_PROFIT),
                    tp_dist,
                    j,
                )

            # Check SL
            if entry_price - lows[ci] >= sl_dist:
                return (
                    int(CandleBarrierHit.STOP_LOSS),
                    -sl_dist,
                    j,
                )

        # Time expiry - use last available close
        last_idx = min(idx + max_candles, n - 1)
        expiry_ret = closes[last_idx] - entry_price
        return (int(CandleBarrierHit.TIME_EXPIRY), expiry_ret, max_candles)

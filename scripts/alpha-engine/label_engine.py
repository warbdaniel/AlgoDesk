"""
Project Alpha - ML Label Engine

Generates supervised-learning labels from tick data for scalping models.

Label types:
  1. Forward returns     – raw return over N-second horizons
  2. Triple barrier      – first-hit of TP / SL / time-expiry → {+1, -1, 0}
  3. Ternary direction   – discretised return into {LONG, SHORT, FLAT}
  4. Binary direction    – simplified {UP, DOWN} for each horizon

All labelling is strictly forward-looking: labels at time *t* are computed
from ticks arriving *after* t, so no look-ahead bias is possible when the
dataset builder aligns features(t) with labels(t).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from enum import IntEnum

from tick_buffer import AlphaTick
from config import LabelConfig


# ---------------------------------------------------------------------------
# Label enums
# ---------------------------------------------------------------------------
class BarrierHit(IntEnum):
    TAKE_PROFIT = 1
    STOP_LOSS = -1
    TIME_EXPIRY = 0


class Direction(IntEnum):
    LONG = 1
    FLAT = 0
    SHORT = -1


# ---------------------------------------------------------------------------
# Label output
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ScalpingLabel:
    """Complete label set for one observation point."""

    symbol: str = ""
    entry_ts: float = 0.0            # timestamp of the observation
    entry_mid: float = 0.0           # mid price at observation

    # Forward returns (keyed by horizon in seconds)
    fwd_return_5s: float = math.nan
    fwd_return_10s: float = math.nan
    fwd_return_30s: float = math.nan
    fwd_return_60s: float = math.nan
    fwd_return_120s: float = math.nan

    # Triple-barrier result
    barrier_label: int = 0            # +1 TP, -1 SL, 0 time-expiry
    barrier_return: float = 0.0       # realised return at barrier hit
    barrier_duration_sec: float = 0.0 # time to barrier hit
    barrier_exit_ts: float = 0.0

    # Ternary direction labels (per horizon)
    dir_5s: int = 0
    dir_10s: int = 0
    dir_30s: int = 0
    dir_60s: int = 0
    dir_120s: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_ts": self.entry_ts,
            "entry_mid": self.entry_mid,
            "fwd_return_5s": self.fwd_return_5s,
            "fwd_return_10s": self.fwd_return_10s,
            "fwd_return_30s": self.fwd_return_30s,
            "fwd_return_60s": self.fwd_return_60s,
            "fwd_return_120s": self.fwd_return_120s,
            "barrier_label": self.barrier_label,
            "barrier_return": self.barrier_return,
            "barrier_duration_sec": self.barrier_duration_sec,
            "barrier_exit_ts": self.barrier_exit_ts,
            "dir_5s": self.dir_5s,
            "dir_10s": self.dir_10s,
            "dir_30s": self.dir_30s,
            "dir_60s": self.dir_60s,
            "dir_120s": self.dir_120s,
        }

    @staticmethod
    def label_names() -> list[str]:
        return [
            "fwd_return_5s", "fwd_return_10s", "fwd_return_30s",
            "fwd_return_60s", "fwd_return_120s",
            "barrier_label", "barrier_return", "barrier_duration_sec",
            "dir_5s", "dir_10s", "dir_30s", "dir_60s", "dir_120s",
        ]


# ---------------------------------------------------------------------------
# Label engine
# ---------------------------------------------------------------------------
class LabelEngine:
    """Generates labels from a sorted list of ticks.

    Designed for offline/batch label generation from historical tick data.
    The ticks list must be sorted ascending by timestamp.
    """

    def __init__(self, config: LabelConfig | None = None, pip_size: float = 0.0001):
        self._cfg = config or LabelConfig()
        self._pip = pip_size

    # ---- public API -------------------------------------------------------

    def label_tick(
        self,
        idx: int,
        ticks: list[AlphaTick],
    ) -> ScalpingLabel | None:
        """Compute labels for the tick at *idx* using future ticks.

        Returns None if there are insufficient future ticks to compute
        at least the shortest-horizon return.
        """
        if idx >= len(ticks) - self._cfg.min_ticks_in_horizon:
            return None

        entry = ticks[idx]
        entry_mid = entry.mid
        if entry_mid == 0:
            return None

        label = ScalpingLabel(
            symbol=entry.symbol,
            entry_ts=entry.ts,
            entry_mid=entry_mid,
        )

        # Build a timestamp index for fast horizon lookup
        future = ticks[idx + 1:]
        future_ts = [t.ts for t in future]

        # Forward returns + direction labels
        self._compute_forward_returns(label, entry_mid, entry.ts, future, future_ts)

        # Triple barrier
        self._compute_triple_barrier(label, entry_mid, entry.ts, future)

        return label

    def label_batch(
        self,
        ticks: list[AlphaTick],
        step: int = 1,
    ) -> list[ScalpingLabel]:
        """Label every *step*-th tick in the list.

        Args:
            ticks: Sorted tick list.
            step:  Stride between labelled ticks (1 = every tick).

        Returns list of ScalpingLabel for each valid observation.
        """
        labels: list[ScalpingLabel] = []
        for i in range(0, len(ticks), step):
            lbl = self.label_tick(i, ticks)
            if lbl is not None:
                labels.append(lbl)
        return labels

    # ---- internal ---------------------------------------------------------

    def _compute_forward_returns(
        self,
        label: ScalpingLabel,
        entry_mid: float,
        entry_ts: float,
        future: list[AlphaTick],
        future_ts: list[float],
    ) -> None:
        horizons = self._cfg.return_horizons_sec
        threshold_long = self._cfg.long_threshold_pips * self._pip
        threshold_short = self._cfg.short_threshold_pips * self._pip

        for h in horizons:
            target_ts = entry_ts + h
            # Find first tick at or after target_ts
            pos = bisect.bisect_left(future_ts, target_ts)

            if pos >= len(future):
                # Not enough future data for this horizon
                continue

            exit_mid = future[pos].mid
            ret = (exit_mid - entry_mid) / entry_mid if entry_mid != 0 else 0.0
            abs_change = exit_mid - entry_mid

            # Set return
            attr_ret = f"fwd_return_{int(h)}s"
            if hasattr(label, attr_ret):
                setattr(label, attr_ret, ret)

            # Set direction
            if abs_change >= threshold_long:
                direction = Direction.LONG
            elif abs_change <= -threshold_short:
                direction = Direction.SHORT
            else:
                direction = Direction.FLAT
            attr_dir = f"dir_{int(h)}s"
            if hasattr(label, attr_dir):
                setattr(label, attr_dir, int(direction))

    def _compute_triple_barrier(
        self,
        label: ScalpingLabel,
        entry_mid: float,
        entry_ts: float,
        future: list[AlphaTick],
    ) -> None:
        tp_dist = self._cfg.tp_pips * self._pip
        sl_dist = self._cfg.sl_pips * self._pip
        max_ts = entry_ts + self._cfg.max_holding_sec

        tp_level = entry_mid + tp_dist
        sl_level = entry_mid - sl_dist

        for tick in future:
            mid = tick.mid

            # Check take-profit (long bias by default; symmetric)
            if mid >= tp_level:
                label.barrier_label = int(BarrierHit.TAKE_PROFIT)
                label.barrier_return = (mid - entry_mid) / entry_mid
                label.barrier_duration_sec = tick.ts - entry_ts
                label.barrier_exit_ts = tick.ts
                return

            # Check stop-loss
            if mid <= sl_level:
                label.barrier_label = int(BarrierHit.STOP_LOSS)
                label.barrier_return = (mid - entry_mid) / entry_mid
                label.barrier_duration_sec = tick.ts - entry_ts
                label.barrier_exit_ts = tick.ts
                return

            # Check time expiry
            if tick.ts >= max_ts:
                label.barrier_label = int(BarrierHit.TIME_EXPIRY)
                label.barrier_return = (mid - entry_mid) / entry_mid
                label.barrier_duration_sec = tick.ts - entry_ts
                label.barrier_exit_ts = tick.ts
                return

        # Ran out of future data: mark as time-expiry with last available
        if future:
            last = future[-1]
            label.barrier_label = int(BarrierHit.TIME_EXPIRY)
            label.barrier_return = (last.mid - entry_mid) / entry_mid
            label.barrier_duration_sec = last.ts - entry_ts
            label.barrier_exit_ts = last.ts

"""
Project Alpha - Scalping Feature Engine

Computes tick-level microstructure features purpose-built for sub-minute
ML scalping signals.  Operates on the TickRing buffer for zero-copy,
allocation-free feature extraction.

Feature groups:
  1. Price dynamics    – mid-price returns, velocity, acceleration
  2. Spread analytics  – spread level, spread changes, spread z-score
  3. Order-flow        – bid/ask size imbalance, trade-flow toxicity proxy
  4. Microprice        – size-weighted fair value
  5. Volatility        – realized vol over tick windows
  6. Tick intensity    – tick arrival rate, inter-tick timing
  7. Level proximity   – distance to round numbers, recent highs/lows
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from tick_buffer import AlphaTick, TickRing, TickBufferManager
from config import FeatureConfig


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ScalpingFeatureVector:
    """Single observation of all scalping features at one tick."""

    symbol: str = ""
    ts: float = 0.0

    # --- Price dynamics (per window) ---
    mid: float = 0.0
    bid: float = 0.0
    ask: float = 0.0

    ret_ticks_10: float = 0.0
    ret_ticks_25: float = 0.0
    ret_ticks_50: float = 0.0
    ret_ticks_100: float = 0.0
    ret_ticks_250: float = 0.0

    velocity_5: float = 0.0       # mid change per tick (5-tick window)
    velocity_10: float = 0.0
    velocity_25: float = 0.0
    velocity_50: float = 0.0
    acceleration_10: float = 0.0  # velocity change

    # --- Spread analytics ---
    spread_raw: float = 0.0
    spread_ema: float = 0.0
    spread_zscore: float = 0.0    # (spread - mean) / std over window
    spread_change: float = 0.0    # spread delta from previous tick

    # --- Order-flow imbalance ---
    ofi: float = 0.0              # (bid_size - ask_size) / (bid_size + ask_size)
    ofi_ema: float = 0.0          # smoothed OFI
    cumulative_ofi_50: float = 0.0

    # --- Microprice ---
    microprice: float = 0.0       # ask_size*bid + bid_size*ask / total_size
    microprice_vs_mid: float = 0.0  # microprice - mid (signed)

    # --- Volatility ---
    realized_vol_25: float = 0.0
    realized_vol_50: float = 0.0
    realized_vol_100: float = 0.0
    high_low_range_50: float = 0.0
    high_low_range_100: float = 0.0

    # --- Tick intensity ---
    tick_rate_10s: float = 0.0    # ticks per second (last 10s)
    tick_rate_30s: float = 0.0
    tick_rate_60s: float = 0.0
    inter_tick_mean: float = 0.0  # mean inter-tick time (ms)
    inter_tick_std: float = 0.0

    # --- Level proximity ---
    dist_round_number: float = 0.0    # distance to nearest round price
    dist_session_high: float = 0.0
    dist_session_low: float = 0.0

    # ---- serialisation ----------------------------------------------------

    _NUMERIC_FIELDS: list[str] = field(
        default=None, init=False, repr=False,  # type: ignore[assignment]
    )

    def __post_init__(self) -> None:
        if ScalpingFeatureVector._NUMERIC_FIELDS is None:  # type: ignore[comparison-overlap]
            import dataclasses as _dc
            ScalpingFeatureVector._NUMERIC_FIELDS = [
                f.name for f in _dc.fields(self)
                if f.name not in ("symbol", "ts", "_NUMERIC_FIELDS")
            ]

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self._NUMERIC_FIELDS}  # type: ignore[union-attr]

    def to_ml_vector(self) -> list[float]:
        return [getattr(self, f) for f in self._NUMERIC_FIELDS]  # type: ignore[union-attr]

    @classmethod
    def feature_names(cls) -> list[str]:
        import dataclasses as _dc
        return [
            f.name for f in _dc.fields(cls)
            if f.name not in ("symbol", "ts", "_NUMERIC_FIELDS")
        ]


# ---------------------------------------------------------------------------
# Feature computation engine
# ---------------------------------------------------------------------------
class ScalpingFeatureEngine:
    """Stateful engine that computes ScalpingFeatureVectors from tick data."""

    def __init__(self, config: FeatureConfig | None = None):
        self._cfg = config or FeatureConfig()
        self._ofi_ema: dict[str, float] = {}   # per-symbol OFI EMA state
        self._ofi_alpha = 0.1                   # OFI EMA smoothing factor

    # ---- public API -------------------------------------------------------

    def compute(self, ring: TickRing, symbol: str) -> ScalpingFeatureVector | None:
        """Compute a full feature vector from the current state of a TickRing.

        Returns None if insufficient data (< 10 ticks).
        """
        ticks = ring.get_latest(max(self._cfg.tick_windows))
        if len(ticks) < 10:
            return None

        current = ticks[-1]
        fv = ScalpingFeatureVector(symbol=symbol, ts=current.ts)

        # Raw prices
        fv.mid = current.mid
        fv.bid = current.bid
        fv.ask = current.ask

        # Price dynamics
        self._compute_returns(fv, ticks)
        self._compute_velocity(fv, ticks)

        # Spread
        self._compute_spread(fv, ticks, ring)

        # Order flow
        self._compute_ofi(fv, ticks, symbol)

        # Microprice
        self._compute_microprice(fv, current)

        # Volatility
        self._compute_volatility(fv, ticks)

        # Tick intensity
        self._compute_tick_intensity(fv, ticks)

        # Level proximity
        self._compute_level_proximity(fv, ticks)

        return fv

    def compute_batch(
        self, buffer: TickBufferManager, symbols: list[str] | None = None,
    ) -> dict[str, ScalpingFeatureVector]:
        """Compute features for multiple symbols at once."""
        symbols = symbols or buffer.symbols
        results: dict[str, ScalpingFeatureVector] = {}
        for sym in symbols:
            ring = buffer.ring(sym)
            if ring is None:
                continue
            fv = self.compute(ring, sym)
            if fv is not None:
                results[sym] = fv
        return results

    # ---- internal ---------------------------------------------------------

    def _compute_returns(self, fv: ScalpingFeatureVector, ticks: list[AlphaTick]) -> None:
        cur_mid = ticks[-1].mid
        n = len(ticks)
        for w in self._cfg.tick_windows:
            if n >= w:
                prev_mid = ticks[-(w)].mid
                ret = (cur_mid - prev_mid) / prev_mid if prev_mid != 0 else 0.0
            else:
                ret = 0.0
            attr = f"ret_ticks_{w}"
            if hasattr(fv, attr):
                setattr(fv, attr, ret)

    def _compute_velocity(self, fv: ScalpingFeatureVector, ticks: list[AlphaTick]) -> None:
        n = len(ticks)
        for w in self._cfg.velocity_lookbacks:
            if n >= w + 1:
                mids = [t.mid for t in ticks[-(w + 1):]]
                velocity = (mids[-1] - mids[0]) / w
            else:
                velocity = 0.0
            attr = f"velocity_{w}"
            if hasattr(fv, attr):
                setattr(fv, attr, velocity)

        # acceleration = change in velocity
        if n >= 11:
            mids = [t.mid for t in ticks[-11:]]
            v_now = (mids[-1] - mids[-6]) / 5
            v_prev = (mids[-6] - mids[-11]) / 5
            fv.acceleration_10 = v_now - v_prev

    def _compute_spread(
        self, fv: ScalpingFeatureVector, ticks: list[AlphaTick], ring: TickRing,
    ) -> None:
        current = ticks[-1]
        fv.spread_raw = current.spread
        fv.spread_ema = ring.spread_ema

        # Spread z-score over last 100 ticks
        window = min(100, len(ticks))
        spreads = [t.spread for t in ticks[-window:]]
        mean_s = sum(spreads) / len(spreads)
        var_s = sum((s - mean_s) ** 2 for s in spreads) / len(spreads)
        std_s = math.sqrt(var_s) if var_s > 0 else 1e-10
        fv.spread_zscore = (current.spread - mean_s) / std_s

        # Spread change from prior tick
        if len(ticks) >= 2:
            fv.spread_change = current.spread - ticks[-2].spread

    def _compute_ofi(
        self, fv: ScalpingFeatureVector, ticks: list[AlphaTick], symbol: str,
    ) -> None:
        current = ticks[-1]
        total_size = current.bid_size + current.ask_size
        if total_size > 0:
            fv.ofi = (current.bid_size - current.ask_size) / total_size
        else:
            fv.ofi = 0.0

        # EMA-smoothed OFI
        prev_ofi = self._ofi_ema.get(symbol, 0.0)
        fv.ofi_ema = prev_ofi + self._ofi_alpha * (fv.ofi - prev_ofi)
        self._ofi_ema[symbol] = fv.ofi_ema

        # Cumulative OFI over lookback
        lookback = min(self._cfg.ofi_lookback, len(ticks))
        cum_ofi = 0.0
        for t in ticks[-lookback:]:
            ts = t.bid_size + t.ask_size
            if ts > 0:
                cum_ofi += (t.bid_size - t.ask_size) / ts
        fv.cumulative_ofi_50 = cum_ofi

    def _compute_microprice(self, fv: ScalpingFeatureVector, tick: AlphaTick) -> None:
        if not self._cfg.microprice_enabled:
            return
        total = tick.bid_size + tick.ask_size
        if total > 0:
            # Microprice: size-weighted mid
            fv.microprice = (tick.ask_size * tick.bid + tick.bid_size * tick.ask) / total
        else:
            fv.microprice = tick.mid
        fv.microprice_vs_mid = fv.microprice - tick.mid

    def _compute_volatility(self, fv: ScalpingFeatureVector, ticks: list[AlphaTick]) -> None:
        n = len(ticks)
        for w in self._cfg.volatility_lookbacks:
            window = min(w, n)
            if window < 3:
                continue
            mids = [t.mid for t in ticks[-window:]]
            # Realized vol = std of log returns
            log_rets = [
                math.log(mids[i] / mids[i - 1])
                for i in range(1, len(mids))
                if mids[i - 1] > 0
            ]
            if log_rets:
                mean_r = sum(log_rets) / len(log_rets)
                var_r = sum((r - mean_r) ** 2 for r in log_rets) / len(log_rets)
                vol = math.sqrt(var_r)
            else:
                vol = 0.0
            attr = f"realized_vol_{w}"
            if hasattr(fv, attr):
                setattr(fv, attr, vol)

        # High-low range
        for w in [50, 100]:
            window = min(w, n)
            if window < 2:
                continue
            mids = [t.mid for t in ticks[-window:]]
            attr = f"high_low_range_{w}"
            if hasattr(fv, attr):
                setattr(fv, attr, max(mids) - min(mids))

    def _compute_tick_intensity(self, fv: ScalpingFeatureVector, ticks: list[AlphaTick]) -> None:
        now = ticks[-1].ts

        for sec, attr in [(10, "tick_rate_10s"), (30, "tick_rate_30s"), (60, "tick_rate_60s")]:
            cutoff = now - sec
            count = sum(1 for t in ticks if t.ts >= cutoff)
            setattr(fv, attr, count / sec)

        # Inter-tick timing stats (last 50 ticks)
        recent = ticks[-min(50, len(ticks)):]
        if len(recent) >= 2:
            deltas = [
                (recent[i].ts - recent[i - 1].ts) * 1000  # ms
                for i in range(1, len(recent))
            ]
            mean_d = sum(deltas) / len(deltas)
            fv.inter_tick_mean = mean_d
            if len(deltas) >= 2:
                var_d = sum((d - mean_d) ** 2 for d in deltas) / len(deltas)
                fv.inter_tick_std = math.sqrt(var_d)

    def _compute_level_proximity(
        self, fv: ScalpingFeatureVector, ticks: list[AlphaTick],
    ) -> None:
        mid = ticks[-1].mid
        if not self._cfg.round_number_enabled or mid == 0:
            return

        # Distance to nearest round number
        # Determine granularity from price magnitude
        if mid > 10:
            # e.g. USDJPY ~150 → round to nearest 0.5
            rounding = 0.5
        else:
            # e.g. EURUSD ~1.08 → round to nearest 0.005 (50 pips)
            rounding = 0.005

        nearest = round(mid / rounding) * rounding
        fv.dist_round_number = mid - nearest

        # Session high/low (entire buffer)
        mids = [t.mid for t in ticks]
        session_high = max(mids)
        session_low = min(mids)
        fv.dist_session_high = mid - session_high  # negative or zero
        fv.dist_session_low = mid - session_low     # positive or zero

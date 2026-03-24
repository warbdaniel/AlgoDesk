"""
Historical 5M Candle ML Pipeline - Feature Engine
==================================================

Computes 64 ML features from historical 5-minute OHLCV candles.
All features are computed from a list of candle dicts (matching the
format returned by data-pipeline's MarketDataStore.get_candles()).

Feature categories (64 total):
  - Trend (13):    SMA, EMA, MACD, ADX
  - Momentum (7):  RSI, Stochastic, CCI, Williams %R, ROC
  - Volatility (12): ATR, Bollinger, Keltner, realised vol, range
  - Volume (1):    tick-count ratio
  - Price action (8): body ratio, shadows, patterns (inside/engulf/pin)
  - Returns (5):   multi-horizon log returns
  - Cross-feature (8): SMA distances, slopes, MA cross, swing levels
  - Temporal (5):  hour/dow sin/cos, session

Requires at least `config.min_warmup` candles for reliable output.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field, fields

from candle_config import CandleFeatureConfig

logger = logging.getLogger("candle_features")

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CandleFeatureVector:
    """Complete feature set for a single 5M candle observation (64 numeric features)."""

    symbol: str = ""
    ts: float = 0.0           # candle open_time

    # ── Trend (13) ────────────────────────────────────────────
    sma_10: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0

    ema_8: float = 0.0
    ema_12: float = 0.0
    ema_21: float = 0.0
    ema_26: float = 0.0

    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0

    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0

    # ── Momentum (7) ─────────────────────────────────────────
    rsi_14: float = 0.0
    stoch_k: float = 0.0
    stoch_d: float = 0.0
    cci_14: float = 0.0            # Commodity Channel Index
    williams_r_14: float = 0.0     # Williams %R
    roc_5: float = 0.0             # Rate of Change 5-bar
    roc_20: float = 0.0            # Rate of Change 20-bar

    # ── Volatility (12) ──────────────────────────────────────
    atr_14: float = 0.0
    atr_ratio: float = 0.0         # ATR / ATR_SMA (expansion/contraction)
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_position: float = 0.0       # 0-1 where price sits in bands
    keltner_upper: float = 0.0     # EMA20 + 1.5*ATR
    keltner_lower: float = 0.0     # EMA20 - 1.5*ATR
    keltner_width: float = 0.0     # (upper - lower) / EMA20
    realized_vol_12: float = 0.0
    realized_vol_24: float = 0.0
    high_low_range: float = 0.0    # current candle range / ATR

    # ── Volume (tick count) (1) ───────────────────────────────
    volume_ratio: float = 0.0      # volume / volume_MA

    # ── Price action (8) ─────────────────────────────────────
    body_ratio: float = 0.0        # |close-open| / (high-low)
    upper_shadow: float = 0.0      # (high - max(O,C)) / range
    lower_shadow: float = 0.0      # (min(O,C) - low) / range
    candle_direction: float = 0.0  # +1 bullish, -1 bearish, 0 doji
    inside_bar: float = 0.0        # 1 if H<prev_H and L>prev_L
    engulfing_bull: float = 0.0    # 1 if bullish engulfing
    engulfing_bear: float = 0.0    # 1 if bearish engulfing
    pin_bar: float = 0.0           # 1 if pin bar (long wick, small body)

    # ── Returns (5) ──────────────────────────────────────────
    ret_1: float = 0.0             # 1-candle (5min) log return
    ret_3: float = 0.0             # 3-candle (15min) log return
    ret_6: float = 0.0             # 6-candle (30min) log return
    ret_12: float = 0.0            # 12-candle (1h) log return
    ret_24: float = 0.0            # 24-candle (2h) log return

    # ── Cross-feature / derived (8) ──────────────────────────
    close_vs_sma_20: float = 0.0   # (close - SMA20) / ATR
    close_vs_sma_50: float = 0.0   # (close - SMA50) / ATR
    sma_20_slope: float = 0.0      # SMA20 change over 5 bars / ATR
    sma_50_slope: float = 0.0      # SMA50 change over 5 bars / ATR
    ma_cross_distance: float = 0.0 # (SMA20 - SMA50) / ATR
    di_spread: float = 0.0         # +DI - -DI
    swing_high_dist_20: float = 0.0  # (close - 20-bar high) / ATR
    swing_low_dist_20: float = 0.0   # (close - 20-bar low) / ATR

    # ── Statistical (5) ──────────────────────────────────────
    price_zscore_20: float = 0.0     # (close - SMA20) / std20
    percentile_rank_100: float = 0.0 # rank of close in 100-bar window (0-1)
    realized_vol_20: float = 0.0     # 20-bar realised volatility
    skewness_20: float = 0.0        # 20-bar return skewness
    kurtosis_20: float = 0.0        # 20-bar return kurtosis

    # ── Temporal (5) ─────────────────────────────────────────
    hour_sin: float = 0.0           # sin(2pi * hour/24)
    hour_cos: float = 0.0           # cos(2pi * hour/24)
    dow_sin: float = 0.0            # sin(2pi * weekday/5)
    dow_cos: float = 0.0            # cos(2pi * weekday/5)
    session: float = 0.0            # 0=asian, 1=london, 2=ny, 3=off

    def to_dict(self) -> dict[str, float]:
        """Return numeric features only (exclude symbol/ts)."""
        result = {}
        for f in fields(self):
            if f.name in ("symbol", "ts"):
                continue
            result[f.name] = getattr(self, f.name)
        return result

    def to_ml_vector(self) -> list[float]:
        """Return flat list of numeric feature values."""
        return list(self.to_dict().values())

    @staticmethod
    def feature_names() -> list[str]:
        """Return ordered feature names matching to_ml_vector()."""
        return [f.name for f in fields(CandleFeatureVector)
                if f.name not in ("symbol", "ts")]


# ---------------------------------------------------------------------------
# Feature engine
# ---------------------------------------------------------------------------
class CandleFeatureEngine:
    """Computes ML features from a list of 5M OHLCV candle dicts.

    Each candle dict must have keys: open, high, low, close, volume, open_time.
    The list must be sorted by open_time ascending.
    """

    def __init__(self, config: CandleFeatureConfig | None = None):
        self._cfg = config or CandleFeatureConfig()

    def compute(
        self,
        candles: list[dict],
        symbol: str = "",
    ) -> list[CandleFeatureVector]:
        """Compute features for all candles after warmup period.

        Returns a list of CandleFeatureVector, one per candle starting
        from index `min_warmup`.
        """
        n = len(candles)
        warmup = self._cfg.min_warmup

        if n < warmup:
            logger.warning(
                "Need >= %d candles for features, got %d", warmup, n
            )
            return []

        # Extract arrays
        opens = [c["open"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]
        volumes = [c.get("volume", 0) for c in candles]
        timestamps = [c.get("open_time", 0.0) for c in candles]

        sym = symbol or candles[0].get("symbol", "")

        # Pre-compute indicator series
        sma_series = {p: _sma(closes, p) for p in self._cfg.sma_periods}
        ema_series = {p: _ema(closes, p) for p in self._cfg.ema_periods}

        # Ensure EMA20 exists for Keltner channels
        ema_20 = ema_series.get(20, _ema(closes, 20))

        ema_fast = ema_series.get(self._cfg.macd_fast, _ema(closes, self._cfg.macd_fast))
        ema_slow = ema_series.get(self._cfg.macd_slow, _ema(closes, self._cfg.macd_slow))
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        macd_sig = _ema(macd_line, self._cfg.macd_signal)

        rsi = _rsi(closes, self._cfg.rsi_period)
        atr = _atr(highs, lows, closes, self._cfg.atr_period)
        atr_sma = _sma(atr, self._cfg.atr_period)
        adx_vals, plus_di, minus_di = _adx(highs, lows, closes, self._cfg.adx_period)
        stoch_k, stoch_d = _stochastic(
            highs, lows, closes, self._cfg.stoch_k_period, self._cfg.stoch_d_period
        )
        bb_upper, bb_lower = _bollinger(closes, self._cfg.bb_period, self._cfg.bb_std)
        cci = _cci(highs, lows, closes, 14)
        williams_r = _williams_r(highs, lows, closes, 14)
        log_returns = _log_returns(closes)
        vol_ma = _sma([float(v) for v in volumes], self._cfg.volume_ma_period)

        # Build feature vectors
        result: list[CandleFeatureVector] = []

        for i in range(warmup, n):
            c_range = highs[i] - lows[i]
            safe_atr = atr[i] if atr[i] > 0 else 1e-10
            sma_20_val = sma_series.get(20, [0.0] * n)[i]
            sma_50_val = sma_series.get(50, [0.0] * n)[i]
            ema_20_val = ema_20[i]

            # Keltner channels
            kelt_mult = 1.5
            kelt_upper = ema_20_val + kelt_mult * atr[i]
            kelt_lower = ema_20_val - kelt_mult * atr[i]

            # Swing high/low over 20 bars
            swing_start = max(0, i - 19)
            swing_high_20 = max(highs[swing_start: i + 1])
            swing_low_20 = min(lows[swing_start: i + 1])

            # Statistical features
            std_20 = _window_std(closes, i, 20)
            zscore_20 = ((closes[i] - sma_20_val) / std_20
                         if std_20 > 0 else 0.0)
            prank = _percentile_rank(closes, i, 100)
            vol_20 = _window_std(log_returns, i, 20)
            skew_20 = _window_skewness(log_returns, i, 20)
            kurt_20 = _window_kurtosis(log_returns, i, 20)

            # Candle patterns
            is_inside = 0.0
            is_engulf_bull = 0.0
            is_engulf_bear = 0.0
            is_pin = 0.0
            if i >= 1:
                is_inside = (1.0 if highs[i] < highs[i - 1]
                             and lows[i] > lows[i - 1] else 0.0)
                is_engulf_bull = (1.0 if closes[i] > opens[i]
                                  and opens[i] <= closes[i - 1]
                                  and closes[i] >= opens[i - 1]
                                  and closes[i - 1] < opens[i - 1] else 0.0)
                is_engulf_bear = (1.0 if closes[i] < opens[i]
                                  and opens[i] >= closes[i - 1]
                                  and closes[i] <= opens[i - 1]
                                  and closes[i - 1] > opens[i - 1] else 0.0)
                body = abs(closes[i] - opens[i])
                if c_range > 0:
                    body_pct = body / c_range
                    upper_wick = highs[i] - max(opens[i], closes[i])
                    lower_wick = min(opens[i], closes[i]) - lows[i]
                    long_wick = max(upper_wick, lower_wick)
                    is_pin = (1.0 if body_pct < 0.33
                              and long_wick / c_range > 0.6 else 0.0)

            # Temporal features
            ts_val = timestamps[i]
            hour, weekday, sess = _time_features(ts_val)

            fv = CandleFeatureVector(
                symbol=sym,
                ts=ts_val,

                # Trend
                sma_10=sma_series.get(10, [0.0] * n)[i],
                sma_20=sma_20_val,
                sma_50=sma_50_val,

                ema_8=ema_series.get(8, [0.0] * n)[i],
                ema_12=ema_series.get(12, [0.0] * n)[i],
                ema_21=ema_series.get(21, [0.0] * n)[i],
                ema_26=ema_series.get(26, [0.0] * n)[i],

                macd_line=macd_line[i],
                macd_signal=macd_sig[i],
                macd_histogram=macd_line[i] - macd_sig[i],

                adx=adx_vals[i],
                plus_di=plus_di[i],
                minus_di=minus_di[i],

                # Momentum
                rsi_14=rsi[i],
                stoch_k=stoch_k[i],
                stoch_d=stoch_d[i],
                cci_14=cci[i],
                williams_r_14=williams_r[i],
                roc_5=((closes[i] / closes[i - 5] - 1) * 100
                       if i >= 5 and closes[i - 5] != 0 else 0.0),
                roc_20=((closes[i] / closes[i - 20] - 1) * 100
                        if i >= 20 and closes[i - 20] != 0 else 0.0),

                # Volatility
                atr_14=atr[i],
                atr_ratio=atr[i] / atr_sma[i] if atr_sma[i] > 0 else 1.0,
                bb_upper=bb_upper[i],
                bb_lower=bb_lower[i],
                bb_width=((bb_upper[i] - bb_lower[i]) / sma_20_val
                          if sma_20_val > 0 else 0.0),
                bb_position=((closes[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i])
                             if (bb_upper[i] - bb_lower[i]) > 0 else 0.5),
                keltner_upper=kelt_upper,
                keltner_lower=kelt_lower,
                keltner_width=((kelt_upper - kelt_lower) / ema_20_val
                               if ema_20_val > 0 else 0.0),
                realized_vol_12=_window_std(log_returns, i, 12),
                realized_vol_24=_window_std(log_returns, i, 24),
                high_low_range=c_range / safe_atr,

                # Volume
                volume_ratio=(volumes[i] / vol_ma[i]
                              if vol_ma[i] > 0 else 1.0),

                # Price action
                body_ratio=(abs(closes[i] - opens[i]) / c_range
                            if c_range > 0 else 0.0),
                upper_shadow=((highs[i] - max(opens[i], closes[i])) / c_range
                              if c_range > 0 else 0.0),
                lower_shadow=((min(opens[i], closes[i]) - lows[i]) / c_range
                              if c_range > 0 else 0.0),
                candle_direction=(1.0 if closes[i] > opens[i]
                                  else (-1.0 if closes[i] < opens[i] else 0.0)),
                inside_bar=is_inside,
                engulfing_bull=is_engulf_bull,
                engulfing_bear=is_engulf_bear,
                pin_bar=is_pin,

                # Returns
                ret_1=_sum_returns(log_returns, i, 1),
                ret_3=_sum_returns(log_returns, i, 3),
                ret_6=_sum_returns(log_returns, i, 6),
                ret_12=_sum_returns(log_returns, i, 12),
                ret_24=_sum_returns(log_returns, i, 24),

                # Cross-feature
                close_vs_sma_20=(closes[i] - sma_20_val) / safe_atr,
                close_vs_sma_50=(closes[i] - sma_50_val) / safe_atr,
                sma_20_slope=((sma_series.get(20, [0.0] * n)[i]
                               - sma_series.get(20, [0.0] * n)[max(0, i - 5)])
                              / safe_atr),
                sma_50_slope=((sma_series.get(50, [0.0] * n)[i]
                               - sma_series.get(50, [0.0] * n)[max(0, i - 5)])
                              / safe_atr),
                ma_cross_distance=(sma_20_val - sma_50_val) / safe_atr,
                di_spread=plus_di[i] - minus_di[i],
                swing_high_dist_20=(closes[i] - swing_high_20) / safe_atr,
                swing_low_dist_20=(closes[i] - swing_low_20) / safe_atr,

                # Statistical
                price_zscore_20=zscore_20,
                percentile_rank_100=prank,
                realized_vol_20=vol_20,
                skewness_20=skew_20,
                kurtosis_20=kurt_20,

                # Temporal
                hour_sin=math.sin(_TWO_PI * hour / 24.0),
                hour_cos=math.cos(_TWO_PI * hour / 24.0),
                dow_sin=math.sin(_TWO_PI * weekday / 5.0),
                dow_cos=math.cos(_TWO_PI * weekday / 5.0),
                session=sess,
            )
            result.append(fv)

        return result

    def compute_latest(
        self, candles: list[dict], symbol: str = "",
    ) -> CandleFeatureVector | None:
        """Compute features for only the most recent candle."""
        vectors = self.compute(candles, symbol)
        return vectors[-1] if vectors else None


# ---------------------------------------------------------------------------
# Indicator helper functions (pure, no external deps)
# ---------------------------------------------------------------------------

def _sma(data: list[float], period: int) -> list[float]:
    n = len(data)
    result = [0.0] * n
    for i in range(period - 1, n):
        result[i] = sum(data[i - period + 1: i + 1]) / period
    return result


def _ema(data: list[float], period: int) -> list[float]:
    n = len(data)
    result = [0.0] * n
    if n < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = sum(data[:period]) / period
    for i in range(period, n):
        result[i] = data[i] * k + result[i - 1] * (1.0 - k)
    return result


def _rsi(closes: list[float], period: int = 14) -> list[float]:
    n = len(closes)
    result = [50.0] * n
    if n < period + 1:
        return result

    gains = []
    losses = []
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return result


def _atr(highs: list[float], lows: list[float], closes: list[float],
         period: int = 14) -> list[float]:
    n = len(highs)
    result = [0.0] * n
    if n < 2:
        return result

    true_ranges = [highs[0] - lows[0]]
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)

    if len(true_ranges) >= period:
        result[period - 1] = sum(true_ranges[:period]) / period
        for i in range(period, len(true_ranges)):
            result[i] = (result[i - 1] * (period - 1) + true_ranges[i]) / period

    return result


def _adx(highs: list[float], lows: list[float], closes: list[float],
         period: int = 14) -> tuple[list[float], list[float], list[float]]:
    n = len(highs)
    adx = [0.0] * n
    plus_di_list = [0.0] * n
    minus_di_list = [0.0] * n

    if n < period + 1:
        return adx, plus_di_list, minus_di_list

    plus_dm = [0.0]
    minus_dm = [0.0]
    tr_list = [highs[0] - lows[0]]

    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        tr_list.append(tr)

    atr_s = sum(tr_list[:period]) / period
    plus_dm_s = sum(plus_dm[:period]) / period
    minus_dm_s = sum(minus_dm[:period]) / period

    dx_values = []
    for i in range(period, n):
        atr_s = (atr_s * (period - 1) + tr_list[i]) / period
        plus_dm_s = (plus_dm_s * (period - 1) + plus_dm[i]) / period
        minus_dm_s = (minus_dm_s * (period - 1) + minus_dm[i]) / period

        plus_di = 100.0 * plus_dm_s / atr_s if atr_s > 0 else 0.0
        minus_di = 100.0 * minus_dm_s / atr_s if atr_s > 0 else 0.0
        plus_di_list[i] = plus_di
        minus_di_list[i] = minus_di

        di_sum = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0
        dx_values.append((i, dx))

    if len(dx_values) >= period:
        adx_val = sum(d for _, d in dx_values[:period]) / period
        adx[dx_values[period - 1][0]] = adx_val
        for j in range(period, len(dx_values)):
            idx, dx = dx_values[j]
            adx_val = (adx_val * (period - 1) + dx) / period
            adx[idx] = adx_val

    return adx, plus_di_list, minus_di_list


def _stochastic(highs: list[float], lows: list[float], closes: list[float],
                k_period: int = 14, d_period: int = 3
                ) -> tuple[list[float], list[float]]:
    n = len(closes)
    k_vals = [50.0] * n
    d_vals = [50.0] * n

    for i in range(k_period - 1, n):
        hh = max(highs[i - k_period + 1: i + 1])
        ll = min(lows[i - k_period + 1: i + 1])
        rng = hh - ll
        k_vals[i] = ((closes[i] - ll) / rng * 100.0) if rng > 0 else 50.0

    for i in range(k_period + d_period - 2, n):
        d_vals[i] = sum(k_vals[i - d_period + 1: i + 1]) / d_period

    return k_vals, d_vals


def _bollinger(closes: list[float], period: int = 20,
               num_std: float = 2.0) -> tuple[list[float], list[float]]:
    n = len(closes)
    upper = [0.0] * n
    lower = [0.0] * n

    for i in range(period - 1, n):
        window = closes[i - period + 1: i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std

    return upper, lower


def _cci(highs: list[float], lows: list[float], closes: list[float],
         period: int = 14) -> list[float]:
    """Commodity Channel Index."""
    n = len(closes)
    result = [0.0] * n
    tp = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]

    for i in range(period - 1, n):
        window = tp[i - period + 1: i + 1]
        mean = sum(window) / period
        mad = sum(abs(x - mean) for x in window) / period
        result[i] = (tp[i] - mean) / (0.015 * mad) if mad > 0 else 0.0

    return result


def _williams_r(highs: list[float], lows: list[float], closes: list[float],
                period: int = 14) -> list[float]:
    """Williams %R oscillator."""
    n = len(closes)
    result = [-50.0] * n

    for i in range(period - 1, n):
        hh = max(highs[i - period + 1: i + 1])
        ll = min(lows[i - period + 1: i + 1])
        rng = hh - ll
        result[i] = ((hh - closes[i]) / rng * -100.0) if rng > 0 else -50.0

    return result


def _log_returns(closes: list[float]) -> list[float]:
    """Compute single-period log returns."""
    result = [0.0]
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            result.append(math.log(closes[i] / closes[i - 1]))
        else:
            result.append(0.0)
    return result


def _sum_returns(log_rets: list[float], idx: int, lookback: int) -> float:
    """Sum of log returns over a lookback window ending at idx."""
    start = max(0, idx - lookback + 1)
    return sum(log_rets[start: idx + 1])


def _window_std(data: list[float], idx: int, window: int) -> float:
    """Standard deviation of data in [idx-window+1, idx]."""
    start = max(0, idx - window + 1)
    vals = data[start: idx + 1]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(var)


def _window_std_raw(data: list[float], idx: int, window: int) -> float:
    """Std dev of raw data values (not centered), for price z-score."""
    start = max(0, idx - window + 1)
    vals = data[start: idx + 1]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(var)


def _percentile_rank(data: list[float], idx: int, window: int) -> float:
    """Percentile rank of data[idx] within [idx-window+1, idx] (0-1)."""
    start = max(0, idx - window + 1)
    vals = data[start: idx + 1]
    if len(vals) < 2:
        return 0.5
    current = data[idx]
    count_below = sum(1 for v in vals if v < current)
    return count_below / (len(vals) - 1)


def _window_skewness(data: list[float], idx: int, window: int) -> float:
    """Sample skewness of data in [idx-window+1, idx]."""
    start = max(0, idx - window + 1)
    vals = data[start: idx + 1]
    n = len(vals)
    if n < 3:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    if var <= 0:
        return 0.0
    std = math.sqrt(var)
    m3 = sum((v - mean) ** 3 for v in vals) / n
    return m3 / (std ** 3)


def _window_kurtosis(data: list[float], idx: int, window: int) -> float:
    """Excess kurtosis of data in [idx-window+1, idx]."""
    start = max(0, idx - window + 1)
    vals = data[start: idx + 1]
    n = len(vals)
    if n < 4:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    if var <= 0:
        return 0.0
    m4 = sum((v - mean) ** 4 for v in vals) / n
    return m4 / (var ** 2) - 3.0


def _time_features(ts: float) -> tuple[float, float, float]:
    """Extract hour (0-23), weekday (0=Mon..4=Fri), session from unix ts.

    Sessions (UTC):
      0 = Asian   (00:00 - 08:00)
      1 = London  (08:00 - 13:00)
      2 = New York (13:00 - 21:00)
      3 = Off     (21:00 - 00:00, weekends)
    """
    import time as _time
    gm = _time.gmtime(ts)
    hour = float(gm.tm_hour)
    weekday = float(gm.tm_wday)  # 0=Monday

    if weekday >= 5:
        session = 3.0
    elif hour < 8:
        session = 0.0
    elif hour < 13:
        session = 1.0
    elif hour < 21:
        session = 2.0
    else:
        session = 3.0

    return hour, weekday, session

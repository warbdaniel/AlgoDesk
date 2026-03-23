"""
Feature engineering pipeline for trading ML models.

Computes technical indicators, statistical features, and
microstructure metrics from candle/tick data. Outputs feature
vectors suitable for regime classification, signal generation,
and risk models.
"""

import math
import logging
from dataclasses import dataclass

logger = logging.getLogger("feature_engine")


@dataclass
class FeatureVector:
    """Complete feature set for a single observation."""
    symbol: str
    timestamp: float

    # Trend indicators
    sma_20: float = 0.0
    sma_50: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0

    # Momentum
    rsi_14: float = 0.0
    stoch_k: float = 0.0
    stoch_d: float = 0.0
    momentum_10: float = 0.0
    roc_10: float = 0.0

    # Volatility
    atr_14: float = 0.0
    atr_ratio: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_position: float = 0.0  # where price sits within bands (0-1)

    # Volume / microstructure
    tick_count: int = 0
    spread_avg: float = 0.0

    # Price action
    body_ratio: float = 0.0  # |close-open| / (high-low)
    upper_shadow: float = 0.0
    lower_shadow: float = 0.0
    return_1: float = 0.0
    return_5: float = 0.0
    return_10: float = 0.0
    volatility_10: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def to_ml_vector(self) -> list[float]:
        """Return numeric features only, for ML input."""
        d = self.to_dict()
        d.pop("symbol")
        d.pop("timestamp")
        return [float(v) for v in d.values()]

    @staticmethod
    def feature_names() -> list[str]:
        """Return ordered feature names matching to_ml_vector()."""
        dummy = FeatureVector(symbol="", timestamp=0)
        d = dummy.to_dict()
        d.pop("symbol")
        d.pop("timestamp")
        return list(d.keys())


class FeatureEngine:
    """Computes trading features from OHLCV candle data."""

    def compute(self, candles: list[dict], symbol: str = "") -> list[FeatureVector]:
        """Compute features for a list of candles (dicts with open/high/low/close/volume)."""
        if len(candles) < 50:
            logger.warning("Need at least 50 candles for reliable features, got %d", len(candles))
            return []

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        opens = [c["open"] for c in candles]
        volumes = [c.get("volume", 0) for c in candles]
        timestamps = [c.get("open_time", 0) for c in candles]

        sym = symbol or candles[0].get("symbol", "")

        # Pre-compute indicator series
        sma_20 = self._sma(closes, 20)
        sma_50 = self._sma(closes, 50)
        ema_12 = self._ema(closes, 12)
        ema_26 = self._ema(closes, 26)
        macd_line = [e12 - e26 for e12, e26 in zip(ema_12, ema_26)]
        macd_signal = self._ema(macd_line, 9)
        rsi = self._rsi(closes, 14)
        atr = self._atr(highs, lows, closes, 14)
        atr_sma = self._sma(atr, 14)
        adx_vals, plus_di, minus_di = self._adx(highs, lows, closes, 14)
        stoch_k, stoch_d = self._stochastic(highs, lows, closes, 14, 3)
        bb_upper, bb_lower = self._bollinger(closes, 20, 2.0)
        returns = self._returns(closes)

        features = []
        # Start from index 50 to ensure all lookbacks are valid
        start = 50
        for i in range(start, len(candles)):
            rng = highs[i] - lows[i]

            fv = FeatureVector(
                symbol=sym,
                timestamp=timestamps[i],

                # Trend
                sma_20=sma_20[i],
                sma_50=sma_50[i],
                ema_12=ema_12[i],
                ema_26=ema_26[i],
                macd=macd_line[i],
                macd_signal=macd_signal[i],
                macd_histogram=macd_line[i] - macd_signal[i],
                adx=adx_vals[i],
                plus_di=plus_di[i],
                minus_di=minus_di[i],

                # Momentum
                rsi_14=rsi[i],
                stoch_k=stoch_k[i],
                stoch_d=stoch_d[i],
                momentum_10=closes[i] - closes[i - 10] if i >= 10 else 0,
                roc_10=(closes[i] / closes[i - 10] - 1) * 100 if i >= 10 and closes[i - 10] != 0 else 0,

                # Volatility
                atr_14=atr[i],
                atr_ratio=atr[i] / atr_sma[i] if atr_sma[i] > 0 else 1.0,
                bb_upper=bb_upper[i],
                bb_lower=bb_lower[i],
                bb_width=(bb_upper[i] - bb_lower[i]) / sma_20[i] if sma_20[i] > 0 else 0,
                bb_position=(closes[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 0 else 0.5,

                # Volume / microstructure
                tick_count=volumes[i],

                # Price action
                body_ratio=abs(closes[i] - opens[i]) / rng if rng > 0 else 0,
                upper_shadow=(highs[i] - max(opens[i], closes[i])) / rng if rng > 0 else 0,
                lower_shadow=(min(opens[i], closes[i]) - lows[i]) / rng if rng > 0 else 0,
                return_1=returns[i],
                return_5=sum(returns[max(0, i - 4):i + 1]),
                return_10=sum(returns[max(0, i - 9):i + 1]),
                volatility_10=self._std(returns[max(0, i - 9):i + 1]),
            )
            features.append(fv)

        return features

    def compute_latest(self, candles: list[dict], symbol: str = "") -> FeatureVector | None:
        """Compute features for only the latest candle."""
        features = self.compute(candles, symbol)
        return features[-1] if features else None

    # ── Technical indicator implementations ──────────────────────

    @staticmethod
    def _sma(data: list[float], period: int) -> list[float]:
        result = [0.0] * len(data)
        for i in range(period - 1, len(data)):
            result[i] = sum(data[i - period + 1:i + 1]) / period
        return result

    @staticmethod
    def _ema(data: list[float], period: int) -> list[float]:
        result = [0.0] * len(data)
        if len(data) < period:
            return result
        k = 2.0 / (period + 1)
        result[period - 1] = sum(data[:period]) / period
        for i in range(period, len(data)):
            result[i] = data[i] * k + result[i - 1] * (1 - k)
        return result

    @staticmethod
    def _rsi(closes: list[float], period: int = 14) -> list[float]:
        result = [50.0] * len(closes)
        if len(closes) < period + 1:
            return result

        gains = []
        losses = []
        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))

        # Wilder smoothing
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

    @staticmethod
    def _atr(highs: list[float], lows: list[float], closes: list[float],
             period: int = 14) -> list[float]:
        result = [0.0] * len(highs)
        if len(highs) < 2:
            return result

        true_ranges = [highs[0] - lows[0]]
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)

        # Wilder smoothing
        if len(true_ranges) >= period:
            result[period - 1] = sum(true_ranges[:period]) / period
            for i in range(period, len(true_ranges)):
                result[i] = (result[i - 1] * (period - 1) + true_ranges[i]) / period

        return result

    @staticmethod
    def _adx(highs: list[float], lows: list[float], closes: list[float],
             period: int = 14) -> tuple[list[float], list[float], list[float]]:
        n = len(highs)
        adx = [0.0] * n
        plus_di_list = [0.0] * n
        minus_di_list = [0.0] * n

        if n < period + 1:
            return adx, plus_di_list, minus_di_list

        # Directional movement
        plus_dm = [0.0]
        minus_dm = [0.0]
        tr_list = [highs[0] - lows[0]]

        for i in range(1, n):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            tr_list.append(tr)

        # Wilder smoothing
        atr_s = sum(tr_list[:period]) / period
        plus_dm_s = sum(plus_dm[:period]) / period
        minus_dm_s = sum(minus_dm[:period]) / period

        dx_values = []
        for i in range(period, n):
            atr_s = (atr_s * (period - 1) + tr_list[i]) / period
            plus_dm_s = (plus_dm_s * (period - 1) + plus_dm[i]) / period
            minus_dm_s = (minus_dm_s * (period - 1) + minus_dm[i]) / period

            plus_di = 100 * plus_dm_s / atr_s if atr_s > 0 else 0
            minus_di = 100 * minus_dm_s / atr_s if atr_s > 0 else 0
            plus_di_list[i] = plus_di
            minus_di_list[i] = minus_di

            di_sum = plus_di + minus_di
            dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
            dx_values.append((i, dx))

        # ADX = smoothed DX
        if len(dx_values) >= period:
            adx_val = sum(d for _, d in dx_values[:period]) / period
            adx[dx_values[period - 1][0]] = adx_val
            for j in range(period, len(dx_values)):
                idx, dx = dx_values[j]
                adx_val = (adx_val * (period - 1) + dx) / period
                adx[idx] = adx_val

        return adx, plus_di_list, minus_di_list

    @staticmethod
    def _stochastic(highs: list[float], lows: list[float], closes: list[float],
                    k_period: int = 14, d_period: int = 3) -> tuple[list[float], list[float]]:
        n = len(closes)
        k_vals = [50.0] * n
        d_vals = [50.0] * n

        for i in range(k_period - 1, n):
            hh = max(highs[i - k_period + 1:i + 1])
            ll = min(lows[i - k_period + 1:i + 1])
            rng = hh - ll
            k_vals[i] = ((closes[i] - ll) / rng * 100) if rng > 0 else 50.0

        # %D = SMA of %K
        for i in range(k_period + d_period - 2, n):
            d_vals[i] = sum(k_vals[i - d_period + 1:i + 1]) / d_period

        return k_vals, d_vals

    @staticmethod
    def _bollinger(closes: list[float], period: int = 20,
                   num_std: float = 2.0) -> tuple[list[float], list[float]]:
        n = len(closes)
        upper = [0.0] * n
        lower = [0.0] * n

        for i in range(period - 1, n):
            window = closes[i - period + 1:i + 1]
            mean = sum(window) / period
            variance = sum((x - mean) ** 2 for x in window) / period
            std = math.sqrt(variance)
            upper[i] = mean + num_std * std
            lower[i] = mean - num_std * std

        return upper, lower

    @staticmethod
    def _returns(closes: list[float]) -> list[float]:
        result = [0.0]
        for i in range(1, len(closes)):
            result.append((closes[i] / closes[i - 1] - 1) if closes[i - 1] != 0 else 0)
        return result

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

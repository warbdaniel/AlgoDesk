"""
Trend Regime Consensus System v2.1.0
4-layer consensus: HMM, Directional Change, ADX+ATR, PELT Change Point
v2.1.0 changes: diagonal covariance for HMM, min variance regularization,
improved flat candle filtering, adaptive DC thresholds, better 4H support.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

class TrendDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class TrendStrength(Enum):
    STRONG_TREND = "STRONG_TREND"
    MILD_TREND = "MILD_TREND"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"

@dataclass
class LayerResult:
    layer_name: str
    direction: TrendDirection
    strength: float
    is_trending: bool
    metadata: Dict = field(default_factory=dict)

@dataclass
class TrendRegimeResult:
    regime: TrendStrength
    direction: TrendDirection
    confidence: float
    consensus_count: int
    total_layers: int
    tradeable: bool
    layer_results: List[LayerResult]
    timestamp: Optional[str] = None

# --- Layer 1: HMM Regime Detection (FIXED for 4H) ---
class HMMRegimeLayer:
    def __init__(self, n_states=3, n_iter=100):
        self.n_states = n_states
        self.n_iter = n_iter

    def detect(self, df: pd.DataFrame) -> LayerResult:
        try:
            from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
        except ImportError:
            return LayerResult("HMM", TrendDirection.NEUTRAL, 0.0, False,
                             {"error": "hmmlearn not installed"})

        try:
            closes = df['close'].values.astype(float)
            returns = np.diff(np.log(closes))

            # Enhanced flat candle filtering for 4H data
            highs = df['high'].values[1:].astype(float)
            lows = df['low'].values[1:].astype(float)
            opens = df['open'].values[1:].astype(float)
            closes_shifted = closes[1:]

            # Filter: remove candles where range is < 0.01% of price (effectively flat)
            candle_range = (highs - lows) / closes_shifted
            min_range = 0.0001  # 0.01% - catches weekend/holiday flat candles
            active_mask = candle_range > min_range

            returns = returns[active_mask]

            if len(returns) < 50:
                return LayerResult("HMM", TrendDirection.NEUTRAL, 0.0, False,
                                 {"error": "insufficient active data",
                                  "active_candles": int(active_mask.sum()),
                                  "total_candles": len(active_mask)})

            vol = pd.Series(returns).rolling(20).std().values
            mask = ~np.isnan(vol)
            features = np.column_stack([returns[mask], vol[mask]])

            if len(features) < 50:
                return LayerResult("HMM", TrendDirection.NEUTRAL, 0.0, False,
                                 {"error": "insufficient data after vol calc"})

            # Add minimum variance regularization
            feature_var = np.var(features, axis=0)
            min_var = 1e-8
            for col in range(features.shape[1]):
                if feature_var[col] < min_var:
                    features[:, col] += np.random.normal(0, np.sqrt(min_var), len(features))

            # Use DIAGONAL covariance (key fix for 4H robustness)
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=self.n_iter,
                random_state=42,
                min_covar=1e-3
            )
            # Check for near-zero variance features (weekend/flat candle protection)
            feature_vars = features.var(axis=0)
            if (feature_vars < 1e-6).any():
                logger.warning("HMM: Near-zero variance detected in features, skipping HMM layer")
                return LayerResult("HMM", TrendDirection.NEUTRAL, 0.0, False,
                                  {"error": "low_variance_skip", "hmm_unavailable": True})
            # Standardize features before fitting
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            model.fit(features)
            states = model.predict(features)

            # Classify states by mean return
            state_means = {}
            for s in range(self.n_states):
                state_mask = states == s
                if state_mask.sum() > 0:
                    state_means[s] = features[state_mask, 0].mean()

            if not state_means:
                return LayerResult("HMM", TrendDirection.NEUTRAL, 0.0, False,
                                 {"error": "no valid states"})

            current_state = states[-1]
            current_mean = state_means.get(current_state, 0)

            sorted_states = sorted(state_means.items(), key=lambda x: x[1])
            bull_state = sorted_states[-1][0]
            bear_state = sorted_states[0][0]

            # Adaptive threshold based on recent volatility
            recent_vol = np.std(returns[-50:]) if len(returns) >= 50 else np.std(returns)
            trend_threshold = max(recent_vol * 0.3, 0.0001)

            if current_state == bull_state and current_mean > trend_threshold:
                direction = TrendDirection.BULLISH
                strength = min(abs(current_mean) / (recent_vol + 1e-10), 1.0)
                is_trending = True
            elif current_state == bear_state and current_mean < -trend_threshold:
                direction = TrendDirection.BEARISH
                strength = min(abs(current_mean) / (recent_vol + 1e-10), 1.0)
                is_trending = True
            else:
                direction = TrendDirection.NEUTRAL
                strength = 0.2
                is_trending = False

            return LayerResult("HMM", direction, float(strength), is_trending,
                             {"current_state": int(current_state),
                              "state_means": {str(k): float(v) for k,v in state_means.items()},
                              "trend_threshold": float(trend_threshold),
                              "active_candles": int(active_mask.sum()),
                              "covariance_type": "diag",
                              "converged": bool(model.monitor_.converged)})

        except Exception as e:
            return LayerResult("HMM", TrendDirection.NEUTRAL, 0.0, False,
                             {"error": str(e)})

# --- Layer 2: Directional Change (adaptive thresholds) ---
class DirectionalChangeLayer:
    def __init__(self, dc_threshold=None):
        # dc_threshold=None means auto-detect based on data
        self.dc_threshold = dc_threshold

    def detect(self, df: pd.DataFrame) -> LayerResult:
        try:
            closes = df['close'].values.astype(float)

            # Auto-detect threshold if not specified
            if self.dc_threshold is None:
                # Use 0.5x median absolute bar-to-bar return
                returns = np.abs(np.diff(closes) / closes[:-1])
                threshold = float(np.median(returns[returns > 0]) * 0.5)
                threshold = max(threshold, 0.0005)  # Floor at 0.05%
                threshold = min(threshold, 0.02)    # Cap at 2%
            else:
                threshold = self.dc_threshold

            # Detect directional changes
            events = []
            mode = "up"
            extreme_price = closes[0]
            extreme_idx = 0

            for i in range(1, len(closes)):
                if mode == "up":
                    if closes[i] > extreme_price:
                        extreme_price = closes[i]
                        extreme_idx = i
                    elif (extreme_price - closes[i]) / extreme_price >= threshold:
                        events.append(("down", extreme_idx, i, extreme_price, closes[i]))
                        mode = "down"
                        extreme_price = closes[i]
                        extreme_idx = i
                else:
                    if closes[i] < extreme_price:
                        extreme_price = closes[i]
                        extreme_idx = i
                    elif (closes[i] - extreme_price) / extreme_price >= threshold:
                        events.append(("up", extreme_idx, i, extreme_price, closes[i]))
                        mode = "up"
                        extreme_price = closes[i]
                        extreme_idx = i

            if len(events) < 2:
                return LayerResult("DC", TrendDirection.NEUTRAL, 0.1, False,
                                 {"events": len(events), "threshold": float(threshold),
                                  "auto_threshold": self.dc_threshold is None})

            # Analyze recent events (last 5)
            recent = events[-min(5, len(events)):]
            up_events = sum(1 for e in recent if e[0] == "up")
            down_events = sum(1 for e in recent if e[0] == "down")

            # Direction based on recent event balance and last event
            last_event = events[-1]
            event_ratio = up_events / max(down_events, 1)

            if event_ratio > 1.5 and last_event[0] == "up":
                direction = TrendDirection.BULLISH
                strength = min(event_ratio / 3.0, 1.0)
                is_trending = True
            elif event_ratio < 0.67 and last_event[0] == "down":
                direction = TrendDirection.BEARISH
                strength = min(1.0 / max(event_ratio, 0.1) / 3.0, 1.0)
                is_trending = True
            else:
                direction = TrendDirection.NEUTRAL
                strength = 0.3
                is_trending = False

            return LayerResult("DC", direction, float(strength), is_trending,
                             {"events": len(events),
                              "recent_up": up_events,
                              "recent_down": down_events,
                              "threshold": float(threshold),
                              "auto_threshold": self.dc_threshold is None,
                              "last_event": last_event[0]})

        except Exception as e:
            return LayerResult("DC", TrendDirection.NEUTRAL, 0.0, False,
                             {"error": str(e)})

# --- Layer 3: ADX + ATR ---
class ADXATRLayer:
    def __init__(self, adx_period=14, atr_period=14, adx_threshold=25):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.adx_threshold = adx_threshold

    def detect(self, df: pd.DataFrame) -> LayerResult:
        try:
            from ta.trend import ADXIndicator
            from ta.volatility import AverageTrueRange

            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)

            adx_ind = ADXIndicator(high, low, close, window=self.adx_period)
            adx = adx_ind.adx().iloc[-1]
            plus_di = adx_ind.adx_pos().iloc[-1]
            minus_di = adx_ind.adx_neg().iloc[-1]

            atr_ind = AverageTrueRange(high, low, close, window=self.atr_period)
            atr = atr_ind.average_true_range()
            current_atr = atr.iloc[-1]

            # Check if ATR is expanding (recent > average)
            atr_mean = atr.iloc[-20:].mean() if len(atr) >= 20 else atr.mean()
            atr_expanding = bool(current_atr > atr_mean * 1.1)

            is_trending = adx > self.adx_threshold and atr_expanding

            if plus_di > minus_di and adx > self.adx_threshold:
                direction = TrendDirection.BULLISH
                strength = min(adx / 50.0, 1.0)
            elif minus_di > plus_di and adx > self.adx_threshold:
                direction = TrendDirection.BEARISH
                strength = min(adx / 50.0, 1.0)
            else:
                direction = TrendDirection.NEUTRAL
                strength = adx / 50.0
                is_trending = False

            return LayerResult("ADX_ATR", direction, float(strength), is_trending,
                             {"adx": float(adx),
                              "plus_di": float(plus_di),
                              "minus_di": float(minus_di),
                              "atr": float(current_atr),
                              "atr_expanding": atr_expanding})

        except Exception as e:
            return LayerResult("ADX_ATR", TrendDirection.NEUTRAL, 0.0, False,
                             {"error": str(e)})

# --- Layer 4: PELT Change Point Detection ---
class PELTChangePointLayer:
    def __init__(self, min_size=20, penalty=3):
        self.min_size = min_size
        self.penalty = penalty

    def detect(self, df: pd.DataFrame) -> LayerResult:
        try:
            import ruptures as rpt

            closes = df['close'].values.astype(float)
            returns = np.diff(np.log(closes))

            if len(returns) < self.min_size * 2:
                return LayerResult("PELT", TrendDirection.NEUTRAL, 0.0, False,
                                 {"error": "insufficient data"})

            algo = rpt.Pelt(model="rbf", min_size=self.min_size).fit(returns.reshape(-1, 1))
            change_points = algo.predict(pen=self.penalty)

            # Get the last regime segment
            if len(change_points) >= 2:
                last_cp = change_points[-2]
            else:
                last_cp = 0

            segment_returns = returns[last_cp:]
            segment_mean = segment_returns.mean()
            segment_vol = segment_returns.std()

            # Direction from segment trend
            vol_threshold = max(segment_vol * 0.3, 0.0001)

            if segment_mean > vol_threshold:
                direction = TrendDirection.BULLISH
                strength = min(abs(segment_mean) / (segment_vol + 1e-10), 1.0)
                is_trending = True
            elif segment_mean < -vol_threshold:
                direction = TrendDirection.BEARISH
                strength = min(abs(segment_mean) / (segment_vol + 1e-10), 1.0)
                is_trending = True
            else:
                direction = TrendDirection.NEUTRAL
                strength = 0.2
                is_trending = False

            # Recency: is the last change point recent?
            bars_since_cp = len(returns) - last_cp
            recency = max(0, 1.0 - bars_since_cp / 100)

            return LayerResult("PELT", direction, float(strength), is_trending,
                             {"change_points": len(change_points) - 1,
                              "segment_length": len(segment_returns),
                              "segment_mean_return": float(segment_mean),
                              "bars_since_cp": int(bars_since_cp),
                              "recency": float(recency)})

        except Exception as e:
            return LayerResult("PELT", TrendDirection.NEUTRAL, 0.0, False,
                             {"error": str(e)})

# --- Consensus Engine ---
class TrendRegimeConsensus:
    def __init__(self, consensus_threshold=3):
        self.consensus_threshold = consensus_threshold
        self.layers = [
            HMMRegimeLayer(n_states=3, n_iter=100),
            DirectionalChangeLayer(dc_threshold=None),  # Auto-detect
            ADXATRLayer(adx_period=14, atr_period=14, adx_threshold=25),
            PELTChangePointLayer(min_size=20, penalty=3)
        ]

    def detect(self, df: pd.DataFrame) -> TrendRegimeResult:
        # Filter flat candles from input (for layers that don't self-filter)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            candle_range = (df['high'].astype(float) - df['low'].astype(float)) / df['close'].astype(float)
            active = candle_range > 0.0001
            df_filtered = df[active].copy()
            if len(df_filtered) < 50:
                df_filtered = df.copy()  # Fallback to unfiltered
        else:
            df_filtered = df.copy()

        layer_results = []
        for layer in self.layers:
            result = layer.detect(df_filtered)
            layer_results.append(result)

        # Count trending layers and direction votes
        trending_count = sum(1 for r in layer_results if r.is_trending)
        direction_votes = {}
        for r in layer_results:
            if r.is_trending and r.direction != TrendDirection.NEUTRAL:
                d = r.direction
                direction_votes[d] = direction_votes.get(d, 0) + 1

        # Determine consensus direction
        if direction_votes:
            direction = max(direction_votes, key=direction_votes.get)
            consensus_count = direction_votes[direction]
        else:
            direction = TrendDirection.NEUTRAL
            consensus_count = 0

        # Determine regime strength
        if consensus_count >= 3:
            regime = TrendStrength.STRONG_TREND
        elif consensus_count >= 2:
            regime = TrendStrength.MILD_TREND
        elif trending_count >= 1:
            regime = TrendStrength.RANGING
        else:
            regime = TrendStrength.CHOPPY

        # Compute confidence as average of agreeing layers
        agreeing_layers = [r for r in layer_results
                          if r.direction == direction and r.is_trending]
        if agreeing_layers:
            confidence = np.mean([r.strength for r in agreeing_layers])
        else:
            confidence = 0.1

        tradeable = consensus_count >= self.consensus_threshold and direction != TrendDirection.NEUTRAL

        return TrendRegimeResult(
            regime=regime,
            direction=direction,
            confidence=float(confidence),
            consensus_count=consensus_count,
            total_layers=len(self.layers),
            tradeable=tradeable,
            layer_results=layer_results,
            timestamp=str(df['timestamp'].iloc[-1]) if 'timestamp' in df.columns else None
        )

def detect_regime_from_candles(df: pd.DataFrame, consensus_threshold=3) -> TrendRegimeResult:
    """Convenience function: run consensus detection on a DataFrame of candles."""
    required_cols = {'open', 'high', 'low', 'close'}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"DataFrame must have columns: {required_cols}")

    detector = TrendRegimeConsensus(consensus_threshold=consensus_threshold)
    return detector.detect(df)

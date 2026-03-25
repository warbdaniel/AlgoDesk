"""
PILLAR 6: Multi-Timeframe Construction
=======================================
Synthetic 15M/1H/4H candles from 5M data.
HTF trend + efficiency confirmation scoring.
"""

import numpy as np
import pandas as pd


def construct_higher_timeframes(df_5m):
    """Build 15M, 1H, and 4H candles from 5M data."""
    htf = {}
    for tf, rule in [('15m', '15min'), ('1h', '1h'), ('4h', '4h')]:
        resampled = df_5m.resample(rule, on='timestamp').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        }).dropna()
        if len(resampled) > 0:
            htf[tf] = resampled
    return htf


def compute_htf_context(htf, current_time):
    """Get higher-timeframe trend direction and strength at current timestamp."""
    context = {}
    for tf_name, df_tf in htf.items():
        mask = df_tf.index <= current_time
        if mask.sum() < 50:
            continue
        recent = df_tf[mask]
        close = recent['close']

        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        trend_dir = 'UP' if ema_fast.iloc[-1] > ema_slow.iloc[-1] else 'DOWN'

        er_window = min(10, len(close) - 1)
        if er_window > 0:
            direction = abs(close.iloc[-1] - close.iloc[-er_window])
            volatility = abs(close.diff()).iloc[-er_window:].sum()
            efficiency = direction / volatility if volatility > 0 else 0
        else:
            efficiency = 0

        last_3 = recent.iloc[-3:]
        bullish_bars = (last_3['close'] > last_3['open']).sum()
        bearish_bars = (last_3['close'] < last_3['open']).sum()

        context[tf_name] = {
            'trend_dir': trend_dir, 'efficiency': efficiency,
            'ema_fast': ema_fast.iloc[-1], 'ema_slow': ema_slow.iloc[-1],
            'bullish_bars': bullish_bars, 'bearish_bars': bearish_bars,
        }
    return context


def htf_confirmation_score(htf_context, trade_direction):
    """How many higher timeframes agree? Returns 0.0 to 1.0."""
    if not htf_context:
        return 0.5
    confirmations = 0
    total_weight = 0
    weights = {'15m': 1.0, '1h': 2.0, '4h': 3.0}

    for tf_name, ctx in htf_context.items():
        w = weights.get(tf_name, 1.0)
        total_weight += w
        if trade_direction == 'long' and ctx['trend_dir'] == 'UP':
            confirmations += w
        elif trade_direction == 'short' and ctx['trend_dir'] == 'DOWN':
            confirmations += w
        if ctx['efficiency'] > 0.4:
            confirmations += w * 0.3

    return min(confirmations / total_weight, 1.0) if total_weight > 0 else 0.5


def mtf_gate(htf_score, entry_type):
    """Minimum HTF confirmation thresholds by entry type."""
    thresholds = {
        'breakout': 0.5,
        'pullback': 0.6,
        'sweep_reversal': 0.3,
    }
    min_score = thresholds.get(entry_type, 0.5)
    return htf_score >= min_score

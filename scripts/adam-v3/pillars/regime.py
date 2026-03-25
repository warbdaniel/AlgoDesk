"""
PILLAR 1: Dual-Timeframe Regime Detection
==========================================
Long-Term Regime (LTR): Structural trend at session/daily level.
Short-Term Regime (STR): Current 1-2 hour momentum.
Interaction Matrix: Maps (LTR, STR) pairs to trade actions.
"""

import numpy as np
import pandas as pd


# ═══ INTERACTION MATRIX ═══
REGIME_INTERACTION = {
    # ALIGNED TREND — highest conviction, full size
    ('LT_TREND_UP',   'ST_IMPULSE_UP'):    'TRADE_LONG_FULL',
    ('LT_TREND_DOWN', 'ST_IMPULSE_DOWN'):  'TRADE_SHORT_FULL',

    # TREND + DRIFT — moderate size
    ('LT_TREND_UP',   'ST_DRIFT_UP'):      'TRADE_LONG_MODERATE',
    ('LT_TREND_DOWN', 'ST_DRIFT_DOWN'):    'TRADE_SHORT_MODERATE',

    # PULLBACK IN TREND — best R:R, wait for re-entry signal
    ('LT_TREND_UP',   'ST_DRIFT_DOWN'):    'WATCH_LONG_PULLBACK',
    ('LT_TREND_UP',   'ST_CHOP'):          'WATCH_LONG_PULLBACK',
    ('LT_TREND_UP',   'ST_PAUSE'):         'WATCH_LONG_PULLBACK',
    ('LT_TREND_DOWN', 'ST_DRIFT_UP'):      'WATCH_SHORT_PULLBACK',
    ('LT_TREND_DOWN', 'ST_CHOP'):          'WATCH_SHORT_PULLBACK',
    ('LT_TREND_DOWN', 'ST_PAUSE'):         'WATCH_SHORT_PULLBACK',

    # COUNTER-TREND IMPULSE — deep pullback, wait for reversal
    ('LT_TREND_UP',   'ST_IMPULSE_DOWN'):  'WAIT_REVERSAL_LONG',
    ('LT_TREND_DOWN', 'ST_IMPULSE_UP'):    'WAIT_REVERSAL_SHORT',

    # TRANSITION — reduced size, tighter stops
    ('LT_TRANSITION', 'ST_IMPULSE_UP'):    'TRADE_LONG_HALF',
    ('LT_TRANSITION', 'ST_IMPULSE_DOWN'):  'TRADE_SHORT_HALF',
    ('LT_TRANSITION', 'ST_DRIFT_UP'):      'WATCH_LONG_PULLBACK',
    ('LT_TRANSITION', 'ST_DRIFT_DOWN'):    'WATCH_SHORT_PULLBACK',
    ('LT_TRANSITION', 'ST_CHOP'):          'NO_TRADE',
    ('LT_TRANSITION', 'ST_PAUSE'):         'NO_TRADE',

    # NO TRADE — structural chop or mean-reversion
    ('LT_RANGE',       'ST_CHOP'):         'NO_TRADE',
    ('LT_RANGE',       'ST_PAUSE'):        'NO_TRADE',
    ('LT_RANGE',       'ST_DRIFT_UP'):     'NO_TRADE',
    ('LT_RANGE',       'ST_DRIFT_DOWN'):   'NO_TRADE',
    ('LT_MEAN_REVERT', 'ST_IMPULSE_UP'):   'NO_TRADE',
    ('LT_MEAN_REVERT', 'ST_IMPULSE_DOWN'): 'NO_TRADE',
    ('LT_MEAN_REVERT', 'ST_DRIFT_UP'):     'NO_TRADE',
    ('LT_MEAN_REVERT', 'ST_DRIFT_DOWN'):   'NO_TRADE',
    ('LT_MEAN_REVERT', 'ST_CHOP'):         'NO_TRADE',
    ('LT_MEAN_REVERT', 'ST_PAUSE'):        'NO_TRADE',

    # RANGE + IMPULSE — possible breakout, half size
    ('LT_RANGE',       'ST_IMPULSE_UP'):   'TRADE_LONG_HALF',
    ('LT_RANGE',       'ST_IMPULSE_DOWN'): 'TRADE_SHORT_HALF',
}


def compute_lt_regime(df, lt_efficiency_thresh=0.35, lt_hurst_thresh=0.55):
    """
    Long-Term Regime: structural trend at session/daily level.
    Uses EMA50/100/200, Efficiency Ratio(60), Hurst proxy(96).
    States: LT_TREND_UP, LT_TREND_DOWN, LT_TRANSITION, LT_RANGE, LT_MEAN_REVERT
    """
    close, high, low = df['close'], df['high'], df['low']

    # MA stacking (structural direction)
    ema_50  = close.ewm(span=50, adjust=False).mean()    # ~4 hours
    ema_100 = close.ewm(span=100, adjust=False).mean()   # ~8 hours
    ema_200 = close.ewm(span=200, adjust=False).mean()   # ~16 hours

    lt_bullish_stack = (ema_50 > ema_100) & (ema_100 > ema_200)
    lt_bearish_stack = (ema_50 < ema_100) & (ema_100 < ema_200)

    # Kaufman Efficiency Ratio (trend quality)
    lt_window = 60  # 5 hours
    lt_direction = abs(close - close.shift(lt_window))
    lt_volatility = abs(close.diff()).rolling(lt_window).sum()
    lt_efficiency = lt_direction / lt_volatility.replace(0, np.nan)

    # Hurst Proxy (trend persistence)
    hurst_window = 96  # 8 hours
    half = hurst_window // 2
    full_range = high.rolling(hurst_window).max() - low.rolling(hurst_window).min()
    half_range = high.rolling(half).max() - low.rolling(half).min()
    lt_hurst = np.log(full_range / half_range.replace(0, np.nan)) / np.log(2)

    df['lt_efficiency'] = lt_efficiency
    df['lt_hurst'] = lt_hurst
    df['ema_50'] = ema_50
    df['ema_100'] = ema_100
    df['ema_200'] = ema_200

    conditions = [
        (lt_efficiency > lt_efficiency_thresh) & (lt_hurst > lt_hurst_thresh) & lt_bullish_stack,
        (lt_efficiency > lt_efficiency_thresh) & (lt_hurst > lt_hurst_thresh) & lt_bearish_stack,
        (lt_efficiency > lt_efficiency_thresh * 0.7) & (~lt_bullish_stack & ~lt_bearish_stack),
        (lt_hurst < 0.45),
    ]
    choices = ['LT_TREND_UP', 'LT_TREND_DOWN', 'LT_TRANSITION', 'LT_MEAN_REVERT']
    df['lt_regime'] = np.select(conditions, choices, default='LT_RANGE')
    return df


def compute_st_regime(df, st_adx_thresh=30, st_efficiency_thresh=0.4):
    """
    Short-Term Regime: current 1-2 hour momentum.
    Uses ADX(14), EMA12/26, Efficiency Ratio(12), momentum slope.
    States: ST_IMPULSE_UP/DOWN, ST_DRIFT_UP/DOWN, ST_CHOP, ST_PAUSE
    """
    close = df['close']

    st_window = 12  # 1 hour
    st_direction = abs(close - close.shift(st_window))
    st_volatility = abs(close.diff()).rolling(st_window).sum()
    st_efficiency = st_direction / st_volatility.replace(0, np.nan)

    momentum_5 = close - close.shift(5)
    momentum_slope = momentum_5 - momentum_5.shift(3)

    df['st_efficiency'] = st_efficiency
    df['momentum_slope'] = momentum_slope

    # EMA 12/26 for short-term direction
    if 'ema_12' not in df.columns:
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
    if 'ema_26' not in df.columns:
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()

    adx = df['adx'] if 'adx' in df.columns else pd.Series(0, index=df.index)
    plus_di = df['plus_di'] if 'plus_di' in df.columns else pd.Series(0, index=df.index)
    minus_di = df['minus_di'] if 'minus_di' in df.columns else pd.Series(0, index=df.index)

    conditions = [
        (adx >= st_adx_thresh) & (st_efficiency > st_efficiency_thresh) & (plus_di > minus_di),
        (adx >= st_adx_thresh) & (st_efficiency > st_efficiency_thresh) & (minus_di > plus_di),
        (st_efficiency > 0.25) & (momentum_slope > 0),
        (st_efficiency > 0.25) & (momentum_slope < 0),
        (st_efficiency < 0.15),
    ]
    choices = [
        'ST_IMPULSE_UP', 'ST_IMPULSE_DOWN',
        'ST_DRIFT_UP', 'ST_DRIFT_DOWN',
        'ST_CHOP'
    ]
    df['st_regime'] = np.select(conditions, choices, default='ST_PAUSE')
    return df


def get_regime_action(lt, st):
    """Look up the interaction matrix for a (LT, ST) regime pair."""
    action = REGIME_INTERACTION.get((lt, st))
    if action is None:
        # Fallback: NO_TRADE for any unmapped combination
        action = 'NO_TRADE'
    return action


def check_pullback_entry(df, idx, lt_direction):
    """
    Check if current bar is a valid pullback entry point.
    Requirements: price near EMA50 + reversal candle pattern.
    """
    row = df.iloc[idx]
    atr = row.get('atr', row.get('atr_14', 0))
    if atr == 0 or np.isnan(atr):
        return False

    pullback_to_ma = abs(row['close'] - row['ema_50']) / atr < 0.5

    body = abs(row['close'] - row['open'])
    bar_range = row['high'] - row['low']
    if bar_range == 0:
        return False
    body_ratio = body / bar_range
    lower_wick = min(row['open'], row['close']) - row['low']
    upper_wick = row['high'] - max(row['open'], row['close'])
    lower_wick_ratio = lower_wick / bar_range
    upper_wick_ratio = upper_wick / bar_range

    ema_12 = row.get('ema_12', row['close'])

    if lt_direction == 'UP':
        reversal_candle = (
            row['close'] > row['open'] and
            body_ratio > 0.5 and
            lower_wick_ratio > 0.3 and
            row['close'] > ema_12
        )
    else:
        reversal_candle = (
            row['close'] < row['open'] and
            body_ratio > 0.5 and
            upper_wick_ratio > 0.3 and
            row['close'] < ema_12
        )

    return pullback_to_ma and reversal_candle


def get_regime_size_multiplier(regime_action):
    """Extract position size multiplier from regime action."""
    size_map = {
        'FULL': 1.0,
        'MODERATE': 0.75,
        'HALF': 0.5,
    }
    for key, mult in size_map.items():
        if key in regime_action:
            return mult
    return 0.75  # default

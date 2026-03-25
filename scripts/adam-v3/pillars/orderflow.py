"""
PILLAR 2: Orderflow from OHLC (17 Indicators)
==============================================
CLV-based delta model, multi-window CVD, pressure ratios,
absorption, Wyckoff, divergence, and composite flow score.
"""

import numpy as np
import pandas as pd


def compute_orderflow(df):
    """Compute all 17 orderflow indicators from OHLC data."""
    h, l, o, c = df['high'], df['low'], df['open'], df['close']
    bar_range = h - l
    body = abs(c - o)
    body_ratio = (body / bar_range.replace(0, np.nan)).fillna(0)

    # 1. Close Location Value: +1=high, -1=low, 0=mid
    clv = (((c - l) - (h - c)) / bar_range.replace(0, np.nan)).fillna(0)
    df['clv'] = clv

    # 2. Bar delta: CLV x relative range x conviction
    range_norm = bar_range / bar_range.rolling(20).mean()
    df['bar_delta'] = clv * range_norm * (0.5 + 0.5 * body_ratio)

    # 3. Multi-window CVD
    df['cvd_12'] = df['bar_delta'].rolling(12).sum()    # 1 hour
    df['cvd_48'] = df['bar_delta'].rolling(48).sum()    # 4 hours
    df['cvd_96'] = df['bar_delta'].rolling(96).sum()    # 8 hours

    # 4. CVD slope (acceleration)
    df['cvd_slope_fast'] = df['cvd_12'] - df['cvd_12'].shift(3)
    df['cvd_slope_slow'] = df['cvd_48'] - df['cvd_48'].shift(6)

    # 5. Buying/selling pressure
    bp = c - l
    sp = h - c
    bp_sum = bp.rolling(12).sum()
    sp_sum = sp.rolling(12).sum()
    df['pressure_ratio'] = bp_sum / (bp_sum + sp_sum).replace(0, np.nan)
    df['pressure_momentum'] = df['pressure_ratio'].diff(5)

    # 6. Enhanced delta + slope
    df['enhanced_delta'] = clv * bar_range
    df['enhanced_delta_cum'] = df['enhanced_delta'].rolling(20).sum()
    df['delta_slope'] = df['enhanced_delta_cum'].diff(5)

    # 7. Delta divergence
    price_high_20 = c.rolling(20).max()
    price_low_20 = c.rolling(20).min()
    cvd_high_20 = df['cvd_48'].rolling(20).max()
    cvd_low_20 = df['cvd_48'].rolling(20).min()
    df['bearish_div'] = ((c >= price_high_20 * 0.9995) & (df['cvd_48'] < cvd_high_20 * 0.7)).astype(float)
    df['bullish_div'] = ((c <= price_low_20 * 1.0005) & (df['cvd_48'] > cvd_low_20 * 0.7)).astype(float)

    # 8. Absorption ratio
    displacement_5 = abs(c - c.shift(5))
    range_sum_5 = bar_range.rolling(5).sum()
    df['absorption_ratio'] = 1 - (displacement_5 / range_sum_5.replace(0, np.nan))

    # 9. Wick rejection at swing levels
    swing_high = h.rolling(20).max()
    swing_low = l.rolling(20).min()
    atr = df['atr'] if 'atr' in df.columns else df.get('atr_14', bar_range.rolling(14).mean())
    dist_sh = (swing_high - c) / atr.replace(0, np.nan)
    dist_sl = (c - swing_low) / atr.replace(0, np.nan)
    proximity_low = np.exp(-dist_sl.clip(lower=0))
    proximity_high = np.exp(-dist_sh.clip(lower=0))
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    df['bull_rejection'] = (lower_wick / bar_range.replace(0, np.nan)) * proximity_low
    df['bear_rejection'] = (upper_wick / bar_range.replace(0, np.nan)) * proximity_high
    df['rejection_score'] = (df['bull_rejection'] - df['bear_rejection']).rolling(3).mean()

    # 10. Absorption candle detector
    prev_dir = np.sign(c - o).shift(1)
    curr_dir = np.sign(c - o)
    range_exp = bar_range > bar_range.rolling(10).mean()
    small_body = body_ratio < 0.3
    dir_flip = (prev_dir * curr_dir) < 0
    abs_candle = (range_exp & small_body & dir_flip).astype(float)
    df['absorption_bull'] = (abs_candle * (curr_dir > 0)).rolling(5).sum()
    df['absorption_bear'] = (abs_candle * (curr_dir < 0)).rolling(5).sum()
    df['absorption_bias'] = df['absorption_bull'] - df['absorption_bear']

    # 11. Range expansion sequence
    range_exp_seq = (bar_range > bar_range.shift(1)).astype(int)

    def count_consec(s):
        result = np.zeros(len(s))
        cnt = 0
        for i in range(len(s)):
            cnt = cnt + 1 if s.iloc[i] == 1 else 0
            result[i] = cnt
        return pd.Series(result, index=s.index)

    df['range_expansion_streak'] = count_consec(range_exp_seq)
    df['expansion_intensity'] = np.clip(df['range_expansion_streak'] / 4.0, 0, 1)
    df['bull_expansion_streak'] = count_consec((range_exp_seq & (curr_dir > 0)).astype(int))
    df['bear_expansion_streak'] = count_consec((range_exp_seq & (curr_dir < 0)).astype(int))

    # 12. Candle clustering (consolidation detection)
    prev_high = h.shift(1)
    prev_low = l.shift(1)
    overlap = np.maximum(np.minimum(h, prev_high) - np.maximum(l, prev_low), 0)
    combined = np.maximum(h, prev_high) - np.minimum(l, prev_low)
    df['cluster_score'] = (overlap / combined.replace(0, np.nan)).rolling(8).mean()
    rel_size = bar_range / bar_range.rolling(20).mean()
    df['cluster_breakout'] = ((df['cluster_score'].shift(1) > 0.6) & (rel_size > 1.5)).astype(float)

    # 13. Wyckoff spring/upthrust
    recent_low = l.rolling(20).min()
    recent_high = h.rolling(20).max()
    df['wyckoff_spring'] = ((l < recent_low.shift(1)) & (c > recent_low.shift(1))).astype(float).rolling(3).max()
    df['wyckoff_upthrust'] = ((h > recent_high.shift(1)) & (c < recent_high.shift(1))).astype(float).rolling(3).max()
    range_ma_s = bar_range.rolling(5).mean()
    range_ma_l = bar_range.rolling(20).mean()
    df['range_trend'] = range_ma_s / range_ma_l.replace(0, np.nan)
    df['wyckoff_accum'] = (1 - df['range_trend'].clip(0, 2)/2)*0.4 + df['wyckoff_spring']*0.3 + df['bull_rejection'].clip(0,1)*0.3
    df['wyckoff_distrib'] = (1 - df['range_trend'].clip(0, 2)/2)*0.4 + df['wyckoff_upthrust']*0.3 + df['bear_rejection'].clip(0,1)*0.3

    # 14. Effort vs Result
    effort = bar_range
    result_col = abs(c - c.shift(1))
    df['evr_ratio'] = result_col / effort.replace(0, np.nan)
    df['evr_ma5'] = df['evr_ratio'].rolling(5).mean()
    df['evr_slope'] = df['evr_ma5'].diff(5)

    # 15. Directional imbalance
    weighted_dir = (c - o) / bar_range.replace(0, np.nan)
    df['directional_imbalance_6'] = weighted_dir.rolling(6).mean()
    df['directional_imbalance_12'] = weighted_dir.rolling(12).mean()
    df['imbalance_accel'] = df['directional_imbalance_6'] - df['directional_imbalance_12']

    # 16. Failed auction
    midpoint = (h + l) / 2
    close_from_mid = abs(c - midpoint) / bar_range.replace(0, np.nan)
    large_bar = bar_range > bar_range.rolling(20).mean() * 1.5
    df['failed_auction'] = large_bar.astype(float) * (1 - close_from_mid)
    df['failed_auction_bull'] = df['failed_auction'] * (c > midpoint).astype(float)
    df['failed_auction_bear'] = df['failed_auction'] * (c < midpoint).astype(float)

    # 17. Composite flow score (-100 to +100)
    cvd_dir = np.sign(df['cvd_48'])
    cvd_acc = np.sign(df['cvd_slope_fast'])
    candle_conv = (clv * body_ratio).rolling(5).mean()
    df['flow_score'] = (cvd_dir * 40 + cvd_acc * 30 + candle_conv * 30).clip(-100, 100)

    return df


def check_flow_alignment(flow_score, cvd_slope_fast, regime_action,
                         bearish_div, bullish_div):
    """
    Returns flow confidence multiplier.
    0.0=veto, 0.3=low, 0.7=moderate, 1.0=confirmed, 1.2=enhanced.
    """
    if 'LONG' in regime_action and bearish_div:
        return 0.0
    if 'SHORT' in regime_action and bullish_div:
        return 0.0

    if 'LONG' in regime_action and 'TRADE' in regime_action:
        if flow_score > 30 and cvd_slope_fast > 0:
            return 1.0
        elif flow_score > 0:
            return 0.7
        elif flow_score < -30:
            return 0.0
        return 0.3

    elif 'SHORT' in regime_action and 'TRADE' in regime_action:
        if flow_score < -30 and cvd_slope_fast < 0:
            return 1.0
        elif flow_score < 0:
            return 0.7
        elif flow_score > 30:
            return 0.0
        return 0.3

    elif 'PULLBACK' in regime_action:
        if 'LONG' in regime_action and flow_score > 10:
            return 1.2
        elif 'SHORT' in regime_action and flow_score < -10:
            return 1.2
        return 0.5

    return 0.0


def compute_exit_indicators(df):
    """Exit-specific indicators for momentum decay and conviction tracking."""
    c, o = df['close'], df['open']
    bar_range = df['high'] - df['low']
    body_ratio = (abs(c - o) / bar_range.replace(0, np.nan)).fillna(0)

    body_ratio_slope = body_ratio.rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=False
    )
    df['momentum_decay'] = np.clip(-body_ratio_slope * 10, 0, 1)

    direction_sign = np.sign(c - o)
    df['candle_conviction'] = body_ratio * direction_sign
    df['conviction_ma5'] = df['candle_conviction'].rolling(5).mean()

    atr = df['atr'] if 'atr' in df.columns else df.get('atr_14', bar_range.rolling(14).mean())
    df['vol_regime_ratio'] = atr / atr.rolling(100).mean().replace(0, np.nan)
    return df

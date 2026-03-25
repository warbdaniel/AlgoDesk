"""
PILLAR 4: Liquidity Interaction Model
======================================
Liquidity pool mapping, sweep detection, interaction classification,
sweep reversal entries, and liquidity-aware stop placement.
"""

import numpy as np
import pandas as pd


def compute_liquidity_map(df, atr):
    """
    Map where stop-loss clusters likely exist.
    Liquidity pools form at swing highs/lows, equal highs/lows,
    session highs/lows, PDH/PDL, and round numbers.
    """
    h, l, c = df['high'], df['low'], df['close']
    levels = []

    # ── SWING HIGHS/LOWS (primary liquidity pools) ──
    for lookback in [10, 20, 50]:
        window = 2 * lookback + 1
        if len(h) < window:
            continue
        swing_h = h[(h == h.rolling(window, center=True).max())]
        for idx_val in swing_h.index:
            levels.append({
                'price': swing_h[idx_val], 'type': 'resistance',
                'source': f'swing_high_{lookback}',
                'bar_idx': idx_val, 'strength': lookback / 50.0,
            })
        swing_l = l[(l == l.rolling(window, center=True).min())]
        for idx_val in swing_l.index:
            levels.append({
                'price': swing_l[idx_val], 'type': 'support',
                'source': f'swing_low_{lookback}',
                'bar_idx': idx_val, 'strength': lookback / 50.0,
            })

    # ── EQUAL HIGHS / EQUAL LOWS (dense pools) ──
    atr_val = atr.iloc[-1] if len(atr) > 0 else 0
    tolerance = atr_val * 0.2 if atr_val > 0 else 0

    resistance_levels = [lev for lev in levels if lev['type'] == 'resistance']
    for i, lev_a in enumerate(resistance_levels):
        for lev_b in resistance_levels[i+1:]:
            if abs(lev_a['price'] - lev_b['price']) < tolerance:
                avg_price = (lev_a['price'] + lev_b['price']) / 2
                levels.append({
                    'price': avg_price, 'type': 'resistance',
                    'source': 'equal_highs', 'strength': 1.5,
                })

    support_levels = [lev for lev in levels if lev['type'] == 'support']
    for i, lev_a in enumerate(support_levels):
        for lev_b in support_levels[i+1:]:
            if abs(lev_a['price'] - lev_b['price']) < tolerance:
                levels.append({
                    'price': (lev_a['price'] + lev_b['price']) / 2,
                    'type': 'support', 'source': 'equal_lows', 'strength': 1.5,
                })

    # ── SESSION HIGHS/LOWS ──
    if 'timestamp' in df.columns:
        df_temp = df.copy()
        df_temp['hour'] = df_temp['timestamp'].dt.hour
        df_temp['date'] = df_temp['timestamp'].dt.date

        for date in sorted(df_temp['date'].unique())[-3:]:
            day = df_temp[df_temp['date'] == date]
            sessions = {
                'asian':  day[(day['hour'] < 7) | (day['hour'] >= 22)],
                'london': day[(day['hour'] >= 7) & (day['hour'] < 16)],
                'ny':     day[(day['hour'] >= 12) & (day['hour'] < 21)],
            }
            for sname, sdata in sessions.items():
                if len(sdata) > 0:
                    levels.append({'price': sdata['high'].max(), 'type': 'resistance',
                                 'source': f'{sname}_high_{date}', 'strength': 1.0})
                    levels.append({'price': sdata['low'].min(), 'type': 'support',
                                 'source': f'{sname}_low_{date}', 'strength': 1.0})

        # ── PREVIOUS DAY HIGH/LOW ──
        dates = sorted(df_temp['date'].unique())
        if len(dates) >= 2:
            prev_day = df_temp[df_temp['date'] == dates[-2]]
            levels.append({'price': prev_day['high'].max(), 'type': 'resistance',
                         'source': 'pdh', 'strength': 2.0})
            levels.append({'price': prev_day['low'].min(), 'type': 'support',
                         'source': 'pdl', 'strength': 2.0})

    # ── ROUND NUMBERS ──
    current_price = c.iloc[-1]
    symbol_str = str(df.get('symbol', ''))
    if 'JPY' in symbol_str:
        round_step = 0.50
    elif 'XAU' in symbol_str:
        round_step = 10.0
    else:
        round_step = 0.0050

    nearest_round = round(current_price / round_step) * round_step
    for offset in range(-3, 4):
        rnd = nearest_round + offset * round_step
        dist = abs(rnd - current_price) / atr_val if atr_val > 0 else 999
        if dist < 5.0:
            levels.append({'price': rnd,
                         'type': 'resistance' if rnd > current_price else 'support',
                         'source': 'round_number', 'strength': 0.7})

    return levels


def classify_liquidity_interaction(df, liquidity_levels, idx, atr_val):
    """
    Classify how price interacted with nearby liquidity pools.
    Three interactions: SWEEP, ACCEPTANCE, REJECTION.
    """
    row = df.iloc[idx]
    h, l, o, c = row['high'], row['low'], row['open'], row['close']
    interactions = []

    for pool in liquidity_levels:
        distance = abs(c - pool['price']) / atr_val if atr_val > 0 else 999
        if distance > 2.5:
            continue

        price = pool['price']
        body = abs(c - o)
        bar_range = h - l
        body_ratio = body / bar_range if bar_range > 0 else 0

        if pool['type'] == 'resistance':
            if h >= price:
                if c > price and c > o:
                    interactions.append({
                        'level': price, 'type': 'ACCEPTANCE_BULL',
                        'confidence': min(body_ratio, 1.0),
                        'source': pool['source'], 'strength': pool['strength'],
                    })
                elif c < price:
                    penetration = (h - price) / atr_val if atr_val > 0 else 0
                    rejection = (h - c) / (h - l) if h != l else 0
                    interactions.append({
                        'level': price, 'type': 'SWEEP_BEAR',
                        'confidence': min(penetration * rejection * 2, 1.0),
                        'source': pool['source'], 'strength': pool['strength'],
                    })
            elif h >= price - atr_val * 0.3:
                interactions.append({
                    'level': price, 'type': 'REJECTION_BELOW',
                    'confidence': 0.5, 'source': pool['source'], 'strength': pool['strength'],
                })

        elif pool['type'] == 'support':
            if l <= price:
                if c < price and c < o:
                    interactions.append({
                        'level': price, 'type': 'ACCEPTANCE_BEAR',
                        'confidence': min(body_ratio, 1.0),
                        'source': pool['source'], 'strength': pool['strength'],
                    })
                elif c > price:
                    penetration = (price - l) / atr_val if atr_val > 0 else 0
                    rejection = (c - l) / (h - l) if h != l else 0
                    interactions.append({
                        'level': price, 'type': 'SWEEP_BULL',
                        'confidence': min(penetration * rejection * 2, 1.0),
                        'source': pool['source'], 'strength': pool['strength'],
                    })
            elif l <= price + atr_val * 0.3:
                interactions.append({
                    'level': price, 'type': 'REJECTION_ABOVE',
                    'confidence': 0.5, 'source': pool['source'], 'strength': pool['strength'],
                })

    return interactions


def check_sweep_reversal_entry(df, idx, lt_regime, interactions, flow_score, atr_val):
    """
    Post-sweep entry: institutions swept liquidity and now driving opposite direction.
    Highest conviction, tightest stops, best R:R.
    """
    row = df.iloc[idx]
    recent_sweeps = [i for i in interactions
                     if 'SWEEP' in i['type'] and i['confidence'] > 0.6]
    if not recent_sweeps:
        return None

    sweep = recent_sweeps[-1]

    if sweep['type'] == 'SWEEP_BULL':
        if lt_regime not in ('LT_TREND_UP', 'LT_TRANSITION', 'LT_RANGE'):
            return None
        if flow_score < 10:
            return None
        return {
            'action': 'long',
            'entry': row['close'],
            'sl': sweep['level'] - atr_val * 0.3,
            'entry_type': 'sweep_reversal',
            'confidence': sweep['confidence'] * 1.3,
        }

    elif sweep['type'] == 'SWEEP_BEAR':
        if lt_regime not in ('LT_TREND_DOWN', 'LT_TRANSITION', 'LT_RANGE'):
            return None
        if flow_score > -10:
            return None
        return {
            'action': 'short',
            'entry': row['close'],
            'sl': sweep['level'] + atr_val * 0.3,
            'entry_type': 'sweep_reversal',
            'confidence': sweep['confidence'] * 1.3,
        }

    return None


def compute_liquidity_aware_stop(entry_price, direction, liquidity_levels, atr_val):
    """Place stops BEYOND liquidity pools, not AT them."""
    sweep_buffer = atr_val * 0.5

    if direction == 'long':
        pools_below = [lev for lev in liquidity_levels
                      if lev['price'] < entry_price and lev['type'] == 'support']
        if pools_below:
            deepest = min(pools_below, key=lambda x: x['price'])
            sl = deepest['price'] - sweep_buffer
            return max(sl, entry_price - atr_val * 3.0)
    else:
        pools_above = [lev for lev in liquidity_levels
                      if lev['price'] > entry_price and lev['type'] == 'resistance']
        if pools_above:
            highest = max(pools_above, key=lambda x: x['price'])
            sl = highest['price'] + sweep_buffer
            return min(sl, entry_price + atr_val * 3.0)

    return None


def filter_regime_for_sweeps(st_regime, recent_interactions):
    """If a sweep just occurred against the ST regime, downgrade the signal."""
    recent_sweeps = [i for i in recent_interactions
                     if 'SWEEP' in i['type'] and i['confidence'] > 0.5]
    if not recent_sweeps:
        return st_regime

    sweep = recent_sweeps[-1]
    if st_regime == 'ST_IMPULSE_UP' and sweep['type'] == 'SWEEP_BEAR':
        return 'ST_PAUSE'
    if st_regime == 'ST_IMPULSE_DOWN' and sweep['type'] == 'SWEEP_BULL':
        return 'ST_PAUSE'

    return st_regime

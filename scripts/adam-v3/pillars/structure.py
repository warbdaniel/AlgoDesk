"""
PILLAR 3: Structural Levels & Targets
======================================
TPO market profiles, multi-system pivot points, level confluence scoring.
"""

import numpy as np
import pandas as pd

TICK_SIZES = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDCHF': 0.0001,
    'AUDUSD': 0.0001, 'USDCAD': 0.0001, 'NZDUSD': 0.0001,
    'EURJPY': 0.01,   'GBPJPY': 0.01,   'USDJPY': 0.01,
    'XAUUSD': 0.10,
}


def compute_tpo_profile(session_bars, tick_size):
    """Compute TPO profile for a set of bars. Returns POC, VAH, VAL."""
    price_levels = {}
    for _, bar in session_bars.iterrows():
        bar_low = np.floor(bar['low'] / tick_size) * tick_size
        bar_high = np.ceil(bar['high'] / tick_size) * tick_size
        level = bar_low
        while level <= bar_high:
            rounded = round(level, 6)
            price_levels[rounded] = price_levels.get(rounded, 0) + 1
            level += tick_size
    if not price_levels:
        return None

    poc = max(price_levels, key=price_levels.get)
    total_tpos = sum(price_levels.values())
    target_tpos = total_tpos * 0.70
    sorted_levels = sorted(price_levels.items(), key=lambda x: x[1], reverse=True)

    va_tpos, va_levels = 0, []
    for level, count in sorted_levels:
        va_levels.append(level)
        va_tpos += count
        if va_tpos >= target_tpos:
            break

    val, vah = min(va_levels), max(va_levels)
    poc_position = (poc - val) / (vah - val) if vah != val else 0.5
    if 0.35 < poc_position < 0.65:
        profile_type = 'BALANCED'
    elif poc_position <= 0.35:
        profile_type = 'SELLER_CONTROL'
    else:
        profile_type = 'BUYER_CONTROL'

    return {'poc': poc, 'vah': vah, 'val': val,
            'profile_type': profile_type, 'poc_position': poc_position}


def compute_multi_context_profiles(df, tick_size):
    """Compute daily and session TPO profiles for last 5 days."""
    if 'timestamp' not in df.columns:
        return {}, []
    df_temp = df.copy()
    df_temp['hour'] = df_temp['timestamp'].dt.hour
    df_temp['date'] = df_temp['timestamp'].dt.date

    profiles = {}
    all_levels = []

    for date in sorted(df_temp['date'].unique())[-5:]:
        day_data = df_temp[df_temp['date'] == date]
        p = compute_tpo_profile(day_data, tick_size)
        if p:
            profiles[f'daily_{date}'] = p
            for key, lt in [('poc','POC'),('vah','VAH'),('val','VAL')]:
                all_levels.append({'price': p[key], 'type': lt,
                    'context': f'daily_{date}', 'weight': 2.0 if lt=='POC' else 1.6,
                    'source': 'profile'})

        sessions = {
            'london': day_data[(day_data['hour'] >= 7) & (day_data['hour'] < 16)],
            'ny':     day_data[(day_data['hour'] >= 12) & (day_data['hour'] < 21)],
            'asian':  day_data[(day_data['hour'] < 7) | (day_data['hour'] >= 22)],
        }
        for sname, sdata in sessions.items():
            if len(sdata) > 6:
                p = compute_tpo_profile(sdata, tick_size)
                if p:
                    profiles[f'{sname}_{date}'] = p
                    for key, lt in [('poc','POC'),('vah','VAH'),('val','VAL')]:
                        all_levels.append({'price': p[key], 'type': lt,
                            'context': f'{sname}_{date}', 'weight': 1.5 if lt=='POC' else 1.2,
                            'source': 'profile'})

    return profiles, all_levels


def compute_pivots(daily_high, daily_low, daily_close):
    """Compute classic, camarilla, and fibonacci pivot points."""
    pp = (daily_high + daily_low + daily_close) / 3
    range_ = daily_high - daily_low
    return {
        'classic': {
            'PP': pp, 'R1': 2*pp - daily_low, 'S1': 2*pp - daily_high,
            'R2': pp + range_, 'S2': pp - range_,
            'R3': daily_high + 2*(pp - daily_low), 'S3': daily_low - 2*(daily_high - pp),
        },
        'camarilla': {
            'R1': daily_close + range_*1.1/12, 'R2': daily_close + range_*1.1/6,
            'R3': daily_close + range_*1.1/4,  'R4': daily_close + range_*1.1/2,
            'S1': daily_close - range_*1.1/12, 'S2': daily_close - range_*1.1/6,
            'S3': daily_close - range_*1.1/4,  'S4': daily_close - range_*1.1/2,
        },
        'fibonacci': {
            'PP': pp,
            'R1': pp + 0.382*range_, 'R2': pp + 0.618*range_, 'R3': pp + range_,
            'S1': pp - 0.382*range_, 'S2': pp - 0.618*range_, 'S3': pp - range_,
        },
    }


def score_level_confluence(profile_levels, pivot_levels, current_price, atr,
                           tolerance_atr=0.3):
    """Score confluence of structural levels. Returns sorted list of clusters."""
    tolerance = atr * tolerance_atr
    all_levels = list(profile_levels)

    for system_name, levels in pivot_levels.items():
        for name, price in levels.items():
            all_levels.append({'price': price, 'source': f'pivot_{system_name}',
                             'detail': name, 'weight': 1.0})
    if not all_levels:
        return []

    sorted_levels = sorted(all_levels, key=lambda x: x['price'])
    clusters, current_cluster = [], [sorted_levels[0]]
    for level in sorted_levels[1:]:
        if abs(level['price'] - current_cluster[-1]['price']) <= tolerance:
            current_cluster.append(level)
        else:
            clusters.append(current_cluster)
            current_cluster = [level]
    clusters.append(current_cluster)

    scored = []
    for cluster in clusters:
        avg_price = np.mean([l['price'] for l in cluster])
        total_weight = sum(l['weight'] for l in cluster)
        n_sources = len(set(l.get('source', l.get('context', '')) for l in cluster))
        scored.append({
            'price': round(avg_price, 6),
            'score': round(total_weight * n_sources, 2),
            'n_sources': n_sources,
            'distance_atr': round(abs(avg_price - current_price) / atr, 2) if atr > 0 else 999,
            'is_above': avg_price > current_price,
            'components': [l.get('detail', l.get('type', '')) for l in cluster],
        })
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored


def compute_structural_targets(entry_price, direction, scored_levels, atr,
                                liquidity_levels=None):
    """Use structural levels for SL/TP placement."""
    if direction == 'long':
        targets = sorted([l for l in scored_levels if l['is_above'] and l['distance_atr'] > 0.5],
                        key=lambda x: x['distance_atr'])
        stops = sorted([l for l in scored_levels if not l['is_above'] and l['distance_atr'] < 3.0],
                      key=lambda x: x['distance_atr'])
    else:
        targets = sorted([l for l in scored_levels if not l['is_above'] and l['distance_atr'] > 0.5],
                        key=lambda x: x['distance_atr'])
        stops = sorted([l for l in scored_levels if l['is_above'] and l['distance_atr'] < 3.0],
                      key=lambda x: x['distance_atr'])

    tp = next((t['price'] for t in targets if t['score'] >= 3.0), None)
    if tp is None:
        tp = entry_price + (2.0 * atr * (1 if direction == 'long' else -1))

    sl = None
    for s in stops:
        if s['score'] >= 2.0:
            sl = s['price'] + (-0.3 * atr if direction == 'long' else 0.3 * atr)
            break

    if sl is None:
        sl = entry_price + (-1.5 * atr if direction == 'long' else 1.5 * atr)

    reward = abs(tp - entry_price)
    risk = abs(entry_price - sl)
    rr_ratio = reward / risk if risk > 0 else 0

    return {'tp': tp, 'sl': sl, 'rr_ratio': rr_ratio,
            'tp_structural': tp is not None, 'sl_structural': sl is not None}

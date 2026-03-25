"""
PILLAR 5: Session & Time-of-Day Awareness
==========================================
Markets behave differently at different times. Session detection,
killzone windows, time-of-day volatility profiles, event filtering.
"""

import numpy as np
import pandas as pd

SESSIONS = {
    'asian_early':     {'start': 22, 'end': 2,  'quality': 0.2},
    'asian_late':      {'start': 2,  'end': 7,  'quality': 0.3},
    'london_open':     {'start': 7,  'end': 9,  'quality': 1.0},
    'london_mid':      {'start': 9,  'end': 12, 'quality': 0.7},
    'overlap':         {'start': 12, 'end': 16, 'quality': 0.9},
    'ny_mid':          {'start': 16, 'end': 19, 'quality': 0.5},
    'ny_close':        {'start': 19, 'end': 22, 'quality': 0.1},
}

SESSION_OVERRIDES = {
    'XAUUSD': {'asian_early': 0.5, 'asian_late': 0.6, 'london_open': 1.0, 'overlap': 1.0},
    'USDJPY': {'asian_early': 0.6, 'asian_late': 0.7},
    'EURJPY': {'asian_early': 0.5, 'asian_late': 0.6},
}

PAIR_CURRENCY_MAP = {
    'EURUSD': ('EUR', 'USD'), 'GBPUSD': ('GBP', 'USD'),
    'USDCHF': ('USD', 'CHF'), 'AUDUSD': ('AUD', 'USD'),
    'USDCAD': ('USD', 'CAD'), 'NZDUSD': ('NZD', 'USD'),
    'EURJPY': ('EUR', 'JPY'), 'GBPJPY': ('GBP', 'JPY'),
    'USDJPY': ('USD', 'JPY'), 'XAUUSD': ('XAU', 'USD'),
}


def _get_session(h):
    for name, s in SESSIONS.items():
        if s['start'] <= s['end']:
            if s['start'] <= h < s['end']:
                return name
        else:
            if h >= s['start'] or h < s['end']:
                return name
    return 'unknown'


def _get_quality(h, sym):
    session = _get_session(h)
    if sym in SESSION_OVERRIDES and session in SESSION_OVERRIDES[sym]:
        return SESSION_OVERRIDES[sym][session]
    return SESSIONS.get(session, {}).get('quality', 0.3)


def compute_session_features(df, symbol=''):
    """Add session-awareness indicators to dataframe."""
    hour = df['timestamp'].dt.hour
    df['session'] = hour.apply(_get_session)
    df['session_quality'] = hour.apply(lambda h: _get_quality(h, symbol))
    df['is_killzone'] = df['session'].isin(['london_open', 'overlap']).astype(float)

    session_starts = {'london_open': 7, 'overlap': 12, 'ny_mid': 16}
    def mins_since_session_start(row):
        sess = row['session']
        if sess in session_starts:
            start_h = session_starts[sess]
            h = row['timestamp'].hour
            m = row['timestamp'].minute
            return (h - start_h) * 60 + m
        return 0
    df['mins_in_session'] = df.apply(mins_since_session_start, axis=1)

    df['hour'] = hour
    if 'atr' in df.columns:
        hourly_avg_atr = df.groupby('hour')['atr'].transform('mean')
        df['tod_vol_ratio'] = df['atr'] / hourly_avg_atr.replace(0, np.nan)
    else:
        df['tod_vol_ratio'] = 1.0

    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['friday_risk'] = ((df['day_of_week'] == 4) & (hour >= 18)).astype(float)
    return df


def session_gate(session_quality, is_killzone, entry_type, friday_risk):
    """Returns position size multiplier based on session timing. 0.0 = don't trade."""
    if friday_risk:
        return 0.0
    if session_quality < 0.2:
        return 0.0
    if entry_type == 'sweep_reversal':
        return max(session_quality, 0.5)
    if entry_type == 'breakout' and not is_killzone:
        return session_quality * 0.5
    return session_quality


def is_event_window(timestamp, symbol, event_calendar=None):
    """Check if current time is within a high-impact event window."""
    if event_calendar is None:
        return False
    base, quote = PAIR_CURRENCY_MAP.get(symbol, ('', ''))
    for event in event_calendar:
        event_time = event['timestamp']
        before_mins = event.get('before_mins', 30)
        after_mins = event.get('after_mins', 60)
        window_start = event_time - pd.Timedelta(minutes=before_mins)
        window_end = event_time + pd.Timedelta(minutes=after_mins)
        if window_start <= timestamp <= window_end:
            affected = event.get('currencies', [])
            if base in affected or quote in affected:
                return True
    return False

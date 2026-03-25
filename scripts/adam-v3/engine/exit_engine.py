"""
SMART EXIT ENGINE (6 Layers)
=============================
L1: Hard SL/TP (always active)
L2: ATR trailing stop (ratchet)
L3: Momentum decay (candle conviction)
L4: Time decay (stale trade cleanup)
L5: Structural level reaction
L6: Liquidity level reaction
"""

import numpy as np


def evaluate_exit(df, i, position, indicators, params, scored_levels=None,
                  current_interactions=None, symbol='', session=''):
    """
    Per-bar exit evaluation across all 6 layers.
    Returns: (exit_reason, exit_price) or (None, None) if no exit.
    """
    from ..pillars.portfolio import estimate_trade_cost

    close = df['close'].iloc[i]
    high = df['high'].iloc[i]
    low = df['low'].iloc[i]
    atr_col = 'atr' if 'atr' in df.columns else 'atr_14'
    atr = df[atr_col].iloc[i] if atr_col in df.columns else 0
    d = position['direction']
    pip_size = params.get('pip_size', 0.0001)

    if atr == 0 or np.isnan(atr):
        return None, None

    unrealized_atr = (close - position['entry']) * d / atr
    bars_in_trade = i - position['bar']
    exit_reason = None
    exit_price = None

    # Extract params
    trailing_stop_atr = params.get('trailing_stop_atr', 1.5)
    trail_activate_atr = params.get('trail_activate_atr', 1.0)
    momentum_exit_thresh = params.get('momentum_exit_thresh', 0.3)
    time_decay_bars = params.get('time_decay_bars', 24)
    time_decay_min_atr = params.get('time_decay_min_atr', 0.5)

    # Update watermarks
    if d == 1:
        position['highest_since_entry'] = max(
            position.get('highest_since_entry', high), high)
    else:
        position['lowest_since_entry'] = min(
            position.get('lowest_since_entry', low), low)

    # ── L1: HARD SL/TP (always active) ──
    if d == 1:
        if low <= position['sl']:
            exit_reason, exit_price = 'sl_hit', position['sl']
        elif high >= position['tp']:
            exit_reason, exit_price = 'tp_hit', position['tp']
    else:
        if high >= position['sl']:
            exit_reason, exit_price = 'sl_hit', position['sl']
        elif low <= position['tp']:
            exit_reason, exit_price = 'tp_hit', position['tp']

    # ── L2: ATR TRAILING STOP (ratchet) ──
    if exit_reason is None:
        if unrealized_atr >= trail_activate_atr:
            position['trail_active'] = True
        if position.get('trail_active', False):
            if d == 1:
                trail_sl = position['highest_since_entry'] - atr * trailing_stop_atr
                position['sl'] = max(position['sl'], trail_sl)
            else:
                trail_sl = position['lowest_since_entry'] + atr * trailing_stop_atr
                position['sl'] = min(position['sl'], trail_sl)

    # ── L3: MOMENTUM DECAY EXIT ──
    if exit_reason is None and position.get('trail_active', False):
        conviction = 0
        decay = 0
        if 'conviction_ma5' in indicators:
            conviction = indicators['conviction_ma5'].iloc[i] if i < len(indicators['conviction_ma5']) else 0
        if 'momentum_decay' in indicators:
            decay = indicators['momentum_decay'].iloc[i] if i < len(indicators['momentum_decay']) else 0

        if d == 1 and conviction < -momentum_exit_thresh and decay > 0.5:
            exit_reason, exit_price = 'momentum_decay', close
        elif d == -1 and conviction > momentum_exit_thresh and decay > 0.5:
            exit_reason, exit_price = 'momentum_decay', close

    # ── L4: TIME DECAY (stale trade) ──
    if exit_reason is None and bars_in_trade >= time_decay_bars:
        if abs(unrealized_atr) < time_decay_min_atr:
            exit_reason, exit_price = 'time_decay', close

    # ── L5: STRUCTURAL LEVEL REACTION ──
    if exit_reason is None and position.get('trail_active', False) and scored_levels:
        for level in scored_levels:
            if level['score'] >= 4.0 and level['distance_atr'] < 0.3:
                if d == 1:
                    position['sl'] = max(position['sl'], high - atr * 0.5)
                else:
                    position['sl'] = min(position['sl'], low + atr * 0.5)
                break

    # ── L6: LIQUIDITY LEVEL REACTION ──
    if exit_reason is None and position.get('trail_active', False) and current_interactions:
        for interaction in current_interactions:
            if d == 1 and interaction['type'] == 'SWEEP_BEAR' and unrealized_atr > 1.0:
                exit_reason, exit_price = 'liquidity_reversal', close
            elif d == -1 and interaction['type'] == 'SWEEP_BULL' and unrealized_atr > 1.0:
                exit_reason, exit_price = 'liquidity_reversal', close

    return exit_reason, exit_price


def build_trade_record(position, exit_price, exit_reason, i, symbol, session,
                       pip_size, atr_pips):
    """Build a standardized trade record dict."""
    from ..pillars.portfolio import estimate_trade_cost

    d = position['direction']
    cost = estimate_trade_cost(symbol, session, atr_pips)
    raw_pnl = (exit_price - position['entry']) * d / pip_size
    net_pnl = raw_pnl - cost

    return {
        'symbol': symbol,
        'direction': d,
        'entry': position['entry'],
        'exit': exit_price,
        'pnl_pips': net_pnl,
        'gross_pnl_pips': raw_pnl,
        'cost_pips': cost,
        'regime': position.get('regime', ''),
        'duration_bars': i - position['bar'],
        'exit_reason': exit_reason,
        'entry_type': position.get('entry_type', 'breakout'),
        'session': session,
    }

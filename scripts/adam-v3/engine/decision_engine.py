"""
UNIFIED DECISION ENGINE (All 7 Pillars, 8 Gates)
==================================================
Each gate can only REDUCE or KILL trades, never CREATE them.
"""

import numpy as np

from ..pillars.regime import get_regime_action, check_pullback_entry, get_regime_size_multiplier
from ..pillars.orderflow import check_flow_alignment
from ..pillars.structure import compute_structural_targets
from ..pillars.liquidity import (check_sweep_reversal_entry, filter_regime_for_sweeps,
                                  compute_liquidity_aware_stop)
from ..pillars.session import session_gate
from ..pillars.mtf import htf_confirmation_score, mtf_gate
from ..pillars.portfolio import (check_portfolio_risk, check_drawdown_limits,
                                  estimate_trade_cost, cost_adjusted_rr)


def unified_trade_decision(df, idx, scored_levels, liquidity_levels,
                           liquidity_interactions, htf_context,
                           open_positions, equity_curve, symbol, config):
    """
    Master decision function. All seven pillars converge through 8 gates.
    Returns trade dict or None.
    """
    row = df.iloc[idx]
    current_price = row['close']
    atr = row.get('atr', row.get('atr_14', 0))
    if atr == 0 or np.isnan(atr):
        return None
    pip_size = config.get('pip_size', 0.0001)
    atr_pips = atr / pip_size

    # ═══ GATE 1: REGIME ═══
    lt_regime = row.get('lt_regime', 'LT_RANGE')
    st_regime = row.get('st_regime', 'ST_PAUSE')
    st_regime = filter_regime_for_sweeps(st_regime, liquidity_interactions)
    regime_action = get_regime_action(lt_regime, st_regime)
    if regime_action == 'NO_TRADE':
        return None

    # ═══ GATE 2: SESSION FILTER ═══
    session_mult = session_gate(
        row.get('session_quality', 0.5),
        row.get('is_killzone', False),
        'breakout',
        row.get('friday_risk', False),
    )
    if session_mult == 0.0:
        return None

    # ═══ GATE 3: LIQUIDITY INTERACTION ROUTING ═══
    direction = 'long' if 'LONG' in regime_action else 'short'
    entry_type = 'breakout'
    sweep_entry = None

    sweep_entry = check_sweep_reversal_entry(
        df, idx, lt_regime, liquidity_interactions,
        row.get('flow_score', 0), atr
    )
    if sweep_entry:
        direction = sweep_entry['action']
        entry_type = 'sweep_reversal'
    elif 'PULLBACK' in regime_action or 'REVERSAL' in regime_action:
        lt_dir = 'UP' if 'LONG' in regime_action else 'DOWN'
        if check_pullback_entry(df, idx, lt_dir):
            entry_type = 'pullback'
        else:
            return None

    session_mult = session_gate(
        row.get('session_quality', 0.5),
        row.get('is_killzone', False),
        entry_type,
        row.get('friday_risk', False),
    )
    if session_mult == 0.0:
        return None

    # ═══ GATE 4: MTF CONFIRMATION ═══
    htf_score = htf_confirmation_score(htf_context, direction)
    if not mtf_gate(htf_score, entry_type):
        return None

    # ═══ GATE 5: ORDERFLOW CONFIRMATION ═══
    flow_confidence = check_flow_alignment(
        row.get('flow_score', 0), row.get('cvd_slope_fast', 0),
        regime_action, row.get('bearish_div', 0), row.get('bullish_div', 0)
    )
    if flow_confidence == 0.0:
        return None

    # ═══ GATE 6: STRUCTURAL TARGETS + R:R ═══
    targets = compute_structural_targets(
        current_price, direction, scored_levels, atr, liquidity_levels
    )
    if sweep_entry and sweep_entry.get('sl'):
        targets['sl'] = sweep_entry['sl']
        risk = abs(current_price - targets['sl'])
        reward = abs(targets['tp'] - current_price)
        targets['rr_ratio'] = reward / risk if risk > 0 else 0

    # ═══ GATE 7: COST-ADJUSTED R:R ═══
    cost = estimate_trade_cost(symbol, row.get('session', ''), atr_pips)
    adj_rr = cost_adjusted_rr(
        abs(targets['tp'] - current_price) / pip_size,
        abs(current_price - targets['sl']) / pip_size,
        cost
    )
    min_rr = config.get('min_rr_ratio', 1.5)
    if adj_rr < min_rr:
        return None

    # ═══ GATE 8: PORTFOLIO RISK ═══
    proposed = {'symbol': symbol,
                'direction': 1 if direction == 'long' else -1,
                'size_mult': 1.0}
    allowed, reason, port_adj = check_portfolio_risk(open_positions, proposed, config)
    if not allowed:
        return None

    dd_mult = check_drawdown_limits(equity_curve, config)
    if dd_mult == 0.0:
        return None

    # ═══ POSITION SIZING ═══
    regime_size = get_regime_size_multiplier(regime_action)
    final_size = (
        regime_size * flow_confidence * session_mult *
        htf_score * port_adj * dd_mult
    )
    if targets.get('tp_structural') and targets.get('sl_structural'):
        final_size *= 1.15
    final_size = min(final_size, 1.5)
    if final_size < 0.2:
        return None

    return {
        'action': direction,
        'entry': current_price,
        'sl': targets['sl'],
        'tp': targets['tp'],
        'rr_ratio': targets['rr_ratio'],
        'cost_adj_rr': adj_rr,
        'confidence': final_size,
        'size_mult': final_size,
        'regime': f"{lt_regime}/{st_regime}",
        'flow_score': row.get('flow_score', 0),
        'entry_type': entry_type,
        'session': row.get('session', 'unknown'),
        'htf_score': htf_score,
    }

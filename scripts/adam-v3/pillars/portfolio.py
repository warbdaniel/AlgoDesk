"""
PILLAR 7: Portfolio Risk Engine
================================
Cross-pair correlation limits, currency exposure, drawdown circuit breakers,
equity curve position sizing, spread/slippage cost model.
"""

import numpy as np

PAIR_CURRENCY_MAP = {
    'EURUSD': ('EUR', 'USD'), 'GBPUSD': ('GBP', 'USD'),
    'USDCHF': ('USD', 'CHF'), 'AUDUSD': ('AUD', 'USD'),
    'USDCAD': ('USD', 'CAD'), 'NZDUSD': ('NZD', 'USD'),
    'EURJPY': ('EUR', 'JPY'), 'GBPJPY': ('GBP', 'JPY'),
    'USDJPY': ('USD', 'JPY'), 'XAUUSD': ('XAU', 'USD'),
}

SPREAD_MODEL = {
    'EURUSD': {'avg_spread': 0.8, 'asian_mult': 1.5, 'news_mult': 3.0},
    'GBPUSD': {'avg_spread': 1.2, 'asian_mult': 1.8, 'news_mult': 4.0},
    'USDCHF': {'avg_spread': 1.3, 'asian_mult': 1.6, 'news_mult': 3.5},
    'AUDUSD': {'avg_spread': 1.0, 'asian_mult': 2.0, 'news_mult': 3.5},
    'USDCAD': {'avg_spread': 1.5, 'asian_mult': 1.8, 'news_mult': 4.0},
    'NZDUSD': {'avg_spread': 1.5, 'asian_mult': 2.5, 'news_mult': 4.0},
    'EURJPY': {'avg_spread': 1.5, 'asian_mult': 1.3, 'news_mult': 4.0},
    'GBPJPY': {'avg_spread': 2.0, 'asian_mult': 1.5, 'news_mult': 5.0},
    'USDJPY': {'avg_spread': 0.9, 'asian_mult': 1.3, 'news_mult': 3.0},
    'XAUUSD': {'avg_spread': 2.5, 'asian_mult': 1.5, 'news_mult': 5.0},
}

COMMISSION_PER_LOT = 7.0  # USD per round trip per lot


def compute_currency_exposure(open_positions):
    """Calculate net exposure per currency across all open positions."""
    exposure = {}
    for pos in open_positions:
        base, quote = PAIR_CURRENCY_MAP.get(pos['symbol'], ('???', '???'))
        size = pos.get('size_mult', 1.0)
        if pos['direction'] == 1:  # long
            exposure[base] = exposure.get(base, 0) + size
            exposure[quote] = exposure.get(quote, 0) - size
        else:  # short
            exposure[base] = exposure.get(base, 0) - size
            exposure[quote] = exposure.get(quote, 0) + size
    return exposure


def check_portfolio_risk(open_positions, proposed_trade, config):
    """
    Portfolio-level risk checks before allowing a new trade.
    Returns: (allowed: bool, reason: str, size_adjustment: float)
    """
    max_positions = config.get('max_positions', 6)
    max_currency_exposure = config.get('max_currency_exposure', 3.0)
    max_correlated_pairs = config.get('max_correlated_pairs', 2)

    if len(open_positions) >= max_positions:
        return False, 'max_positions_reached', 0.0

    test_positions = open_positions + [proposed_trade]
    exposure = compute_currency_exposure(test_positions)
    for currency, net in exposure.items():
        if abs(net) > max_currency_exposure:
            return False, f'currency_exposure_{currency}', 0.0

    proposed_base, proposed_quote = PAIR_CURRENCY_MAP.get(proposed_trade['symbol'], ('???', '???'))
    same_direction_count = 0
    for pos in open_positions:
        pos_base, pos_quote = PAIR_CURRENCY_MAP.get(pos['symbol'], ('???', '???'))
        if (proposed_base == pos_base or proposed_quote == pos_quote):
            if pos['direction'] == proposed_trade['direction']:
                same_direction_count += 1

    if same_direction_count >= max_correlated_pairs:
        return True, 'correlated_size_reduction', 0.5

    return True, 'ok', 1.0


def check_drawdown_limits(equity_curve, config):
    """Multi-level circuit breakers based on drawdown."""
    if len(equity_curve) < 10:
        return 1.0

    peak = max(equity_curve)
    current = equity_curve[-1]
    drawdown_pct = (peak - current) / peak * 100 if peak > 0 else 0

    if drawdown_pct > config.get('dd_red', 8.0):
        return 0.0
    if drawdown_pct > config.get('dd_orange', 5.0):
        return 0.25
    if drawdown_pct > config.get('dd_yellow', 3.0):
        return 0.5

    recent_results = config.get('recent_results', [])
    if len(recent_results) >= 5 and all(r < 0 for r in recent_results[-5:]):
        return 0.5

    return 1.0


def estimate_trade_cost(symbol, session, atr_pips):
    """Estimate total round-trip cost in pips."""
    cfg = SPREAD_MODEL.get(symbol, {'avg_spread': 1.5, 'asian_mult': 2.0, 'news_mult': 3.0})
    spread = cfg['avg_spread']
    if 'asian' in str(session):
        spread *= cfg['asian_mult']
    slippage = min(0.2 + atr_pips * 0.01, 1.0) if atr_pips > 0 else 0.2
    return spread + slippage


def cost_adjusted_rr(reward_pips, risk_pips, cost_pips):
    """Adjust R:R for transaction costs."""
    net_reward = reward_pips - cost_pips
    net_risk = risk_pips + cost_pips
    return net_reward / net_risk if net_risk > 0 else 0

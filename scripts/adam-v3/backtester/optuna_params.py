"""
Complete Optuna Parameter Space for ADAM v3
============================================
All optimizable parameters across all 7 pillars.
"""


def suggest_params(trial):
    """Suggest parameter set for a single Optuna trial."""
    return {
        # ── Regime thresholds ──
        'lt_efficiency_thresh': trial.suggest_float('lt_efficiency_thresh', 0.25, 0.50, step=0.05),
        'lt_hurst_thresh': trial.suggest_float('lt_hurst_thresh', 0.45, 0.60, step=0.05),
        'st_adx_thresh': trial.suggest_int('st_adx_thresh', 25, 40),
        'st_efficiency_thresh': trial.suggest_float('st_efficiency_thresh', 0.25, 0.50, step=0.05),

        # ── Orderflow thresholds ──
        'flow_confirm_thresh': trial.suggest_int('flow_confirm_thresh', 15, 50, step=5),
        'flow_veto_thresh': trial.suggest_int('flow_veto_thresh', -50, -15, step=5),
        'divergence_lookback': trial.suggest_int('divergence_lookback', 15, 30),

        # ── Liquidity ──
        'sweep_confidence_min': trial.suggest_float('sweep_confidence_min', 0.4, 0.8, step=0.1),
        'sweep_lookback_bars': trial.suggest_int('sweep_lookback_bars', 1, 5),

        # ── Entry ──
        'min_rr_ratio': trial.suggest_float('min_rr_ratio', 1.2, 2.5, step=0.1),
        'min_htf_score': trial.suggest_float('min_htf_score', 0.3, 0.7, step=0.1),

        # ── Exit ──
        'atr_sl_mult': trial.suggest_float('atr_sl_mult', 1.0, 3.0, step=0.1),
        'atr_tp_mult': trial.suggest_float('atr_tp_mult', 1.5, 5.0, step=0.1),
        'trailing_stop_atr': trial.suggest_float('trailing_stop_atr', 0.8, 2.5, step=0.1),
        'trail_activate_atr': trial.suggest_float('trail_activate_atr', 0.5, 2.0, step=0.1),
        'momentum_exit_thresh': trial.suggest_float('momentum_exit_thresh', 0.1, 0.5, step=0.05),
        'time_decay_bars': trial.suggest_int('time_decay_bars', 12, 48),
        'time_decay_min_atr': trial.suggest_float('time_decay_min_atr', 0.2, 1.0, step=0.1),

        # ── Portfolio risk ──
        'max_positions': trial.suggest_int('max_positions', 3, 8),
        'max_currency_exposure': trial.suggest_float('max_currency_exposure', 2.0, 4.0, step=0.5),
        'dd_yellow': trial.suggest_float('dd_yellow', 2.0, 5.0, step=0.5),
        'dd_red': trial.suggest_float('dd_red', 5.0, 12.0, step=1.0),
    }


def get_adaptive_params(base_params, lt_regime, vol_regime_ratio, symbol):
    """Adjust parameters based on current market state."""
    import numpy as np
    params = base_params.copy()
    vol_adj = np.clip(vol_regime_ratio, 0.6, 1.8)

    params['atr_sl_mult'] = base_params['atr_sl_mult'] * vol_adj
    params['atr_tp_mult'] = base_params['atr_tp_mult'] * (1 + (vol_adj - 1) * 0.5)
    params['trailing_stop_atr'] = base_params['trailing_stop_atr'] * vol_adj

    if lt_regime in ('LT_TREND_UP', 'LT_TREND_DOWN'):
        params['atr_tp_mult'] *= 1.2
        params['trail_activate_atr'] = params.get('trail_activate_atr', 1.0) * 1.1
        params['time_decay_bars'] = int(params.get('time_decay_bars', 24) * 1.3)
    elif lt_regime == 'LT_TRANSITION':
        params['atr_tp_mult'] *= 0.8
        params['trail_activate_atr'] = params.get('trail_activate_atr', 1.0) * 0.8
        params['time_decay_bars'] = int(params.get('time_decay_bars', 24) * 0.7)

    if symbol == 'XAUUSD':
        params['atr_tp_mult'] *= 1.15
        params['trail_activate_atr'] = params.get('trail_activate_atr', 1.0) * 1.1
    elif 'JPY' in symbol:
        params['atr_sl_mult'] *= 1.05

    return params

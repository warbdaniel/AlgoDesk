"""
Walk-Forward Validation Framework
===================================
Rolling walk-forward: train on N bars, test on next M bars, roll forward.
The only honest way to measure system performance.
"""

import numpy as np


def walk_forward_validation(df, symbol, param_optimizer_fn, backtest_fn, config,
                            train_bars=50000, test_bars=10000, step_bars=10000):
    """
    Rolling walk-forward validation.
    param_optimizer_fn: function(train_data, symbol, config) -> best_params
    backtest_fn: function(test_data, symbol, params, config) -> result_dict
    """
    results = []
    total_bars = len(df)
    start = 0

    while start + train_bars + test_bars <= total_bars:
        train_data = df.iloc[start : start + train_bars]
        best_params = param_optimizer_fn(train_data, symbol, config)

        test_data = df.iloc[start + train_bars : start + train_bars + test_bars]
        test_result = backtest_fn(test_data, symbol, best_params, config)

        results.append({
            'train_start': train_data['timestamp'].iloc[0] if 'timestamp' in train_data.columns else start,
            'train_end': train_data['timestamp'].iloc[-1] if 'timestamp' in train_data.columns else start + train_bars,
            'test_start': test_data['timestamp'].iloc[0] if 'timestamp' in test_data.columns else start + train_bars,
            'test_end': test_data['timestamp'].iloc[-1] if 'timestamp' in test_data.columns else start + train_bars + test_bars,
            'params': best_params,
            'trades': test_result.get('n_trades', 0),
            'win_rate': test_result.get('win_rate', 0),
            'pf': test_result.get('profit_factor', 0),
            'sharpe': test_result.get('sharpe', 0),
            'pnl_pips': test_result.get('total_pnl', 0),
            'max_dd': test_result.get('max_drawdown', 0),
        })

        start += step_bars

    if not results:
        return {'windows': [], 'oos_total_trades': 0, 'oos_total_pnl': 0,
                'oos_avg_wr': 0, 'param_stability': {}}

    oos_trades = sum(r['trades'] for r in results)
    oos_pnl = sum(r['pnl_pips'] for r in results)
    oos_wr = np.mean([r['win_rate'] for r in results if r['trades'] > 0]) if any(r['trades'] > 0 for r in results) else 0

    param_stability = {}
    if results and results[0]['params']:
        param_keys = results[0]['params'].keys()
        for key in param_keys:
            values = [r['params'][key] for r in results if key in r['params']]
            if values:
                mean_val = np.mean(values)
                param_stability[key] = {
                    'mean': mean_val,
                    'std': np.std(values),
                    'cv': np.std(values) / mean_val if mean_val != 0 else 999,
                }

    return {
        'windows': results,
        'oos_total_trades': oos_trades,
        'oos_total_pnl': oos_pnl,
        'oos_avg_wr': oos_wr,
        'param_stability': param_stability,
    }


def detect_overfitting(in_sample_sharpe, oos_sharpe, n_params):
    """
    Check if the system is overfit.
    OOS Sharpe should be > 50% of IS Sharpe.
    """
    if oos_sharpe <= 0:
        return True, "OOS Sharpe is negative — system is overfit"

    ratio = oos_sharpe / in_sample_sharpe if in_sample_sharpe > 0 else 0
    if ratio < 0.5:
        return True, f"OOS/IS Sharpe ratio {ratio:.2f} < 0.50 — likely overfit"

    if n_params > 15 and ratio < 0.6:
        return True, f"High param count ({n_params}) with low OOS ratio ({ratio:.2f})"

    return False, f"OOS/IS ratio {ratio:.2f} — acceptable"

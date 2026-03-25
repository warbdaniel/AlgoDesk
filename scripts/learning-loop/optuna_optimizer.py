#!/usr/bin/env python3
"""
Autonomous ML Optimizer for ADAM Trading Strategy
Uses Optuna for hyperparameter optimization, replacing Claude Code dependency.
Runs as a self-contained feedback loop: backtest -> analyze -> optimize -> update -> repeat
"""

import sys
import os
sys.path.insert(0, '/opt/trading-desk')
sys.path.insert(0, '/opt/trading-desk/orchestrator')
sys.path.insert(0, '/opt/trading-desk/scripts/learning-loop')

import json
import yaml
import shutil
import logging
import time
import numpy as np
import optuna
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy

# Silence Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [OptunaOptimizer] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/opt/trading-desk/logs/learning-loop/optuna_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("/opt/trading-desk")
CONFIGS_DIR = BASE_DIR / "configs"
RESULTS_DIR = BASE_DIR / "data" / "backtest_results"
ITER_DIR = BASE_DIR / "data" / "learning-loop" / "iterations"
LOGS_DIR = BASE_DIR / "logs" / "learning-loop"
PARQUET_DIR = BASE_DIR / "data" / "historical" / "candles_5m"
OPTUNA_DB = f"sqlite:///{BASE_DIR / 'data' / 'learning-loop' / 'optuna_study.db'}"

# Success criteria from LEARNING_LOOP_PROMPT.md
TARGET_SHARPE = 1.0
TARGET_WIN_RATE = 0.55
TARGET_MAX_DD_PCT = 0.15


class AdamParameterSpace:
    """Defines the parameter search space for ADAM strategy."""
    
    @staticmethod
    def suggest_params(trial):
        """Suggest parameters for an Optuna trial."""
        params = {
            # Regime classification
            'adx_min_strong': trial.suggest_float('adx_min_strong', 28.0, 40.0, step=1.0),  # v2: raised floor
            'di_spread_min_strong': trial.suggest_float('di_spread_min_strong', 7.0, 15.0, step=0.5),  # v2: raised
            'adx_min_mild': trial.suggest_float('adx_min_mild', 12.0, 22.0, step=1.0),
            'di_spread_min_mild': trial.suggest_float('di_spread_min_mild', 1.5, 5.0, step=0.5),
            'adx_max_choppy': trial.suggest_float('adx_max_choppy', 10.0, 20.0, step=1.0),
            
            # Entry rules
            'min_confirming': trial.suggest_int('min_confirming', 2, 4),
            'macd_confirm': trial.suggest_categorical('macd_confirm', [True, False]),
            
            # RSI filters
            'rsi_long_low': trial.suggest_float('rsi_long_low', 25.0, 45.0, step=1.0),
            'rsi_long_high': trial.suggest_float('rsi_long_high', 65.0, 80.0, step=1.0),
            'rsi_short_low': trial.suggest_float('rsi_short_low', 20.0, 40.0, step=1.0),
            'rsi_short_high': trial.suggest_float('rsi_short_high', 55.0, 75.0, step=1.0),
            
            # Bollinger Band filters
            'bb_long_max': trial.suggest_float('bb_long_max', 0.7, 1.3, step=0.05),
            'bb_short_min': trial.suggest_float('bb_short_min', -0.3, 0.3, step=0.05),
            
            # Exit rules
            'atr_sl_mult': trial.suggest_float('atr_sl_mult', 1.0, 3.0, step=0.1),
            'atr_tp_mult': trial.suggest_float('atr_tp_mult', 1.5, 4.0, step=0.1),
            'trailing_stop_atr': trial.suggest_float('trailing_stop_atr', 1.0, 3.0, step=0.1),

            # Volume proxy filters (v2)
            'min_body_ratio': trial.suggest_float('min_body_ratio', 0.2, 0.6, step=0.05),
            'min_directional_streak': trial.suggest_int('min_directional_streak', 1, 4),
            'min_relative_candle_size': trial.suggest_float('min_relative_candle_size', 0.5, 1.2, step=0.1),
            'require_cvd_alignment': trial.suggest_categorical('require_cvd_alignment', [True, False]),
            
            # Regime filter for trading
            'allowed_regimes': trial.suggest_categorical('allowed_regimes', [
                'STRONG_TREND_ONLY',
                'STRONG_AND_MILD',
                'ALL_EXCEPT_CHOPPY'
            ]),
        }
        return params

    @staticmethod
    def params_to_backtest_format(params):
        """Convert Optuna params to backtester-compatible format."""
        regime_map = {
            'STRONG_TREND_ONLY': ['STRONG_TREND'],
            'STRONG_AND_MILD': ['STRONG_TREND', 'MILD_TREND'],
            'ALL_EXCEPT_CHOPPY': ['STRONG_TREND', 'MILD_TREND', 'RANGING'],
        }
        
        return {
            'min_confirming': params['min_confirming'],
            'macd_confirm': params['macd_confirm'],
            'rsi_long_range': [params['rsi_long_low'], params['rsi_long_high']],
            'rsi_short_range': [params['rsi_short_low'], params['rsi_short_high']],
            'bb_long_max': params['bb_long_max'],
            'bb_short_min': params['bb_short_min'],
            'atr_sl_mult': params['atr_sl_mult'],
            'atr_tp_mult': params['atr_tp_mult'],
            'trailing_stop_atr': params.get('trailing_stop_atr', 2.0),
            # Volume proxy filters
            'min_body_ratio': params.get('min_body_ratio', 0.4),
            'min_directional_streak': params.get('min_directional_streak', 2),
            'min_relative_candle_size': params.get('min_relative_candle_size', 0.8),
            'require_cvd_alignment': params.get('require_cvd_alignment', True),
            'allowed_regimes': regime_map.get(params['allowed_regimes'], ['STRONG_TREND']),
            'adx_min_strong': params['adx_min_strong'],
            'di_spread_min_strong': params['di_spread_min_strong'],
            'adx_min_mild': params['adx_min_mild'],
            'di_spread_min_mild': params['di_spread_min_mild'],
            'adx_max_choppy': params['adx_max_choppy'],
        }

    @staticmethod
    def params_to_yaml(params, existing_rules):
        """Convert optimized params back to YAML config format."""
        rules = deepcopy(existing_rules)
        
        # Update regime classification
        if 'regime' not in rules:
            rules['regime'] = {}
        if 'strong_trend' not in rules['regime']:
            rules['regime']['strong_trend'] = {}
        rules['regime']['strong_trend']['adx_min'] = params['adx_min_strong']
        rules['regime']['strong_trend']['di_spread_min'] = params['di_spread_min_strong']
        
        if 'mild_trend' not in rules['regime']:
            rules['regime']['mild_trend'] = {}
        rules['regime']['mild_trend']['adx_min'] = params['adx_min_mild']
        rules['regime']['mild_trend']['di_spread_min'] = params['di_spread_min_mild']
        
        if 'choppy' not in rules['regime']:
            rules['regime']['choppy'] = {}
        rules['regime']['choppy']['adx_max'] = params['adx_max_choppy']
        
        # Update entry rules
        if 'entry' not in rules:
            rules['entry'] = {}
        
        regime_map = {
            'STRONG_TREND_ONLY': ['STRONG_TREND'],
            'STRONG_AND_MILD': ['STRONG_TREND', 'MILD_TREND'],
            'ALL_EXCEPT_CHOPPY': ['STRONG_TREND', 'MILD_TREND', 'RANGING'],
        }
        rules['entry']['allowed_regimes'] = regime_map.get(
            params['allowed_regimes'], ['STRONG_TREND'])
        rules['entry']['min_confirming'] = params['min_confirming']
        rules['entry']['macd_confirm'] = params['macd_confirm']
        rules['entry']['rsi_long_range'] = [params['rsi_long_low'], params['rsi_long_high']]
        rules['entry']['rsi_short_range'] = [params['rsi_short_low'], params['rsi_short_high']]
        rules['entry']['bb_long_max'] = params['bb_long_max']
        rules['entry']['bb_short_min'] = params['bb_short_min']
        
        # Update exit rules
        if 'exit' not in rules:
            rules['exit'] = {}
        rules['exit']['atr_sl_mult'] = params['atr_sl_mult']
        rules['exit']['atr_tp_mult'] = params['atr_tp_mult']
        rules['exit']['trailing_stop_atr'] = params.get('trailing_stop_atr', 2.0)
        
        # Update metadata
        rules['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        rules['version'] = str(float(rules.get('version', '2.0')) + 0.1)
        
        return rules


class FastBacktester:
    """Lightweight backtester for Optuna optimization trials.
    Loads data once, runs many parameter combinations quickly."""
    
    def __init__(self):
        self.data_cache = {}
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all parquet data into memory."""
        import pandas as pd
        parquet_files = list(PARQUET_DIR.glob("*.parquet"))
        logger.info(f"Loading {len(parquet_files)} symbol data files...")
        
        for pf in parquet_files:
            symbol = pf.stem
            try:
                df = pd.read_parquet(pf)
                if len(df) >= 100:
                    self.data_cache[symbol] = df
                    logger.info(f"  Loaded {symbol}: {len(df)} bars")
            except Exception as e:
                logger.warning(f"  Failed to load {symbol}: {e}")
        
        logger.info(f"Loaded {len(self.data_cache)} symbols into memory")
    
    def run_backtest(self, params):
        """Run backtest with given parameters across all symbols."""
        from regime_backtester import (
            compute_indicators, classify_regime, 
            generate_adam_signals, compute_metrics, analyze_by_regime
        )
        from orch_config import PIP_SIZES, DEFAULT_PIP_SIZE
        
        bt_params = AdamParameterSpace.params_to_backtest_format(params)
        allowed_regimes = bt_params.get('allowed_regimes', ['STRONG_TREND'])
        
        all_trades = []
        per_symbol = {}
        
        for symbol, df in self.data_cache.items():
            try:
                indicators = compute_indicators(df)
                signals = generate_adam_signals(df, indicators, bt_params)
                
                # Classify regimes
                regimes = []
                for i in range(len(indicators)):
                    row = indicators.iloc[i]
                    adx = row.get('adx', 0)
                    plus_di = row.get('plus_di', 0)
                    minus_di = row.get('minus_di', 0)
                    di_spread = abs(plus_di - minus_di)
                    
                    if adx >= bt_params.get('adx_min_strong', 23) and \
                       di_spread >= bt_params.get('di_spread_min_strong', 5):
                        regimes.append('STRONG_TREND')
                    elif adx >= bt_params.get('adx_min_mild', 17) and \
                         di_spread >= bt_params.get('di_spread_min_mild', 3):
                        regimes.append('MILD_TREND')
                    elif adx <= bt_params.get('adx_max_choppy', 15):
                        regimes.append('CHOPPY')
                    else:
                        regimes.append('RANGING')
                
                # Get pip size
                pip_size = PIP_SIZES.get(symbol, DEFAULT_PIP_SIZE)
                
                # Simulate trades with regime filtering
                sl_mult = bt_params.get('atr_sl_mult', 1.5)
                tp_mult = bt_params.get('atr_tp_mult', 2.0)
                
                position = None
                trades = []
                
                for i in range(len(df)):
                    if i >= len(signals) or i >= len(regimes):
                        continue
                    
                    close = df['close'].iloc[i]
                    atr = indicators['atr'].iloc[i] if 'atr' in indicators.columns else 0
                    
                    if atr == 0 or np.isnan(atr):
                        continue
                    
                    regime = regimes[i]
                    signal = signals.iloc[i] if hasattr(signals, 'iloc') else signals[i]
                    
            # Check if we have an open position
            if position is not None:
                high_i = df['high'].iloc[i]
                low_i = df['low'].iloc[i]
                d = position['direction']
                
                # Update watermarks
                if d == 1:
                    position['highest_since_entry'] = max(position.get('highest_since_entry', high_i), high_i)
                else:
                    position['lowest_since_entry'] = min(position.get('lowest_since_entry', low_i), low_i)
                
                unrealized_atr = (close - position['entry']) * d / atr if atr > 0 else 0
                exit_reason = None
                exit_price = None
                
                # L1: Hard SL/TP
                if d == 1:
                    if low_i <= position['sl']:
                        exit_reason, exit_price = 'sl_hit', position['sl']
                    elif high_i >= position['tp']:
                        exit_reason, exit_price = 'tp_hit', position['tp']
                else:
                    if high_i >= position['sl']:
                        exit_reason, exit_price = 'sl_hit', position['sl']
                    elif low_i <= position['tp']:
                        exit_reason, exit_price = 'tp_hit', position['tp']
                
                # L2: ATR Trailing Stop (ratchet) - THE BUG FIX
                if exit_reason is None:
                    trail_atr = params.get('trailing_stop_atr', 2.0)
                    trail_activate = params.get('trail_activate_atr', 1.0)
                    if unrealized_atr >= trail_activate:
                        position['trail_active'] = True
                    if position.get('trail_active', False):
                        if d == 1:
                            trail_sl = position['highest_since_entry'] - atr * trail_atr
                            position['sl'] = max(position['sl'], trail_sl)
                        else:
                            trail_sl = position['lowest_since_entry'] + atr * trail_atr
                            position['sl'] = min(position['sl'], trail_sl)
                
                if exit_reason is not None:
                    pnl_pips = (exit_price - position['entry']) * d / pip_size
                    trades.append({
                        'symbol': symbol,
                        'direction': d,
                        'entry': position['entry'],
                        'exit': exit_price,
                        'pnl_pips': pnl_pips,
                        'regime': position['regime'],
                        'duration_bars': i - position['bar'],
                        'exit_reason': exit_reason,
                    })
                    position = None


            # Open new position if no position and valid signal
            if position is None and signal != 0 and regime in allowed_regimes:
                if signal == 1:   # Long
                    position = {
                        'entry': close,
                        'sl': close - atr * sl_mult,
                        'tp': close + atr * tp_mult,
                        'direction': 1,
                        'regime': regime,
                        'bar': i,
                        'highest_since_entry': close,
                        'lowest_since_entry': close,
                        'trail_active': False,
                    }
                elif signal == -1:   # Short
                    position = {
                        'entry': close,
                        'sl': close + atr * sl_mult,
                        'tp': close - atr * tp_mult,
                        'direction': -1,
                        'regime': regime,
                        'bar': i,
                        'highest_since_entry': close,
                        'lowest_since_entry': close,
                        'trail_active': False,
                    }


                if trades:
                    sym_metrics = compute_metrics(trades)
                    per_symbol[symbol] = sym_metrics
                    all_trades.extend(trades)
                    
            except Exception as e:
                logger.debug(f"Backtest failed for {symbol}: {e}")
                continue
        
        if not all_trades:
            return {'sharpe': -10, 'win_rate': 0, 'profit_factor': 0,
                    'total_pnl': 0, 'max_dd': 999999, 'total_trades': 0,
                    'per_symbol': {}, 'by_regime': {}}
        
        # Calculate aggregate metrics
        aggregate = compute_metrics(all_trades)
        by_regime = analyze_by_regime(all_trades)
        
        aggregate['per_symbol'] = per_symbol
        aggregate['by_regime'] = by_regime
        aggregate['total_trades'] = len(all_trades)
        
        return aggregate


class AutonomousOptimizer:
    """Main autonomous optimization loop."""
    
    def __init__(self, max_iterations=50, trials_per_iteration=30,
                 cooldown_seconds=60):
        self.max_iterations = max_iterations
        self.trials_per_iteration = trials_per_iteration
        self.cooldown = cooldown_seconds
        self.state_file = ITER_DIR / "optimizer_state.json"
        self.state = self._load_state()
        self.backtester = None  # Lazy init
        
        # Ensure directories
        ITER_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except:
                pass
        return {
            "current_iteration": 0,
            "best_sharpe": None,
            "best_iteration": None,
            "best_params": None,
            "history": [],
            "status": "idle",
            "baseline_sharpe": None,
        }
    
    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def _backup_config(self):
        """Backup current YAML config before modifying."""
        src = CONFIGS_DIR / "adam_trading_rules.yaml"
        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        dst = CONFIGS_DIR / f"adam_trading_rules.yaml.bak.{ts}"
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"Config backed up to {dst.name}")
            return dst
        return None
    
    def _load_yaml_config(self):
        """Load current ADAM trading rules."""
        yaml_path = CONFIGS_DIR / "adam_trading_rules.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _save_yaml_config(self, rules):
        """Save updated ADAM trading rules."""
        yaml_path = CONFIGS_DIR / "adam_trading_rules.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False, sort_keys=False)
        logger.info("YAML config updated")
    
    def _get_baseline_results(self):
        """Load the most recent backtest results as baseline."""
        latest = RESULTS_DIR / "latest_backtest.json"
        if latest.exists():
            with open(latest) as f:
                data = json.load(f)
            overall = data.get('overall_metrics', data.get('aggregate_metrics', {}))
            return {
                'sharpe': overall.get('sharpe_ratio', overall.get('sharpe', -10)),
                'win_rate': overall.get('win_rate', 0),
                'profit_factor': overall.get('profit_factor', 0),
                'total_pnl': overall.get('total_pnl', 0),
                'max_dd': overall.get('max_dd', 999999),
            }
        return None
    
    def _run_optuna_optimization(self, iteration):
        """Run Optuna optimization to find better parameters."""
        logger.info(f"Starting Optuna optimization: {self.trials_per_iteration} trials")
        
        if self.backtester is None:
            self.backtester = FastBacktester()
        
        study = optuna.create_study(
            study_name=f"adam_iter_{iteration}",
            direction="maximize",
            storage=OPTUNA_DB,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=iteration * 42),
        )
        
        def objective(trial):
            params = AdamParameterSpace.suggest_params(trial)
            results = self.backtester.run_backtest(params)
            
            sharpe = results.get('sharpe', -10)
            win_rate = results.get('win_rate', 0)
            pf = results.get('profit_factor', 0)
            total_trades = results.get('total_trades', 0)
            max_dd = results.get('max_dd', 999999)
            
            # === HARD RULE: Minimum 100 trades for statistical significance ===
            # A backtest with fewer than 100 trades cannot confirm if a variable
            # change is significant. This is enforced at every level of the system.
            if total_trades < 100:
                return -10.0
            
            # Multi-objective score: primarily Sharpe, with bonuses
            score = sharpe
            
            # Bonus for meeting win rate target
            if win_rate >= TARGET_WIN_RATE:
                score += 0.2
            
            # Bonus for positive profit factor
            if pf > 1.0:
                score += 0.1 * min(pf - 1.0, 1.0)
            
            # Penalty for extreme drawdown
            if max_dd > 100000:
                score -= 0.5
            
            # Store full results as user attributes
            trial.set_user_attr('win_rate', win_rate)
            trial.set_user_attr('profit_factor', pf)
            trial.set_user_attr('total_pnl', results.get('total_pnl', 0))
            trial.set_user_attr('total_trades', total_trades)
            trial.set_user_attr('max_dd', max_dd)
            
            return score
        
        study.optimize(objective, n_trials=self.trials_per_iteration,
                       show_progress_bar=False)
        
        best = study.best_trial
        logger.info(f"Optuna best score: {best.value:.4f}")
        logger.info(f"  Win Rate: {best.user_attrs.get('win_rate', 'N/A')}")
        logger.info(f"  Profit Factor: {best.user_attrs.get('profit_factor', 'N/A')}")
        logger.info(f"  Total Trades: {best.user_attrs.get('total_trades', 'N/A')}")
        logger.info(f"  Max DD: {best.user_attrs.get('max_dd', 'N/A')}")
        
        return best.params, best.value, best.user_attrs
    
    def _run_full_backtest(self):
        """Run the full regime backtester to get official results."""
        import subprocess
        logger.info("Running full regime backtester for validation...")
        try:
            result = subprocess.run(
                ['/opt/trading-desk/venv/bin/python3',
                 '/opt/trading-desk/scripts/learning-loop/regime_backtester.py'],
                capture_output=True, text=True, timeout=600,
                cwd=str(BASE_DIR),
                env={
                    **os.environ,
                    'PYTHONPATH': '/opt/trading-desk/orchestrator:/opt/trading-desk/scripts/learning-loop',
                    'VIRTUAL_ENV': '/opt/trading-desk/venv',
                    'PATH': '/opt/trading-desk/venv/bin:/usr/local/bin:/usr/bin:/bin',
                }
            )
            if result.returncode == 0:
                logger.info("Full backtest completed successfully")
                # Parse the output for key metrics
                for line in result.stdout.split('\n')[-20:]:
                    if 'OVERALL' in line or 'Sharpe' in line or 'Win rate' in line:
                        logger.info(f"  {line.strip()}")
                return True
            else:
                logger.error(f"Full backtest failed: {result.stderr[:500]}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Full backtest timed out after 600s")
            return False
        except Exception as e:
            logger.error(f"Full backtest error: {e}")
            return False

    
    def run_iteration(self):
        """Run a single optimization iteration."""
        self.state['current_iteration'] += 1
        iteration = self.state['current_iteration']
        self.state['status'] = 'running'
        self._save_state()
        
        iter_dir = ITER_DIR / f"iteration_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = datetime.now(timezone.utc)
        logger.info(f"\n{'='*70}")
        logger.info(f" AUTONOMOUS ML OPTIMIZER - ITERATION {iteration}")
        logger.info(f" Started: {start_time.isoformat()}")
        logger.info(f"{'='*70}")
        
        iteration_result = {
            'iteration': iteration,
            'start_time': start_time.isoformat(),
            'steps': {},
        }
        
        try:
            # Step 1: Get baseline metrics
            logger.info("\n--- Step 1: Baseline Assessment ---")
            baseline = self._get_baseline_results()
            if baseline:
                logger.info(f"  Baseline Sharpe: {baseline['sharpe']}")
                logger.info(f"  Baseline WR: {baseline['win_rate']}")
                logger.info(f"  Baseline PF: {baseline['profit_factor']}")
                if self.state['baseline_sharpe'] is None:
                    self.state['baseline_sharpe'] = baseline['sharpe']
            else:
                logger.info("  No baseline results found, starting fresh")
                baseline = {'sharpe': -10, 'win_rate': 0, 'profit_factor': 0}
            
            iteration_result['steps']['baseline'] = baseline
            
            # Step 2: Run Optuna optimization
            logger.info("\n--- Step 2: Optuna Parameter Optimization ---")
            best_params, best_score, best_attrs = self._run_optuna_optimization(iteration)
            
            iteration_result['steps']['optuna'] = {
                'best_score': best_score,
                'best_params': best_params,
                'best_attrs': best_attrs,
            }
            
            # Step 3: Compare with baseline
            logger.info("\n--- Step 3: Comparing with Baseline ---")
            current_sharpe = baseline.get('sharpe', -10)
            improvement = best_score - current_sharpe
            logger.info(f"  Current Sharpe: {current_sharpe:.4f}")
            logger.info(f"  Optuna Best Score: {best_score:.4f}")
            logger.info(f"  Improvement: {improvement:+.4f}")
            
            # Step 4: Apply if improvement found
            if best_score > current_sharpe or current_sharpe < -5:
                logger.info("\n--- Step 4: Applying Optimized Parameters ---")
                
                # Backup current config
                backup_path = self._backup_config()
                
                # Load and update YAML
                rules = self._load_yaml_config()
                new_rules = AdamParameterSpace.params_to_yaml(best_params, rules)
                self._save_yaml_config(new_rules)
                
                # Save optimized params
                with open(iter_dir / "optimized_params.json", 'w') as f:
                    json.dump(best_params, f, indent=2, default=str)
                
                # Step 5: Run full validation backtest
                logger.info("\n--- Step 5: Full Validation Backtest ---")
                bt_success = self._run_full_backtest()
                
                if bt_success:
                    # Load new results
                    new_baseline = self._get_baseline_results()
                    if new_baseline:
                        new_sharpe = new_baseline.get('sharpe', -10)
                        logger.info(f"\n  Validation Sharpe: {new_sharpe:.4f}")
                        
                        # Check if validation confirms improvement
                        if new_sharpe > current_sharpe:
                            logger.info("  IMPROVEMENT CONFIRMED - Keeping changes")
                            iteration_result['steps']['validation'] = {
                                'status': 'improved',
                                'old_sharpe': current_sharpe,
                                'new_sharpe': new_sharpe,
                            }
                        else:
                            logger.warning("  Validation did NOT confirm improvement")
                            logger.warning("  Rolling back to previous config")
                            if backup_path and backup_path.exists():
                                shutil.copy2(backup_path, 
                                           CONFIGS_DIR / "adam_trading_rules.yaml")
                                self._run_full_backtest()  # Re-run with old config
                            iteration_result['steps']['validation'] = {
                                'status': 'rolled_back',
                                'old_sharpe': current_sharpe,
                                'new_sharpe': new_sharpe,
                            }
                    else:
                        logger.warning("  Could not load validation results")
                        iteration_result['steps']['validation'] = {'status': 'no_results'}
                else:
                    logger.error("  Validation backtest failed")
                    # Rollback
                    if backup_path and backup_path.exists():
                        shutil.copy2(backup_path, CONFIGS_DIR / "adam_trading_rules.yaml")
                    iteration_result['steps']['validation'] = {'status': 'backtest_failed'}
            else:
                logger.info("\n--- Step 4: No Improvement Found ---")
                logger.info("  Keeping current parameters (Optuna didn't find better)")
                iteration_result['steps']['decision'] = 'no_improvement'
            
            # Record iteration
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            final_baseline = self._get_baseline_results()
            final_sharpe = final_baseline.get('sharpe', -10) if final_baseline else -10
            
            iteration_result.update({
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'final_sharpe': final_sharpe,
                'status': 'complete',
            })
            
            # Save iteration result
            with open(iter_dir / "iteration_result.json", 'w') as f:
                json.dump(iteration_result, f, indent=2, default=str)
            
            # Update state
            self.state['last_sharpe'] = final_sharpe
            if self.state['best_sharpe'] is None or final_sharpe > self.state['best_sharpe']:
                self.state['best_sharpe'] = final_sharpe
                self.state['best_iteration'] = iteration
                self.state['best_params'] = best_params
            
            self.state['history'].append({
                'iteration': iteration,
                'sharpe': final_sharpe,
                'optuna_score': best_score,
                'duration': duration,
                'timestamp': end_time.isoformat(),
            })
            self.state['status'] = 'idle'
            self._save_state()
            
            logger.info(f"\nIteration {iteration} complete in {duration:.0f}s")
            logger.info(f"Sharpe: {final_sharpe}, Best: {self.state['best_sharpe']} "
                       f"(iter {self.state['best_iteration']})")
            
            return True
            
        except Exception as e:
            logger.error(f"Iteration {iteration} failed: {e}")
            import traceback
            traceback.print_exc()
            iteration_result['status'] = 'error'
            iteration_result['error'] = str(e)
            with open(iter_dir / "iteration_result.json", 'w') as f:
                json.dump(iteration_result, f, indent=2, default=str)
            self.state['status'] = 'error'
            self._save_state()
            return False
    
    def _check_convergence(self):
        """Check if optimization has converged."""
        history = self.state.get('history', [])
        if len(history) < 5:
            return False
        
        recent = history[-5:]
        sharpes = [h.get('sharpe', 0) for h in recent if h.get('sharpe') is not None]
        if len(sharpes) < 5:
            return False
        
        # Check if improvement is less than 1% over last 5 iterations
        improvement = max(sharpes) - min(sharpes)
        if improvement < 0.01 and all(s > 0 for s in sharpes):
            logger.info("Convergence detected: minimal improvement over last 5 iterations")
            return True
        
        # Check if we've met targets
        best = max(sharpes)
        if best >= TARGET_SHARPE:
            logger.info(f"Target Sharpe ratio achieved: {best:.4f} >= {TARGET_SHARPE}")
            return True
        
        return False
    
    def run_continuous(self):
        """Run the optimization loop continuously."""
        logger.info("="*70)
        logger.info(" AUTONOMOUS ML OPTIMIZATION LOOP")
        logger.info(f" Max iterations: {self.max_iterations}")
        logger.info(f" Trials per iteration: {self.trials_per_iteration}")
        logger.info(f" Cooldown: {self.cooldown}s between iterations")
        logger.info("="*70)
        
        start_iter = self.state['current_iteration'] + 1
        
        for i in range(start_iter, start_iter + self.max_iterations):
            try:
                if self._check_convergence():
                    logger.info("Optimization has converged. Stopping.")
                    break
                
                success = self.run_iteration()
                
                if not success:
                    logger.warning(f"Iteration failed. Waiting {self.cooldown * 3}s...")
                    time.sleep(self.cooldown * 3)
                else:
                    logger.info(f"Cooling down {self.cooldown}s before next iteration...")
                    time.sleep(self.cooldown)
                    
            except KeyboardInterrupt:
                logger.info("Interrupted by user. Stopping gracefully.")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(self.cooldown * 5)
        
        logger.info("\nAutonomous optimization complete.")
        logger.info(f"Best Sharpe: {self.state.get('best_sharpe')} "
                   f"(iteration {self.state.get('best_iteration')})")
        self.state['status'] = 'finished'
        self._save_state()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Autonomous ML Optimizer for ADAM')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Max iterations (default: 50)')
    parser.add_argument('--trials', type=int, default=30,
                       help='Optuna trials per iteration (default: 30)')
    parser.add_argument('--cooldown', type=int, default=60,
                       help='Seconds between iterations (default: 60)')
    parser.add_argument('--single', action='store_true',
                       help='Run single iteration only')
    parser.add_argument('--status', action='store_true',
                       help='Show current status and exit')
    args = parser.parse_args()
    
    optimizer = AutonomousOptimizer(
        max_iterations=args.iterations,
        trials_per_iteration=args.trials,
        cooldown_seconds=args.cooldown,
    )
    
    if args.status:
        print(json.dumps(optimizer.state, indent=2, default=str))
    elif args.single:
        optimizer.run_iteration()
    else:
        optimizer.run_continuous()

# Optimizer V2 Changes — 2026-03-27

## Summary
Stopped PID 1132395 (old optimizer), applied modifications, restarted as PID recorded in logs.

## Changes Made

### 1. Objective Function: Monthly Account % Return (not raw pip EV)
- Added `backtest_strategy_equity()` function to `orchestrator/backtester.py`
- Simulates equity curve with 2% risk per trade, half-Kelly position sizing
- Each trade's P&L converted to R-multiples × risk fraction = account % change
- Monthly return % calculated from equity curve over backtest period
- Added `_half_kelly_factor()` helper function

### 2. Hard Drawdown Constraint: 10% (was 20%)
- `MAX_DRAWDOWN_LIMIT = 0.10` (was 0.20) in `continuous_optimizer_48h.py`
- DD constraint uses worst-case across all active regimes
- `orch_config.py` `MAX_ACCOUNT_DRAWDOWN = 0.40` left as-is (live kill switch)

### 3. Multi-Regime Optimization (was STRONG_TREND only)
- Added `adx_mild_threshold` trial parameter (12-25)
- Added `mild_trend_entry_mult` and `ranging_entry_mult` multipliers
- Added `trade_mild_trend` and `trade_ranging` boolean gates
- Backtests run on strong_df, mild_df, and ranging_df separately
- Each regime gets adjusted SL/TP via entry multipliers
- Expanded from 2 symbols (XAUUSD, USDJPY) to 9 (top pairs by data)

### 4. Trades/Month: Regime-Prevalence-Weighted
- Added `REGIME_PREVALENCE` dict from STAGE_2_LOG.md Section 11
- trades/month = Σ (regime_trades/regime_bars × BARS_PER_MONTH × regime_prevalence)
- XAUUSD: 32.1% strong, ~30% mild, ~38% ranging
- USDJPY: 28.2% strong, ~30% mild, ~42% ranging
- No longer inflates by extrapolating from strong-trend-only subset

### 5. Composite Score: Monthly % Gain Priority
- 60% weight: Monthly account % return (was 50% raw pip EV)
- 20% weight: Trade frequency sqrt(trades/month) (was 15%)
- 15% weight: DD bonus (staying under 10%) (was 10%)
- 5% weight: Sharpe ratio (was 25%)
- Prioritizes throughput: more small-edge trades > fewer big-edge trades

## Files Modified
- `continuous_optimizer_48h.py` — main optimizer (backed up as .bak.*)
- `orchestrator/backtester.py` — added equity-curve backtest (backed up)

## Files NOT Modified
- `orchestrator/orch_config.py` — live kill switch left at 40%
- `orchestrator/risk_manager.py` — unchanged (already has half-Kelly)

## Backups
- `continuous_optimizer_48h.py.bak.*` 
- `orchestrator/backtester.py.bak.*`

## How to Pick Up
1. Check optimizer running: `ps aux | grep continuous_optimizer_48h`
2. Check latest log: `tail -f logs/continuous_optimizer/optimizer_v3_*.log`
3. Results in: `continuous_optimizer_results/`
4. Champion: `continuous_optimizer_results/champion_settings.json`
5. To stop gracefully: `kill -SIGTERM <PID>` (has SIGTERM handler)

---

# Interim Results Review — 2026-03-27 09:30 UTC

Reviewed at 17% completion (8.15h / 48h, 972 cycles, 17,618 trials).

## Overall Champion: XAUUSD
- OOS Monthly EV: 632.3, Sharpe: 10.59, PF: 3.37, DD: 8.65%, Score: 709,324

## Pareto Selection (Primary 3 Assets)
1. XAUUSD — gold/haven (OOS EV 632, Sharpe 10.59)
2. EURJPY — carry/sentiment (OOS EV 146, Sharpe 8.50)
3. USDCAD — USD/oil (OOS EV 103, Sharpe 11.47)

## Decision
Proceeding to Stage 3 roadmap. Full analysis in STAGE_2_LOG.md Section 15.

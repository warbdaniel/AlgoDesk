#!/usr/bin/env python3
"""
Full Training Pipeline
======================

For each symbol in the feature store:
  1. Train LightGBM with Optuna HPO (lib/trainer.py)
  2. Build stacked ensemble (lib/ensemble.py)
  3. Backtest evaluation with walk-forward (lib/evaluator.py)
  4. Generate summary report to reports/training_report.md

Usage:
    cd scripts/alpha-engine
    python train_all.py [--symbols EURUSD GBPUSD USDJPY] [--trials 50]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure alpha-engine is on the path
ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

from candle_config import CANDLE_DATA_DIR, DEFAULT_CANDLE_SYMBOLS
from lib.trainer import WalkForwardTrainer, TrainerConfig, load_split_from_db
from lib.ensemble import EnsembleBuilder, EnsembleConfig
from lib.evaluator import (
    BacktestEvaluator,
    WalkForwardEvaluator,
    plot_equity_curve,
    plot_calibration_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_all")

REPORTS_DIR = ENGINE_DIR / "reports"
MODELS_DIR = ENGINE_DIR / "models"


def discover_symbols(db_path: str) -> list[str]:
    """Find symbols that have datasets in the DB."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM candle_datasets ORDER BY symbol"
    ).fetchall()
    conn.close()
    return [r["symbol"] for r in rows]


def run_pipeline(symbols: list[str], db_path: str, n_trials: int) -> list[dict]:
    """Run the full pipeline for each symbol."""
    results = []

    for symbol in symbols:
        logger.info("=" * 60)
        logger.info("Processing %s", symbol)
        logger.info("=" * 60)

        result = {"symbol": symbol}

        # ── 1. Train LightGBM with Optuna ────────────────────────
        try:
            trainer_cfg = TrainerConfig(
                n_trials=n_trials,
                model_dir=str(MODELS_DIR),
            )
            trainer = WalkForwardTrainer(config=trainer_cfg)
            train_result = trainer.train(db_path, symbol)
            result["trainer"] = train_result
            logger.info("%s trainer: %s", symbol, train_result.get("status"))
        except Exception as e:
            logger.error("Trainer failed for %s: %s", symbol, e, exc_info=True)
            result["trainer"] = {"status": "error", "error": str(e)}

        # ── 2. Build ensemble ────────────────────────────────────
        try:
            ens_cfg = EnsembleConfig(model_dir=str(MODELS_DIR))
            builder = EnsembleBuilder(config=ens_cfg)
            ens_result = builder.build(db_path, symbol)
            result["ensemble"] = ens_result
            logger.info("%s ensemble: %s", symbol, ens_result.get("status"))
        except Exception as e:
            logger.error("Ensemble failed for %s: %s", symbol, e, exc_info=True)
            result["ensemble"] = {"status": "error", "error": str(e)}

        # ── 3. Backtest (simple + walk-forward) ──────────────────
        try:
            # Simple backtest on test set
            X_test, y_test, feat_names, ts_test = load_split_from_db(
                db_path, symbol, "test",
            )
            if len(X_test) > 0 and result.get("trainer", {}).get("status") == "ok":
                import joblib
                import numpy as np
                model_path = result["trainer"]["model_path"]
                artifact = joblib.load(model_path)
                model = artifact["model"]
                np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                y_proba = model.predict_proba(X_test)[:, 1]

                # Load test labels for barrier info
                conn = sqlite3.connect(db_path, timeout=30)
                conn.row_factory = sqlite3.Row
                ds_row = conn.execute(
                    "SELECT id FROM candle_datasets WHERE symbol=? "
                    "ORDER BY created_at DESC LIMIT 1", (symbol,),
                ).fetchone()
                if ds_row:
                    lbl_rows = conn.execute(
                        "SELECT labels FROM candle_dataset_samples "
                        "WHERE dataset_id=? AND split_name='test' AND symbol=? "
                        "ORDER BY ts", (ds_row["id"], symbol),
                    ).fetchall()
                    test_labels = [json.loads(r["labels"]) for r in lbl_rows]
                else:
                    test_labels = []
                conn.close()

                if test_labels:
                    bt = BacktestEvaluator(prob_threshold=0.5)
                    metrics, equity = bt.evaluate(y_proba, test_labels, symbol)
                    result["backtest"] = metrics.to_dict()
                    result["equity_curve"] = equity

                    # Plot equity curve
                    eq_path = REPORTS_DIR / f"{symbol}_equity.png"
                    plot_equity_curve(equity, symbol, eq_path)
                    result["equity_plot"] = str(eq_path)

                    # Plot calibration curve
                    if "calibration" in result.get("trainer", {}):
                        cal_path = REPORTS_DIR / f"{symbol}_calibration.png"
                        plot_calibration_curve(
                            result["trainer"]["calibration"], symbol, cal_path,
                        )
                        result["calibration_plot"] = str(cal_path)

            # Walk-forward evaluation
            wf = WalkForwardEvaluator(retrain_months=2, test_months=1)
            wf_result = wf.run(db_path, symbol)
            result["walk_forward"] = {
                k: v for k, v in wf_result.items() if k != "equity_curve"
            }

            if wf_result.get("equity_curve"):
                wf_eq_path = REPORTS_DIR / f"{symbol}_walkforward_equity.png"
                plot_equity_curve(
                    wf_result["equity_curve"], symbol, wf_eq_path,
                    title=f"{symbol} — Walk-Forward Equity Curve",
                )
                result["walkforward_equity_plot"] = str(wf_eq_path)

        except Exception as e:
            logger.error("Backtest failed for %s: %s", symbol, e, exc_info=True)
            result["backtest"] = {"status": "error", "error": str(e)}

        results.append(result)

    return results


def generate_report(results: list[dict], output_path: Path) -> None:
    """Generate markdown summary report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# AlgoDesk Training Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Symbol | LightGBM AUC | LightGBM Acc | Ensemble AUC | Ensemble Acc | Backtest WR | PF | Sharpe | MaxDD |",
        "|--------|-------------|-------------|-------------|-------------|------------|-----|--------|-------|",
    ]

    for r in results:
        sym = r["symbol"]
        t = r.get("trainer", {})
        e = r.get("ensemble", {})
        b = r.get("backtest", {})

        t_auc = f"{t.get('test_auc', 0):.4f}" if t.get("status") == "ok" else "N/A"
        t_acc = f"{t.get('test_accuracy', 0):.4f}" if t.get("status") == "ok" else "N/A"
        e_auc = f"{e.get('test_auc', 0):.4f}" if e.get("status") == "ok" else "N/A"
        e_acc = f"{e.get('test_accuracy', 0):.4f}" if e.get("status") == "ok" else "N/A"
        b_wr = f"{b.get('win_rate', 0):.2%}" if isinstance(b, dict) and "win_rate" in b else "N/A"
        b_pf = f"{b.get('profit_factor', 0):.2f}" if isinstance(b, dict) and "profit_factor" in b else "N/A"
        b_sh = f"{b.get('sharpe_ratio', 0):.2f}" if isinstance(b, dict) and "sharpe_ratio" in b else "N/A"
        b_dd = f"{b.get('max_drawdown_pips', 0):.1f}" if isinstance(b, dict) and "max_drawdown_pips" in b else "N/A"

        lines.append(
            f"| {sym} | {t_auc} | {t_acc} | {e_auc} | {e_acc} | {b_wr} | {b_pf} | {b_sh} | {b_dd} |"
        )

    lines.extend(["", "---", ""])

    # Detailed per-symbol sections
    for r in results:
        sym = r["symbol"]
        lines.extend([f"## {sym}", ""])

        # Trainer details
        t = r.get("trainer", {})
        if t.get("status") == "ok":
            lines.extend([
                "### LightGBM (Optuna-tuned)",
                f"- **Test AUC:** {t['test_auc']:.4f}",
                f"- **Test Accuracy:** {t['test_accuracy']:.4f}",
                f"- **Test Log Loss:** {t['test_logloss']:.4f}",
                f"- **Best Val Loss:** {t['best_val_loss']:.5f}",
                f"- **Train/Val/Test:** {t['train_size']}/{t['val_size']}/{t['test_size']}",
                f"- **Model:** `{t['model_path']}`",
                "",
                "**Best Hyperparameters:**",
                "```json",
                json.dumps(t["best_params"], indent=2),
                "```",
                "",
                "**Classification Report:**",
                "```",
                t.get("classification_report", "N/A"),
                "```",
                "",
                "**Top 10 Features:**",
                "| Feature | Importance |",
                "|---------|-----------|",
            ])
            for fname, fimp in t.get("feature_importance", [])[:10]:
                lines.append(f"| {fname} | {fimp} |")
            lines.append("")

        # Ensemble details
        e = r.get("ensemble", {})
        if e.get("status") == "ok":
            lines.extend([
                "### Stacked Ensemble (LightGBM + XGBoost → LR)",
                f"- **Test AUC:** {e['test_auc']:.4f}",
                f"- **Test Accuracy:** {e['test_accuracy']:.4f}",
                f"- **Model:** `{e['model_path']}`",
                "",
                "**Classification Report:**",
                "```",
                e.get("classification_report", "N/A"),
                "```",
                "",
            ])

        # Backtest details
        b = r.get("backtest", {})
        if isinstance(b, dict) and "win_rate" in b:
            lines.extend([
                "### Backtest (Test Set)",
                f"- **Total Trades:** {b['total_trades']}",
                f"- **Win Rate:** {b['win_rate']:.2%}",
                f"- **Profit Factor:** {b['profit_factor']:.2f}",
                f"- **Sharpe Ratio:** {b['sharpe_ratio']:.2f}",
                f"- **Max Drawdown:** {b['max_drawdown_pips']:.1f} pips",
                f"- **Expected Value:** {b['expected_value_pips']:.2f} pips/trade",
                f"- **Total PnL:** {b['total_pnl_pips']:.1f} pips",
                "",
            ])

        # Walk-forward
        wf = r.get("walk_forward", {})
        if wf.get("status") == "ok":
            lines.extend([
                "### Walk-Forward (retrain 2mo, test 1mo)",
                f"- **Windows:** {wf['walk_forward_windows']}",
                f"- **Total Trades:** {wf['total_trades']}",
                f"- **Total PnL:** {wf['total_pnl_pips']} pips",
                f"- **Overall Win Rate:** {wf['overall_win_rate']:.2%}",
                "",
            ])

        # Plot references
        if r.get("equity_plot"):
            lines.append(f"![Equity Curve]({Path(r['equity_plot']).name})")
        if r.get("walkforward_equity_plot"):
            lines.append(f"![Walk-Forward Equity]({Path(r['walkforward_equity_plot']).name})")
        if r.get("calibration_plot"):
            lines.append(f"![Calibration]({Path(r['calibration_plot']).name})")
        lines.append("")

    output_path.write_text("\n".join(lines))
    logger.info("Report saved: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="AlgoDesk ML Training Pipeline")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to train (default: auto-discover from DB)",
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Optuna trials per symbol (default: 50)",
    )
    parser.add_argument(
        "--db", type=str,
        default=str(CANDLE_DATA_DIR / "candle_ml.db"),
        help="Path to candle_ml.db",
    )
    args = parser.parse_args()

    db_path = args.db
    logger.info("DB: %s", db_path)

    # Discover or use specified symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = discover_symbols(db_path)
        if not symbols:
            symbols = list(DEFAULT_CANDLE_SYMBOLS)
            logger.info("No datasets found, using defaults: %s", symbols)

    logger.info("Symbols: %s", symbols)
    logger.info("Optuna trials: %d", args.trials)

    t0 = time.time()
    results = run_pipeline(symbols, db_path, args.trials)
    elapsed = time.time() - t0

    # Generate report
    report_path = REPORTS_DIR / "training_report.md"
    generate_report(results, report_path)

    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("Report: %s", report_path)

    # Print quick summary
    for r in results:
        t = r.get("trainer", {})
        if t.get("status") == "ok":
            print(f"  {r['symbol']}: AUC={t['test_auc']:.4f} Acc={t['test_accuracy']:.4f}")
        else:
            print(f"  {r['symbol']}: {t.get('status', 'unknown')}")


if __name__ == "__main__":
    main()

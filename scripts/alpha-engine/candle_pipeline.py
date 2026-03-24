"""
Historical 5M Candle ML Pipeline - End-to-End Orchestrator
==========================================================

Loads historical 5M candles from the data-pipeline's MarketDataStore
(SQLite), computes features, generates labels, builds ML-ready
train/val/test datasets, and persists everything to candle_ml.db.

Usage:
    python candle_pipeline.py                          # defaults
    python candle_pipeline.py --symbol EURUSD           # single symbol
    python candle_pipeline.py --symbols EURUSD,GBPUSD   # multiple
    python candle_pipeline.py --source-db /path/to/market_data.db
    python candle_pipeline.py --interval 5m --limit 50000

Pipeline steps:
    1. Load 5M candles from data-pipeline DB (MarketDataStore)
    2. Compute 50+ ML features (CandleFeatureEngine)
    3. Generate forward-looking labels (CandleLabelEngine)
    4. Build temporal train/val/test dataset (CandleDatasetBuilder)
    5. Persist features, labels, and dataset (CandleMLStore)
    6. Print summary report
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from candle_config import (
    CandleMLConfig,
    CandleFeatureConfig,
    CandleLabelConfig,
    CandleDatasetConfig,
    CANDLE_SYMBOLS,
    DEFAULT_CANDLE_SYMBOLS,
)
from candle_features import CandleFeatureEngine, CandleFeatureVector
from candle_labels import CandleLabelEngine, CandleLabel
from candle_dataset import CandleDatasetBuilder, CandleDataset
from candle_store import CandleMLStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("candle_pipeline")


# ---------------------------------------------------------------------------
# Candle loader (reads from data-pipeline's SQLite DB)
# ---------------------------------------------------------------------------
def load_candles_from_db(
    db_path: str,
    symbol: str,
    interval: str = "5m",
    start_ts: float = 0,
    end_ts: float = 0,
    limit: int = 100000,
) -> list[dict]:
    """Load historical candles from the data-pipeline MarketDataStore DB.

    Returns list of dicts with keys: symbol, interval, open, high, low,
    close, volume, open_time, close_time.
    """
    if not Path(db_path).exists():
        logger.error("Source DB not found: %s", db_path)
        return []

    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row

    if end_ts <= 0:
        end_ts = time.time() + 86400

    if start_ts <= 0:
        rows = conn.execute(
            "SELECT symbol, interval, open, high, low, close, volume, "
            "open_time, close_time "
            "FROM candles WHERE symbol = ? AND interval = ? AND open_time <= ? "
            "ORDER BY open_time DESC LIMIT ?",
            (symbol, interval, end_ts, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]

    rows = conn.execute(
        "SELECT symbol, interval, open, high, low, close, volume, "
        "open_time, close_time "
        "FROM candles WHERE symbol = ? AND interval = ? "
        "AND open_time >= ? AND open_time <= ? "
        "ORDER BY open_time LIMIT ?",
        (symbol, interval, start_ts, end_ts, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
class CandleMLPipeline:
    """End-to-end orchestrator for the historical 5M candle ML pipeline."""

    def __init__(self, config: CandleMLConfig | None = None):
        self._cfg = config or CandleMLConfig()
        self._feature_engine = CandleFeatureEngine(self._cfg.features)
        self._label_engine = CandleLabelEngine(self._cfg.labels)
        self._dataset_builder = CandleDatasetBuilder(self._cfg.dataset)
        self._store = CandleMLStore(self._cfg.db_path)

    def run(
        self,
        symbols: list[str] | None = None,
        start_ts: float = 0,
        end_ts: float = 0,
        limit: int = 100000,
        save: bool = True,
    ) -> dict[str, CandleDataset]:
        """Run the full pipeline for one or more symbols.

        Returns a dict mapping symbol -> CandleDataset.
        """
        symbols = symbols or self._cfg.symbols
        results: dict[str, CandleDataset] = {}

        for symbol in symbols:
            logger.info("=" * 60)
            logger.info("Processing %s (%s)", symbol, self._cfg.interval)
            logger.info("=" * 60)

            ds = self._run_symbol(symbol, start_ts, end_ts, limit, save)
            if ds is not None:
                results[symbol] = ds

        return results

    def _run_symbol(
        self,
        symbol: str,
        start_ts: float,
        end_ts: float,
        limit: int,
        save: bool,
    ) -> CandleDataset | None:
        """Run the pipeline for a single symbol."""

        # Step 1: Load candles
        logger.info("[1/5] Loading %s candles from %s",
                     self._cfg.interval, self._cfg.data_pipeline_db)
        candles = load_candles_from_db(
            self._cfg.data_pipeline_db, symbol, self._cfg.interval,
            start_ts, end_ts, limit,
        )
        if not candles:
            logger.warning("No candles found for %s - skipping", symbol)
            return None
        logger.info("  Loaded %d candles  [%s -> %s]",
                     len(candles),
                     _fmt_ts(candles[0].get("open_time", 0)),
                     _fmt_ts(candles[-1].get("open_time", 0)))

        # Step 2: Compute features
        logger.info("[2/5] Computing features...")
        features = self._feature_engine.compute(candles, symbol)
        if not features:
            logger.warning("  No features produced (need >= %d candles)",
                           self._cfg.features.min_warmup)
            return None
        logger.info("  Computed %d feature vectors (%d features each)",
                     len(features), len(features[0].to_dict()))

        # Step 3: Generate labels
        logger.info("[3/5] Generating labels...")
        labels = self._label_engine.label_all(candles, symbol)
        if not labels:
            logger.warning("  No labels produced")
            return None
        logger.info("  Generated %d label vectors (%d labels each)",
                     len(labels), len(labels[0].to_dict()))

        # Step 4: Build dataset
        logger.info("[4/5] Building dataset (temporal split)...")
        dataset = self._dataset_builder.build(
            features, labels, symbol, self._cfg.interval,
        )
        summary = dataset.summary()
        logger.info("  Raw samples:   %d", summary["total_raw"])
        logger.info("  Valid samples: %d", summary["total_valid"])
        logger.info("  Train: %d  Val: %d  Test: %d",
                     summary["train"]["size"],
                     summary["val"]["size"],
                     summary["test"]["size"])
        logger.info("  Features: %d  Norm: %s",
                     summary["feature_count"], summary["norm_method"])

        # Step 5: Persist
        if save:
            logger.info("[5/5] Persisting to %s...", self._cfg.db_path)
            n_feat = self._store.save_features(features, self._cfg.interval)
            n_lbl = self._store.save_labels(labels, self._cfg.interval)
            ds_id = self._store.save_dataset(dataset)
            logger.info("  Saved %d features, %d labels, dataset #%d",
                         n_feat, n_lbl, ds_id)
        else:
            logger.info("[5/5] Skipping persistence (save=False)")

        return dataset

    def get_store(self) -> CandleMLStore:
        """Access the underlying store for queries."""
        return self._store

    def close(self):
        self._store.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt_ts(ts: float) -> str:
    """Format a unix timestamp as ISO string."""
    if ts <= 0:
        return "N/A"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Historical 5M Candle ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python candle_pipeline.py
  python candle_pipeline.py --symbol EURUSD
  python candle_pipeline.py --symbols EURUSD,GBPUSD,USDJPY
  python candle_pipeline.py --source-db /opt/trading-desk/data/market_data.db
  python candle_pipeline.py --limit 20000 --no-save
        """,
    )
    parser.add_argument(
        "--symbol", type=str, default="",
        help="Single symbol to process",
    )
    parser.add_argument(
        "--symbols", type=str, default="",
        help="Comma-separated list of symbols",
    )
    parser.add_argument(
        "--interval", type=str, default="5m",
        help="Candle interval (default: 5m)",
    )
    parser.add_argument(
        "--source-db", type=str, default="",
        help="Path to data-pipeline's market_data.db",
    )
    parser.add_argument(
        "--output-db", type=str, default="",
        help="Path for candle ML output DB",
    )
    parser.add_argument(
        "--limit", type=int, default=100000,
        help="Max candles to load per symbol (default: 100000)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Run pipeline without persisting results",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show store statistics and exit",
    )

    args = parser.parse_args()

    # Build config
    cfg = CandleMLConfig()
    cfg.interval = args.interval

    if args.source_db:
        cfg.data_pipeline_db = args.source_db
    if args.output_db:
        cfg.db_path = args.output_db

    # Determine symbols
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = list(DEFAULT_CANDLE_SYMBOLS)

    # Run
    pipeline = CandleMLPipeline(cfg)

    if args.stats:
        stats = pipeline.get_store().get_stats()
        print(json.dumps(stats, indent=2))
        pipeline.close()
        return

    t0 = time.time()
    results = pipeline.run(
        symbols=symbols,
        limit=args.limit,
        save=not args.no_save,
    )
    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Symbols processed: {len(results)}")
    print(f"Elapsed: {elapsed:.1f}s")
    for sym, ds in results.items():
        s = ds.summary()
        print(f"\n  {sym}:")
        print(f"    Samples: {s['total_valid']} valid / {s['total_raw']} raw")
        print(f"    Train: {s['train']['size']}  "
              f"Val: {s['val']['size']}  "
              f"Test: {s['test']['size']}")
        print(f"    Features: {s['feature_count']}")

    if not args.no_save:
        stats = pipeline.get_store().get_stats()
        print(f"\n  Store: {stats['db_path']}")
        print(f"    Total features: {stats['total_features']}")
        print(f"    Total labels:   {stats['total_labels']}")
        print(f"    Total datasets: {stats['total_datasets']}")

    pipeline.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Dukascopy Historical 5M Candle Downloader
==========================================

Downloads 5-minute OHLCV candles from Dukascopy via `npx dukascopy-node`,
then combines the monthly CSV chunks into a single Parquet file per symbol.

Usage:
    python download_5m_dukascopy.py                        # all defaults
    python download_5m_dukascopy.py --symbols EURUSD,GBPUSD
    python download_5m_dukascopy.py --start 2024-06-01 --end 2025-12-31
    python download_5m_dukascopy.py --output-dir ./my_data

Requirements:
    - Node.js / npx  (for dukascopy-node)
    - pandas, pyarrow  (for CSV→Parquet conversion)

Output:
    data/historical/candles_5m/{SYMBOL}_5M.parquet
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("download_5m_dukascopy")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "XAUUSD",
    "EURJPY", "GBPJPY", "AUDUSD", "USDCAD",
]

DEFAULT_START = "2024-01-01"
DEFAULT_END = "2026-03-23"

ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "historical" / "candles_5m"

# Dukascopy instrument mapping (lowercase as dukascopy-node expects)
DUKASCOPY_INSTRUMENTS = {
    "EURUSD": "eurusd",
    "GBPUSD": "gbpusd",
    "USDJPY": "usdjpy",
    "XAUUSD": "xauusd",
    "EURJPY": "eurjpy",
    "GBPJPY": "gbpjpy",
    "AUDUSD": "audusd",
    "USDCAD": "usdcad",
    "USDCHF": "usdchf",
    "NZDUSD": "nzdusd",
    "EURGBP": "eurgbp",
    "CHFJPY": "chfjpy",
    "CADJPY": "cadjpy",
    "NZDJPY": "nzdjpy",
    "BTCUSD": "btcusd",
}


# ---------------------------------------------------------------------------
# Month chunker (Dukascopy works best with monthly chunks)
# ---------------------------------------------------------------------------
def _month_ranges(start: str, end: str) -> list[tuple[str, str]]:
    """Split a date range into monthly chunks for reliable downloads."""
    fmt = "%Y-%m-%d"
    s = datetime.strptime(start, fmt)
    e = datetime.strptime(end, fmt)
    ranges = []

    current = s
    while current < e:
        # End of this month
        if current.month == 12:
            month_end = datetime(current.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = datetime(current.year, current.month + 1, 1) - timedelta(days=1)
        chunk_end = min(month_end, e)
        ranges.append((current.strftime(fmt), chunk_end.strftime(fmt)))
        current = chunk_end + timedelta(days=1)

    return ranges


# ---------------------------------------------------------------------------
# Download a single chunk via npx dukascopy-node
# ---------------------------------------------------------------------------
def _download_chunk(
    instrument: str,
    start_date: str,
    end_date: str,
    output_path: str,
    timeframe: str = "m5",
) -> bool:
    """Run npx dukascopy-node to download a CSV chunk.

    Returns True on success, False on failure.
    """
    cmd = [
        "npx", "dukascopy-node",
        "-i", instrument,
        "-from", start_date,
        "-to", end_date,
        "-t", timeframe,
        "-f", "csv",
        "-dir", str(Path(output_path).parent),
        "-fn", Path(output_path).stem,
    ]

    logger.debug("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning(
                "dukascopy-node failed for %s %s→%s: %s",
                instrument, start_date, end_date,
                result.stderr.strip()[:200],
            )
            return False
        return True
    except FileNotFoundError:
        logger.error(
            "npx not found. Install Node.js and run: npm install -g dukascopy-node"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.warning(
            "Timeout downloading %s %s→%s", instrument, start_date, end_date
        )
        return False


# ---------------------------------------------------------------------------
# Combine CSVs into Parquet
# ---------------------------------------------------------------------------
def _combine_csvs_to_parquet(
    csv_paths: list[Path],
    output_parquet: Path,
    symbol: str,
) -> int:
    """Combine multiple CSV files into a single Parquet file.

    Tries pandas+pyarrow first, falls back to pure-CSV output.
    Returns row count.
    """
    try:
        import pandas as pd

        dfs = []
        for csv_path in csv_paths:
            if csv_path.exists() and csv_path.stat().st_size > 0:
                df = pd.read_csv(csv_path)
                dfs.append(df)

        if not dfs:
            logger.warning("No CSV data found for %s", symbol)
            return 0

        combined = pd.concat(dfs, ignore_index=True)

        # Standardise column names
        col_map = {}
        for col in combined.columns:
            cl = col.strip().lower()
            if cl in ("timestamp", "date", "time", "datetime"):
                col_map[col] = "timestamp"
            elif cl == "open":
                col_map[col] = "open"
            elif cl == "high":
                col_map[col] = "high"
            elif cl == "low":
                col_map[col] = "low"
            elif cl == "close":
                col_map[col] = "close"
            elif cl in ("volume", "vol"):
                col_map[col] = "volume"
        combined.rename(columns=col_map, inplace=True)

        # Add symbol column
        combined["symbol"] = symbol

        # Sort by timestamp and deduplicate
        if "timestamp" in combined.columns:
            combined.sort_values("timestamp", inplace=True)
            combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)

        combined.reset_index(drop=True, inplace=True)

        # Write parquet
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_parquet, index=False, engine="pyarrow")
        logger.info(
            "  Wrote %d rows to %s", len(combined), output_parquet
        )
        return len(combined)

    except ImportError:
        logger.warning(
            "pandas/pyarrow not available - falling back to combined CSV"
        )
        return _combine_csvs_fallback(csv_paths, output_parquet, symbol)


def _combine_csvs_fallback(
    csv_paths: list[Path],
    output_path: Path,
    symbol: str,
) -> int:
    """Fallback: combine CSVs into a single CSV (rename .parquet → .csv)."""
    csv_output = output_path.with_suffix(".csv")
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    header_written = False

    with open(csv_output, "w", newline="") as out_f:
        writer = None
        for csv_path in csv_paths:
            if not csv_path.exists() or csv_path.stat().st_size == 0:
                continue
            with open(csv_path, "r") as in_f:
                reader = csv.reader(in_f)
                header = next(reader, None)
                if header and not header_written:
                    writer = csv.writer(out_f)
                    writer.writerow(header + ["symbol"])
                    header_written = True
                if writer:
                    for row in reader:
                        writer.writerow(row + [symbol])
                        total_rows += 1

    logger.info(
        "  Wrote %d rows to %s (CSV fallback)", total_rows, csv_output
    )
    return total_rows


# ---------------------------------------------------------------------------
# Main download orchestrator
# ---------------------------------------------------------------------------
def download_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> int:
    """Download all 5M data for a symbol, combine into Parquet.

    Returns total row count.
    """
    instrument = DUKASCOPY_INSTRUMENTS.get(symbol, symbol.lower())
    output_parquet = output_dir / f"{symbol}_5M.parquet"

    logger.info("Downloading %s (%s → %s)...", symbol, start_date, end_date)

    # Create temp dir for CSV chunks
    with tempfile.TemporaryDirectory(prefix=f"dukascopy_{symbol}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        chunks = _month_ranges(start_date, end_date)
        csv_paths: list[Path] = []

        for idx, (chunk_start, chunk_end) in enumerate(chunks):
            csv_name = f"{symbol}_{chunk_start}_{chunk_end}"
            csv_file = tmp_path / f"{csv_name}.csv"

            logger.info(
                "  [%d/%d] %s → %s",
                idx + 1, len(chunks), chunk_start, chunk_end,
            )

            success = _download_chunk(
                instrument, chunk_start, chunk_end, str(csv_file),
            )

            # dukascopy-node may create the file with a different name
            # Check for any new CSV files in the temp dir
            found_csvs = list(tmp_path.glob(f"{csv_name}*"))
            if not found_csvs:
                found_csvs = list(tmp_path.glob("*.csv"))
            for f in found_csvs:
                if f not in csv_paths:
                    csv_paths.append(f)

            if not success:
                logger.warning(
                    "  Chunk %s→%s failed, continuing...",
                    chunk_start, chunk_end,
                )

        if not csv_paths:
            logger.error("No data downloaded for %s", symbol)
            return 0

        # Combine all CSVs into Parquet
        total = _combine_csvs_to_parquet(csv_paths, output_parquet, symbol)

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Download 5M historical candles from Dukascopy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Default symbols: {', '.join(DEFAULT_SYMBOLS)}
Default range:   {DEFAULT_START} to {DEFAULT_END}
Output:          data/historical/candles_5m/{{SYMBOL}}_5M.parquet

Prerequisites:
  npm install -g dukascopy-node
  pip install pandas pyarrow
        """,
    )
    parser.add_argument(
        "--symbols", type=str, default="",
        help=f"Comma-separated symbols (default: {','.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument(
        "--start", type=str, default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end", type=str, default=DEFAULT_END,
        help=f"End date YYYY-MM-DD (default: {DEFAULT_END})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="",
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else DEFAULT_SYMBOLS
    )
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Dukascopy 5M Historical Data Download")
    logger.info("=" * 60)
    logger.info("Symbols: %s", ", ".join(symbols))
    logger.info("Range:   %s → %s", args.start, args.end)
    logger.info("Output:  %s", output_dir)
    logger.info("")

    results = {}
    for symbol in symbols:
        try:
            count = download_symbol(symbol, args.start, args.end, output_dir)
            results[symbol] = count
        except Exception as e:
            logger.error("Failed to download %s: %s", symbol, e)
            results[symbol] = 0

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    total = 0
    for sym, count in results.items():
        status = f"{count:,} rows" if count > 0 else "FAILED"
        print(f"  {sym:8s}  {status}")
        total += count
    print(f"\n  Total: {total:,} rows across {len(results)} symbols")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()

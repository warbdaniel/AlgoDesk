#!/usr/bin/env python3
"""Cron job: Regime scanner - detect regime changes."""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.api_client import RegimeAPI, FixAPI
from lib.symbols import id_to_name, all_symbols

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
CHANGES_LOG = os.path.join(REPORTS_DIR, "regime_changes.log")
LAST_SCAN_FILE = os.path.join(REPORTS_DIR, ".last_regime_scan.json")


def scan():
    regime_api = RegimeAPI()
    fix = FixAPI()

    # Get symbols to scan
    prices_data, _ = fix.prices()
    symbols = []
    if prices_data and isinstance(prices_data, dict):
        symbols = [(sid, id_to_name(sid)) for sid in prices_data.keys()]
    if not symbols:
        symbols = all_symbols()

    # Load previous scan
    last_scan = {}
    if os.path.exists(LAST_SCAN_FILE):
        with open(LAST_SCAN_FILE, "r") as f:
            last_scan = json.load(f)

    now = datetime.now().isoformat()
    current_scan = {}
    changes = []

    for sid, sname in symbols:
        data, err = regime_api.regime(sname, "1h")
        if not data:
            continue

        regime = data.get("regime", "UNKNOWN")
        current_scan[sname] = regime

        prev = last_scan.get(sname)
        if prev and prev != regime:
            change = f"[{now}] {sname}: {prev} -> {regime} (conf: {data.get('confidence', 0):.1%})"
            changes.append(change)

    # Save current scan
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(LAST_SCAN_FILE, "w") as f:
        json.dump(current_scan, f, indent=2)

    # Log changes
    if changes:
        with open(CHANGES_LOG, "a") as f:
            for c in changes:
                f.write(c + "\n")
                print(c)
    else:
        print(f"[{now}] No regime changes detected")


if __name__ == "__main__":
    scan()

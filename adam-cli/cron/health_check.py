#!/usr/bin/env python3
"""Cron job: Health check - alert on service failures."""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.api_client import RegimeAPI, FixAPI, DataAPI, DashboardAPI, ClaudeAPI

ALERTS_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "alerts.log")


def check():
    services = [
        ("Regime Detector", 5000, RegimeAPI()),
        ("Dashboard", 5100, DashboardAPI()),
        ("FIX API", 5200, FixAPI()),
        ("Data Pipeline", 5300, DataAPI()),
        ("Claude API Agent", 5400, ClaudeAPI()),
    ]

    now = datetime.now().isoformat()
    alerts = []

    for name, port, api in services:
        data, err, elapsed = api.health()
        if not data:
            alerts.append(f"[{now}] ALERT: {name} (:{port}) is DOWN - {err}")

        # Check FIX connection health
        if data and name == "FIX API":
            if not data.get("price_connected"):
                alerts.append(f"[{now}] WARN: FIX Price connection is disconnected")
            if not data.get("trade_connected"):
                alerts.append(f"[{now}] WARN: FIX Trade connection is disconnected")
            if data.get("kill_switch_active"):
                alerts.append(f"[{now}] CRITICAL: Kill switch is ACTIVE")

    if alerts:
        os.makedirs(os.path.dirname(ALERTS_LOG), exist_ok=True)
        with open(ALERTS_LOG, "a") as f:
            for alert in alerts:
                f.write(alert + "\n")
                print(alert)
    else:
        print(f"[{now}] All services healthy")


if __name__ == "__main__":
    check()

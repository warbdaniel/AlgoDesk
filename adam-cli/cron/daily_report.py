#!/usr/bin/env python3
"""Cron job: Generate daily summary report."""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.api_client import RegimeAPI, FixAPI, DataAPI, DashboardAPI, ClaudeAPI
from lib.symbols import id_to_name, all_symbols

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "daily")


def generate():
    fix = FixAPI()
    regime_api = RegimeAPI()
    dash = DashboardAPI()
    data_api = DataAPI()
    claude = ClaudeAPI()

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    lines = [
        f"AlgoDesk Daily Report - {date_str}",
        "=" * 50,
        "",
    ]

    # Health
    lines.append("SERVICE HEALTH")
    lines.append("-" * 30)
    for name, port, api in [
        ("Regime Detector", 5000, regime_api),
        ("Dashboard", 5100, dash),
        ("FIX API", 5200, fix),
        ("Data Pipeline", 5300, data_api),
        ("Claude API", 5400, claude),
    ]:
        data, err, elapsed = api.health()
        status = "UP" if data else "DOWN"
        lines.append(f"  {name:20s} :{port}  {status}  {elapsed:.0f}ms")
    lines.append("")

    # Account
    acct, err = fix.account()
    if acct:
        lines.append("ACCOUNT SUMMARY")
        lines.append("-" * 30)
        for key in ["balance", "equity", "free_margin", "margin_level"]:
            val = acct.get(key)
            if val is not None:
                lines.append(f"  {key.replace('_', ' ').title():20s} {val:,.2f}")
        lines.append("")

    # Positions
    pos_data, _ = fix.positions()
    positions = pos_data.get("positions", []) if isinstance(pos_data, dict) else (pos_data or [])
    lines.append(f"OPEN POSITIONS: {len(positions) if isinstance(positions, list) else 0}")
    lines.append("-" * 30)
    if isinstance(positions, list):
        for p in positions:
            sym = p.get("symbol", "?")
            side = p.get("side", p.get("direction", "?"))
            vol = p.get("volume", p.get("quantity", 0))
            pnl = p.get("pnl", p.get("unrealized_pnl", 0))
            lines.append(f"  {sym:10s} {side:4s} {vol:.2f} lots  P&L: {pnl:+.2f}")
    lines.append("")

    # Analytics
    analytics, _ = dash.analytics()
    if analytics:
        overall = analytics.get("overall", analytics)
        lines.append("PERFORMANCE")
        lines.append("-" * 30)
        for key in ["total_trades", "win_rate", "profit_factor", "net_pnl", "sharpe_ratio"]:
            val = overall.get(key)
            if val is not None:
                lines.append(f"  {key.replace('_', ' ').title():20s} {val}")
    lines.append("")

    # Regimes
    prices_data, _ = fix.prices()
    symbols = []
    if prices_data and isinstance(prices_data, dict):
        symbols = [(sid, id_to_name(sid)) for sid in list(prices_data.keys())[:10]]
    if not symbols:
        symbols = all_symbols()[:10]

    lines.append("MARKET REGIMES (top 10)")
    lines.append("-" * 30)
    for sid, sname in symbols:
        data, _ = regime_api.regime(sname, "1h")
        if data:
            lines.append(f"  {sname:10s} {data.get('regime', '?'):15s} {data.get('confidence', 0):.1%}  {data.get('direction', '?')}")
    lines.append("")

    lines.append(f"Generated: {now.isoformat()}")

    # Write report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filepath = os.path.join(REPORTS_DIR, f"{date_str}.txt")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    print(f"Daily report saved to {filepath}")


if __name__ == "__main__":
    generate()

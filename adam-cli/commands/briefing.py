"""adam briefing - Combined morning report."""

import click
from rich.console import Console
from rich.panel import Panel

from lib.api_client import RegimeAPI, FixAPI, DataAPI, DashboardAPI, ClaudeAPI
from lib.symbols import id_to_name, all_symbols
from lib.config import get_alert
from lib.formatter import (
    health_table, account_panel, positions_table, regime_table,
    print_error, console, colorize_pnl, print_json,
)

console_out = Console()


@click.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def briefing(as_json):
    """Morning briefing: health + account + positions + regimes + alerts."""
    fix = FixAPI()
    regime_api = RegimeAPI()
    data_api = DataAPI()
    dash = DashboardAPI()
    claude = ClaudeAPI()

    json_output = {} if as_json else None

    # --- 1. Health Check ---
    services_info = [
        ("Regime Detector", 5000, regime_api),
        ("Dashboard", 5100, dash),
        ("FIX API", 5200, fix),
        ("Data Pipeline", 5300, data_api),
        ("Claude API Agent", 5400, claude),
    ]

    health_results = []
    for name, port, api in services_info:
        data, err, elapsed = api.health()
        health_results.append({
            "name": name, "port": port,
            "status": "UP" if data else "DOWN",
            "time_ms": elapsed, "details": "",
        })

    if as_json:
        json_output["health"] = health_results
    else:
        console.rule("[bold]Morning Briefing[/bold]")
        console.print()
        health_table(health_results)
        console.print()

    # --- 2. Account Summary ---
    acct_data, acct_err = fix.account()
    if as_json:
        json_output["account"] = acct_data
    else:
        if acct_data:
            account_panel(acct_data)
        else:
            print_error(f"Account data unavailable: {acct_err}")
        console.print()

    # --- 3. Open Positions ---
    pos_data, pos_err = fix.positions()
    positions = pos_data.get("positions", pos_data) if isinstance(pos_data, dict) else pos_data
    if as_json:
        json_output["positions"] = positions
    else:
        if positions:
            positions_table(positions if isinstance(positions, list) else [])
        else:
            console.print("[dim]No open positions[/dim]")
        console.print()

    # --- 4. Regime Scan (top 10) ---
    prices_data, _ = fix.prices()
    symbols_to_scan = []
    if prices_data and isinstance(prices_data, dict):
        symbols_to_scan = [(sid, id_to_name(sid)) for sid in list(prices_data.keys())[:10]]
    elif prices_data and isinstance(prices_data, list):
        symbols_to_scan = [(p.get("symbol"), id_to_name(p.get("symbol", ""))) for p in prices_data[:10]]

    if not symbols_to_scan:
        symbols_to_scan = all_symbols()[:10]

    regimes = []
    for sid, sname in symbols_to_scan:
        data, err = regime_api.regime(sname, "1h")
        if data:
            regimes.append({"symbol": sname, **data})

    if as_json:
        json_output["regimes"] = regimes
    else:
        if regimes:
            regime_table(regimes)
        console.print()

    # --- 5. Technical Alerts ---
    rsi_ob = get_alert("rsi_overbought") or 70
    rsi_os = get_alert("rsi_oversold") or 30

    alerts = []
    for sid, sname in symbols_to_scan:
        feat, err = data_api.features_latest(sname, "1h")
        if not feat:
            continue
        rsi = feat.get("rsi", feat.get("rsi_14"))
        macd_hist = feat.get("macd_histogram", feat.get("macd_hist"))

        if rsi and (rsi > rsi_ob or rsi < rsi_os):
            label = "OVERBOUGHT" if rsi > rsi_ob else "OVERSOLD"
            alerts.append(f"  {sname}: RSI {rsi:.1f} - [yellow]{label}[/yellow]")

        if macd_hist and abs(macd_hist) < 0.00005:
            alerts.append(f"  {sname}: [cyan]MACD Cross[/cyan]")

    if as_json:
        json_output["alerts"] = alerts
        print_json(json_output)
    else:
        if alerts:
            alert_text = "\n".join(alerts)
            console.print(Panel(alert_text, title="Technical Alerts", border_style="yellow"))
        else:
            console.print("[dim]No technical alerts[/dim]")

        console.print()
        up = sum(1 for r in health_results if r["status"] == "UP")
        console.rule(f"[dim]{up}/{len(health_results)} services up[/dim]")

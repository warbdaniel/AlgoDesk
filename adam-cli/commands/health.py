"""adam health - Check all services."""

import click
from lib.api_client import RegimeAPI, FixAPI, DataAPI, DashboardAPI, ClaudeAPI
from lib.formatter import console, health_table, print_json


@click.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def health(as_json):
    """Check health of all AlgoDesk services."""
    services = [
        ("Regime Detector", 5000, RegimeAPI()),
        ("Dashboard", 5100, DashboardAPI()),
        ("FIX API", 5200, FixAPI()),
        ("Data Pipeline", 5300, DataAPI()),
        ("Claude API Agent", 5400, ClaudeAPI()),
    ]

    results = []
    for name, port, api in services:
        data, err, elapsed = api.health()
        details = ""
        if data and name == "FIX API":
            parts = []
            if data.get("price_connected"):
                parts.append("Price: [green]Connected[/green]")
            else:
                parts.append("Price: [red]Disconnected[/red]")
            if data.get("trade_connected"):
                parts.append("Trade: [green]Connected[/green]")
            else:
                parts.append("Trade: [red]Disconnected[/red]")
            if data.get("kill_switch_active"):
                parts.append("[bold red]KILL SWITCH ACTIVE[/bold red]")
            details = " | ".join(parts)
        elif data and name == "Data Pipeline":
            ds = data.get("data_store", {})
            if ds:
                details = f"Symbols: {ds.get('symbols', '?')} | Candles: {ds.get('candles', '?')}"

        results.append({
            "name": name,
            "port": port,
            "status": "UP" if data else "DOWN",
            "time_ms": elapsed,
            "details": details,
            "error": err,
        })

    if as_json:
        print_json([{k: v for k, v in r.items() if k != "details"} for r in results])
    else:
        health_table(results)

        up = sum(1 for r in results if r["status"] == "UP")
        total = len(results)
        color = "green" if up == total else "yellow" if up > 0 else "red"
        console.print(f"\n[{color}]{up}/{total} services healthy[/{color}]")

"""Rich terminal formatting for Adam CLI."""

import json as json_module
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

# Regime color mapping
REGIME_COLORS = {
    "STRONG_TREND": "green",
    "MILD_TREND": "blue",
    "RANGING": "yellow",
    "CHOPPY": "red",
}

# Direction colors
DIRECTION_COLORS = {
    "BULL": "green",
    "BUY": "green",
    "BEAR": "red",
    "SELL": "red",
}

# Status colors
STATUS_UP = "[bold green]UP[/bold green]"
STATUS_DOWN = "[bold red]DOWN[/bold red]"
STATUS_WARN = "[bold yellow]WARN[/bold yellow]"


def print_json(data):
    """Print raw JSON for --json flag."""
    console.print_json(json_module.dumps(data, default=str))


def print_error(msg):
    console.print(f"[bold red]Error:[/bold red] {msg}")


def print_success(msg):
    console.print(f"[bold green]✓[/bold green] {msg}")


def print_warning(msg):
    console.print(f"[bold yellow]![/bold yellow] {msg}")


def colorize_regime(regime):
    color = REGIME_COLORS.get(regime, "white")
    return f"[{color}]{regime}[/{color}]"


def colorize_direction(direction):
    color = DIRECTION_COLORS.get(direction, "white")
    return f"[{color}]{direction}[/{color}]"


def colorize_pnl(value):
    if value > 0:
        return f"[green]+{value:.2f}[/green]"
    elif value < 0:
        return f"[red]{value:.2f}[/red]"
    return f"{value:.2f}"


def colorize_percent(value, invert=False):
    """Color a percentage value. Green=good, Red=bad. invert flips meaning."""
    if invert:
        color = "red" if value > 0 else "green" if value < 0 else "white"
    else:
        color = "green" if value > 0 else "red" if value < 0 else "white"
    return f"[{color}]{value:.2f}%[/{color}]"


def health_table(services):
    """Render health check table.
    services: list of dicts with keys: name, port, status, time_ms, details
    """
    table = Table(title="AlgoDesk Service Health", show_lines=True)
    table.add_column("Service", style="bold")
    table.add_column("Port", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Response", justify="right")
    table.add_column("Details")

    for svc in services:
        status = STATUS_UP if svc["status"] == "UP" else STATUS_DOWN
        time_str = f"{svc['time_ms']:.0f}ms" if svc["time_ms"] > 0 else "-"
        table.add_row(svc["name"], str(svc["port"]), status, time_str, svc.get("details", ""))

    console.print(table)


def prices_table(prices_data):
    """Render prices table. prices_data: dict of symbol -> price info."""
    table = Table(title="Live Prices")
    table.add_column("Symbol", style="bold")
    table.add_column("Bid", justify="right")
    table.add_column("Ask", justify="right")
    table.add_column("Spread (pips)", justify="right")
    table.add_column("Mid", justify="right")

    for symbol in sorted(prices_data.keys()):
        p = prices_data[symbol]
        bid = p.get("bid", 0)
        ask = p.get("ask", 0)
        mid = (bid + ask) / 2 if bid and ask else 0
        # Estimate pip size: JPY pairs use 0.01, others 0.0001
        pip_size = 0.01 if "JPY" in str(symbol).upper() else 0.0001
        spread_pips = (ask - bid) / pip_size if pip_size else 0
        digits = 3 if "JPY" in str(symbol).upper() else 5

        spread_color = "green" if spread_pips < 2 else "yellow" if spread_pips < 5 else "red"
        table.add_row(
            str(symbol),
            f"{bid:.{digits}f}",
            f"{ask:.{digits}f}",
            f"[{spread_color}]{spread_pips:.1f}[/{spread_color}]",
            f"{mid:.{digits}f}",
        )

    console.print(table)


def regime_table(regimes):
    """Render regime scan table. regimes: list of dicts with symbol, regime, confidence, direction, volatility."""
    table = Table(title="Market Regime Scan")
    table.add_column("Symbol", style="bold")
    table.add_column("Regime", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Direction", justify="center")
    table.add_column("Volatility", justify="center")

    for r in regimes:
        regime_str = colorize_regime(r.get("regime", "?"))
        conf = r.get("confidence", 0)
        conf_color = "green" if conf > 0.7 else "yellow" if conf > 0.4 else "red"
        direction = colorize_direction(r.get("direction", "?"))
        vol = r.get("volatility", "?")
        vol_color = "red" if vol == "EXPANDING" else "green"

        table.add_row(
            r["symbol"],
            regime_str,
            f"[{conf_color}]{conf:.1%}[/{conf_color}]",
            direction,
            f"[{vol_color}]{vol}[/{vol_color}]",
        )

    console.print(table)


def positions_table(positions):
    """Render positions table."""
    if not positions:
        console.print("[dim]No open positions[/dim]")
        return

    table = Table(title="Open Positions")
    table.add_column("Symbol", style="bold")
    table.add_column("Side", justify="center")
    table.add_column("Volume", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("P&L", justify="right")

    for pos in positions:
        side = colorize_direction(pos.get("side", pos.get("direction", "?")))
        pnl = pos.get("pnl", pos.get("unrealized_pnl", 0))
        table.add_row(
            str(pos.get("symbol", "?")),
            side,
            f"{pos.get('volume', pos.get('quantity', 0)):.2f}",
            f"{pos.get('entry_price', pos.get('avg_price', 0)):.5f}",
            f"{pos.get('current_price', pos.get('market_price', 0)):.5f}",
            colorize_pnl(pnl),
        )

    console.print(table)


def orders_table(orders):
    """Render orders table."""
    if not orders:
        console.print("[dim]No open orders[/dim]")
        return

    table = Table(title="Orders")
    table.add_column("Order ID", style="bold")
    table.add_column("Symbol")
    table.add_column("Side", justify="center")
    table.add_column("Type", justify="center")
    table.add_column("Volume", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Status", justify="center")

    for o in orders:
        side = colorize_direction(o.get("side", "?"))
        table.add_row(
            str(o.get("cl_ord_id", o.get("order_id", "?")))[:12],
            str(o.get("symbol", "?")),
            side,
            o.get("type", o.get("order_type", "?")),
            f"{o.get('quantity', o.get('volume', 0)):.2f}",
            f"{o.get('price', 0):.5f}",
            o.get("status", "?"),
        )

    console.print(table)


def account_panel(data):
    """Render account summary panel."""
    if not data:
        print_error("No account data available")
        return

    lines = []
    for key in ["balance", "equity", "free_margin", "margin", "margin_level", "unrealized_pnl"]:
        val = data.get(key)
        if val is not None:
            label = key.replace("_", " ").title()
            if key == "margin_level":
                lines.append(f"  {label}: {val:.1f}%")
            else:
                lines.append(f"  {label}: ${val:,.2f}")

    panel = Panel("\n".join(lines) if lines else "  No data", title="Account Summary", border_style="blue")
    console.print(panel)


def risk_panel(data):
    """Render risk status panel."""
    if not data:
        print_error("No risk data available")
        return

    kill = data.get("kill_switch_active", False)
    kill_str = "[bold red]ACTIVE[/bold red]" if kill else "[green]Inactive[/green]"

    lines = [f"  Kill Switch: {kill_str}"]

    limits = data.get("limits", data.get("risk_limits", {}))
    counters = data.get("counters", {})

    if counters:
        lines.append("")
        lines.append("  [bold]Counters:[/bold]")
        for k, v in counters.items():
            label = k.replace("_", " ").title()
            lines.append(f"    {label}: {v}")

    if limits:
        lines.append("")
        lines.append("  [bold]Limits:[/bold]")
        for k, v in limits.items():
            label = k.replace("_", " ").title()
            lines.append(f"    {label}: {v}")

    panel = Panel("\n".join(lines), title="Risk Status", border_style="red" if kill else "green")
    console.print(panel)


def performance_panel(data):
    """Render performance analytics panel."""
    if not data:
        print_error("No analytics data available")
        return

    overall = data.get("overall", data)

    lines = []
    metrics = [
        ("Total Trades", "total_trades", None),
        ("Win Rate", "win_rate", "pct"),
        ("Profit Factor", "profit_factor", "2f"),
        ("Sharpe Ratio", "sharpe_ratio", "2f"),
        ("Expected Value", "expected_value", "money"),
        ("Net P&L", "net_pnl", "money"),
        ("Max Drawdown", "max_drawdown", "money"),
        ("Best Trade", "best_trade", "money"),
        ("Worst Trade", "worst_trade", "money"),
        ("Max Consec Wins", "max_consec_wins", None),
        ("Max Consec Losses", "max_consec_losses", None),
    ]

    for label, key, fmt in metrics:
        val = overall.get(key)
        if val is not None:
            if fmt == "pct":
                lines.append(f"  {label}: {val:.1%}" if val <= 1 else f"  {label}: {val:.1f}%")
            elif fmt == "money":
                lines.append(f"  {label}: {colorize_pnl(val)}")
            elif fmt == "2f":
                lines.append(f"  {label}: {val:.2f}")
            else:
                lines.append(f"  {label}: {val}")

    panel = Panel("\n".join(lines) if lines else "  No data", title="Trading Performance", border_style="cyan")
    console.print(panel)


def trades_table(trades):
    """Render trades history table."""
    if not trades:
        console.print("[dim]No trades found[/dim]")
        return

    table = Table(title=f"Trade History ({len(trades)} trades)")
    table.add_column("ID", style="dim")
    table.add_column("Symbol", style="bold")
    table.add_column("Dir", justify="center")
    table.add_column("Entry", justify="right")
    table.add_column("Exit", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Pips", justify="right")
    table.add_column("Regime")
    table.add_column("Time")

    for t in trades:
        pnl = t.get("pnl", 0)
        pips = t.get("pnl_pips", 0)
        direction = colorize_direction(t.get("direction", "?"))
        regime = colorize_regime(t.get("regime", "?"))
        tid = str(t.get("trade_id", ""))[:8]
        entry_time = str(t.get("entry_time", ""))[:16]

        table.add_row(
            tid,
            t.get("symbol", "?"),
            direction,
            f"{t.get('entry_price', 0):.5f}",
            f"{t.get('exit_price', 0):.5f}",
            colorize_pnl(pnl),
            colorize_pnl(pips),
            regime,
            entry_time,
        )

    console.print(table)


def candles_table(candles, symbol, interval):
    """Render candles table."""
    if not candles:
        console.print("[dim]No candles found[/dim]")
        return

    table = Table(title=f"{symbol} {interval} Candles ({len(candles)})")
    table.add_column("Time", style="dim")
    table.add_column("Open", justify="right")
    table.add_column("High", justify="right")
    table.add_column("Low", justify="right")
    table.add_column("Close", justify="right")
    table.add_column("Volume", justify="right")
    table.add_column("Change", justify="right")

    for c in candles:
        o = c.get("open", 0)
        cl = c.get("close", 0)
        change = ((cl - o) / o * 100) if o else 0
        change_str = colorize_pnl(change)

        from datetime import datetime
        ts = c.get("open_time", c.get("timestamp", 0))
        if isinstance(ts, (int, float)) and ts > 0:
            time_str = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
        else:
            time_str = str(ts)[:16]

        digits = 3 if "JPY" in str(symbol).upper() else 5
        table.add_row(
            time_str,
            f"{o:.{digits}f}",
            f"{c.get('high', 0):.{digits}f}",
            f"{c.get('low', 0):.{digits}f}",
            f"{cl:.{digits}f}",
            str(c.get("volume", 0)),
            change_str,
        )

    console.print(table)


def features_table(data, symbol, interval):
    """Render features in a two-column key-value table."""
    if not data:
        console.print("[dim]No feature data[/dim]")
        return

    table = Table(title=f"{symbol} {interval} Features")
    table.add_column("Indicator", style="bold")
    table.add_column("Value", justify="right")

    for key, val in sorted(data.items()):
        if key in ("timestamp", "open_time", "close_time"):
            continue
        if isinstance(val, float):
            table.add_row(key, f"{val:.6f}")
        else:
            table.add_row(key, str(val))

    console.print(table)


def events_table(events):
    """Render events table."""
    if not events:
        console.print("[dim]No events found[/dim]")
        return

    table = Table(title=f"Data Bus Events ({len(events)})")
    table.add_column("Time", style="dim")
    table.add_column("Type", style="bold")
    table.add_column("Source")
    table.add_column("Data")

    for e in events:
        from datetime import datetime
        ts = e.get("timestamp", 0)
        if isinstance(ts, (int, float)) and ts > 0:
            time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        else:
            time_str = str(ts)[:8]

        data_str = str(e.get("data", e.get("payload", "")))[:60]
        table.add_row(
            time_str,
            e.get("event_type", e.get("type", "?")),
            e.get("source", "?"),
            data_str,
        )

    console.print(table)


def scan_table(setups):
    """Render technical scanner results."""
    if not setups:
        console.print("[dim]No interesting setups found[/dim]")
        return

    table = Table(title="Technical Scanner - Active Setups")
    table.add_column("Symbol", style="bold")
    table.add_column("Signal", justify="center")
    table.add_column("RSI", justify="right")
    table.add_column("MACD", justify="center")
    table.add_column("ADX", justify="right")
    table.add_column("BB %B", justify="right")
    table.add_column("Flags")

    for s in setups:
        flags = ", ".join(s.get("flags", []))
        rsi = s.get("rsi", 0)
        rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "white"
        adx = s.get("adx", 0)
        adx_color = "green" if adx > 25 else "white"

        table.add_row(
            s["symbol"],
            s.get("signal", "-"),
            f"[{rsi_color}]{rsi:.1f}[/{rsi_color}]",
            s.get("macd_status", "-"),
            f"[{adx_color}]{adx:.1f}[/{adx_color}]",
            f"{s.get('bb_pctb', 0):.2f}",
            f"[yellow]{flags}[/yellow]",
        )

    console.print(table)

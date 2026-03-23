"""adam trades - Trade history."""

import click
from lib.api_client import DashboardAPI
from lib.formatter import trades_table, print_json, print_error


@click.command()
@click.option("--symbol", "-s", default=None, help="Filter by symbol")
@click.option("--regime", "-r", default=None, help="Filter by regime")
@click.option("--limit", "-l", default=20, help="Number of trades to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def trades(symbol, regime, limit, as_json):
    """Show trade history with optional filters."""
    dash = DashboardAPI()
    data, err = dash.trades(symbol=symbol, regime=regime, limit=limit)

    if err:
        print_error(f"Dashboard: {err}")
        return

    trade_list = data if isinstance(data, list) else data.get("trades", []) if isinstance(data, dict) else []

    if as_json:
        print_json(trade_list)
    else:
        trades_table(trade_list)

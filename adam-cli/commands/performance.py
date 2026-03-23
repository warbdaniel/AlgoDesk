"""adam performance - Trading analytics."""

import click
from lib.api_client import DashboardAPI
from lib.formatter import performance_panel, print_json, print_error


@click.command()
@click.option("--symbol", "-s", default=None, help="Filter by symbol")
@click.option("--regime", "-r", default=None, help="Filter by regime")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def performance(symbol, regime, as_json):
    """Show trading performance analytics."""
    dash = DashboardAPI()
    data, err = dash.analytics(symbol=symbol, regime=regime)

    if err:
        print_error(f"Dashboard: {err}")
        return

    if as_json:
        print_json(data)
    else:
        performance_panel(data)

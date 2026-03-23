"""adam positions - Show open positions."""

import click
from lib.api_client import FixAPI
from lib.formatter import positions_table, print_json, print_error


@click.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def positions(as_json):
    """Show open positions with unrealized P&L."""
    fix = FixAPI()
    data, err = fix.positions()

    if err:
        print_error(f"FIX API: {err}")
        return

    pos = data.get("positions", data) if isinstance(data, dict) else data
    if not isinstance(pos, list):
        pos = []

    if as_json:
        print_json(pos)
    else:
        positions_table(pos)

"""adam orders - Show open/pending orders."""

import click
from lib.api_client import FixAPI
from lib.formatter import orders_table, print_json, print_error


@click.command()
@click.option("--active", is_flag=True, help="Show only active orders")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def orders(active, as_json):
    """Show open and pending orders."""
    fix = FixAPI()
    data, err = fix.orders(active_only=active)

    if err:
        print_error(f"FIX API: {err}")
        return

    ords = data.get("orders", data) if isinstance(data, dict) else data
    if not isinstance(ords, list):
        ords = []

    if as_json:
        print_json(ords)
    else:
        orders_table(ords)

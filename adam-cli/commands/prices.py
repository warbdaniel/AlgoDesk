"""adam prices - Show live prices."""

import click
from lib.api_client import FixAPI
from lib.symbols import id_to_name
from lib.formatter import prices_table, print_json, print_error


@click.command()
@click.option("--symbol", "-s", default=None, help="Filter by symbol name or ID")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def prices(symbol, as_json):
    """Show live prices from FIX API."""
    fix = FixAPI()

    if symbol:
        data, err = fix.price(symbol)
        if err:
            print_error(f"FIX API: {err}")
            return
        resolved = {id_to_name(symbol): data} if data else {}
    else:
        data, err = fix.prices()
        if err:
            print_error(f"FIX API: {err}")
            return
        # Resolve symbol IDs to names
        resolved = {}
        if isinstance(data, dict):
            for sid, pdata in data.items():
                resolved[id_to_name(sid)] = pdata
        elif isinstance(data, list):
            for p in data:
                sid = p.get("symbol", "?")
                resolved[id_to_name(sid)] = p

    if as_json:
        print_json(resolved)
    else:
        if not resolved:
            print_error("No price data available")
        else:
            prices_table(resolved)

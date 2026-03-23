"""adam candles - Display recent candles."""

import click
from lib.api_client import DataAPI
from lib.symbols import resolve
from lib.formatter import candles_table, print_json, print_error


@click.command()
@click.argument("symbol")
@click.option("--interval", "-i", default="1h", help="Candle interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)")
@click.option("--limit", "-l", default=20, help="Number of candles")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def candles(symbol, interval, limit, as_json):
    """Display recent candles for a symbol."""
    sid, sname = resolve(symbol)
    data_api = DataAPI()
    data, err = data_api.candles(sname, interval, limit)

    if err:
        print_error(f"Data Pipeline: {err}")
        return

    candle_list = data.get("candles", []) if isinstance(data, dict) else data

    if as_json:
        print_json(candle_list)
    else:
        candles_table(candle_list, sname, interval)

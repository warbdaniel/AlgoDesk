"""adam features - Show indicator values."""

import click
from lib.api_client import DataAPI
from lib.symbols import resolve
from lib.formatter import features_table, print_json, print_error


@click.command()
@click.argument("symbol")
@click.option("--interval", "-i", default="1h", help="Candle interval")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def features(symbol, interval, as_json):
    """Show all indicator values for a symbol."""
    sid, sname = resolve(symbol)
    data_api = DataAPI()
    data, err = data_api.features_latest(sname, interval)

    if err:
        print_error(f"Data Pipeline: {err}")
        return

    if as_json:
        print_json(data)
    else:
        features_table(data, sname, interval)

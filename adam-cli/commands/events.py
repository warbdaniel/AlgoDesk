"""adam events - Show data bus events."""

import click
from lib.api_client import DataAPI
from lib.formatter import events_table, print_json, print_error


@click.command()
@click.option("--type", "-t", "event_type", default=None, help="Filter by event type")
@click.option("--limit", "-l", default=20, help="Number of events")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def events(event_type, limit, as_json):
    """Show recent data bus events."""
    data_api = DataAPI()
    data, err = data_api.events(event_type=event_type, limit=limit)

    if err:
        print_error(f"Data Pipeline: {err}")
        return

    event_list = data.get("events", data) if isinstance(data, dict) else data

    if as_json:
        print_json(event_list)
    else:
        events_table(event_list if isinstance(event_list, list) else [])

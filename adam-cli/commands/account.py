"""adam account - Show account summary."""

import click
from lib.api_client import FixAPI
from lib.formatter import account_panel, print_json, print_error


@click.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def account(as_json):
    """Show account balance, equity, margin."""
    fix = FixAPI()
    data, err = fix.account()

    if err:
        print_error(f"FIX API: {err}")
        return

    if as_json:
        print_json(data)
    else:
        account_panel(data)

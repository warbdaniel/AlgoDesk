"""adam risk - Show risk status and violations."""

import click
from lib.api_client import FixAPI
from lib.formatter import risk_panel, print_json, print_error, console
from rich.table import Table


@click.command()
@click.option("--violations", "-v", default=10, help="Number of recent violations to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def risk(violations, as_json):
    """Show risk status, utilization, and recent violations."""
    fix = FixAPI()

    status_data, status_err = fix.risk_status()
    violations_data, violations_err = fix.risk_violations(count=violations)

    if as_json:
        print_json({"status": status_data, "violations": violations_data})
        return

    if status_err:
        print_error(f"Risk status: {status_err}")
    else:
        risk_panel(status_data)

    console.print()

    if violations_err:
        print_error(f"Violations: {violations_err}")
    else:
        vlist = violations_data.get("violations", violations_data) if isinstance(violations_data, dict) else violations_data
        if vlist and isinstance(vlist, list) and len(vlist) > 0:
            table = Table(title="Recent Risk Violations")
            table.add_column("Time", style="dim")
            table.add_column("Type", style="bold red")
            table.add_column("Details")

            for v in vlist[:violations]:
                table.add_row(
                    str(v.get("timestamp", v.get("time", "")))[:19],
                    v.get("type", v.get("violation_type", "?")),
                    str(v.get("details", v.get("message", "")))[:60],
                )
            console.print(table)
        else:
            console.print("[green]No recent violations[/green]")

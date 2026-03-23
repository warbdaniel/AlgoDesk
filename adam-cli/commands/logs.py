"""adam logs - Show PM2 service logs."""

import click
import subprocess
from lib.formatter import console, print_error

SERVICE_MAP = {
    "regime": "regime-detector",
    "regime-detector": "regime-detector",
    "dashboard": "dashboard",
    "fix": "fix-api",
    "fix-api": "fix-api",
    "data": "data-pipeline",
    "data-pipeline": "data-pipeline",
    "claude": "claude-api",
    "claude-api": "claude-api",
    "guardian": "trading-guardian",
}


@click.command()
@click.argument("service")
@click.option("--lines", "-n", default=50, help="Number of log lines")
def logs(service, lines):
    """Show PM2 logs for a service."""
    pm2_name = SERVICE_MAP.get(service.lower())
    if not pm2_name:
        print_error(f"Unknown service '{service}'. Valid: {', '.join(sorted(set(SERVICE_MAP.values())))}")
        return

    console.print(f"[dim]Showing last {lines} lines for {pm2_name}...[/dim]\n")
    try:
        result = subprocess.run(
            ["pm2", "logs", pm2_name, "--lines", str(lines), "--nostream"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            console.print(result.stderr)
        if result.returncode != 0 and not result.stdout and not result.stderr:
            print_error(f"pm2 logs returned exit code {result.returncode}")
    except FileNotFoundError:
        print_error("pm2 not found. Is PM2 installed?")
    except subprocess.TimeoutExpired:
        print_error("Timeout reading logs")

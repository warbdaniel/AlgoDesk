"""adam restart - Restart a PM2 service."""

import click
import subprocess
from lib.formatter import console, print_error, print_success

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
}


@click.command()
@click.argument("service")
def restart(service):
    """Restart a PM2 service (with confirmation)."""
    pm2_name = SERVICE_MAP.get(service.lower())
    if not pm2_name:
        print_error(f"Unknown service '{service}'. Valid: {', '.join(sorted(set(SERVICE_MAP.values())))}")
        return

    if not click.confirm(f"Restart {pm2_name}?"):
        console.print("[dim]Cancelled[/dim]")
        return

    try:
        result = subprocess.run(
            ["pm2", "restart", pm2_name],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            print_success(f"{pm2_name} restarted")
        else:
            print_error(f"Failed: {result.stderr or result.stdout}")
    except FileNotFoundError:
        print_error("pm2 not found. Is PM2 installed?")
    except subprocess.TimeoutExpired:
        print_error("Timeout restarting service")

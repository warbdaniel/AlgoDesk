#!/usr/bin/env python3
"""
Adam CLI - AlgoDesk Trading Desk Command Line Tool

Fast, cheap, repeatable operations via direct REST API calls.
No AI agent required.

Usage:
    adam health              Check all services
    adam briefing            Morning briefing report
    adam prices              Live prices
    adam regime [SYMBOL]     Market regime scan
    adam scan                Technical scanner
    adam positions           Open positions
    adam orders              Open orders
    adam account             Account summary
    adam risk                Risk status
    adam performance         Trading analytics
    adam trades              Trade history
    adam candles SYMBOL      Recent candles
    adam features SYMBOL     Indicator values
    adam events              Data bus events
    adam logs SERVICE        PM2 service logs
    adam restart SERVICE     Restart PM2 service
    adam ask "QUESTION"      Ask Claude API agent
"""

import sys
import os

# Add adam-cli root to path so lib/ and commands/ are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click

from commands.health import health
from commands.prices import prices
from commands.regime import regime
from commands.scan import scan
from commands.briefing import briefing
from commands.positions import positions
from commands.orders import orders
from commands.account import account
from commands.risk import risk
from commands.performance import performance
from commands.trades import trades
from commands.candles import candles
from commands.features import features
from commands.events import events
from commands.logs import logs
from commands.restart import restart
from commands.ask import ask


@click.group()
@click.version_option(version="1.0.0", prog_name="adam")
def cli():
    """Adam CLI - AlgoDesk Trading Desk operations tool.

    Fast, direct REST API calls to all trading desk services.
    No AI agent, no tokens, no latency.
    """
    pass


# Register all commands
cli.add_command(health)
cli.add_command(prices)
cli.add_command(regime)
cli.add_command(scan)
cli.add_command(briefing)
cli.add_command(positions)
cli.add_command(orders)
cli.add_command(account)
cli.add_command(risk)
cli.add_command(performance)
cli.add_command(trades)
cli.add_command(candles)
cli.add_command(features)
cli.add_command(events)
cli.add_command(logs)
cli.add_command(restart)
cli.add_command(ask)


if __name__ == "__main__":
    cli()

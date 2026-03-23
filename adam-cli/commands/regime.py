"""adam regime - Market regime detection."""

import click
from lib.api_client import RegimeAPI, FixAPI
from lib.symbols import all_symbols, resolve, id_to_name
from lib.formatter import regime_table, print_json, print_error, console


@click.command()
@click.argument("symbol", required=False)
@click.option("--timeframe", "-t", default="1h", help="Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def regime(symbol, timeframe, as_json):
    """Detect market regime for symbol(s). If no symbol, scans all."""
    api = RegimeAPI()

    if symbol:
        sid, sname = resolve(symbol)
        data, err = api.regime(sname, timeframe)
        if err:
            print_error(f"Regime API: {err}")
            return
        if as_json:
            print_json(data)
        else:
            regime_table([{"symbol": sname, **data}])
    else:
        # Scan all symbols - get subscribed symbols from FIX API
        fix = FixAPI()
        prices_data, prices_err = fix.prices()

        symbols_to_scan = []
        if prices_data and isinstance(prices_data, dict):
            symbols_to_scan = [(sid, id_to_name(sid)) for sid in prices_data.keys()]
        elif prices_data and isinstance(prices_data, list):
            symbols_to_scan = [(p.get("symbol"), id_to_name(p.get("symbol", ""))) for p in prices_data]

        if not symbols_to_scan:
            # Fall back to known symbols
            symbols_to_scan = all_symbols()

        regimes = []
        for sid, sname in symbols_to_scan:
            data, err = api.regime(sname, timeframe)
            if data:
                regimes.append({"symbol": sname, **data})
            else:
                regimes.append({
                    "symbol": sname,
                    "regime": "ERROR",
                    "confidence": 0,
                    "direction": "?",
                    "volatility": "?",
                })

        if as_json:
            print_json(regimes)
        else:
            regime_table(regimes)

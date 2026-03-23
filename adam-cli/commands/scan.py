"""adam scan - Technical scanner."""

import click
from lib.api_client import DataAPI, FixAPI
from lib.symbols import all_symbols, id_to_name
from lib.config import get_alert
from lib.formatter import scan_table, print_json, print_error


@click.command()
@click.option("--timeframe", "-t", default="1h", help="Timeframe for features")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def scan(timeframe, as_json):
    """Technical scanner - flag interesting setups across all symbols."""
    data_api = DataAPI()
    fix = FixAPI()

    rsi_ob = get_alert("rsi_overbought") or 70
    rsi_os = get_alert("rsi_oversold") or 30
    adx_strong = get_alert("adx_strong") or 25

    # Get subscribed symbols
    prices_data, _ = fix.prices()
    symbols_to_scan = []
    if prices_data and isinstance(prices_data, dict):
        symbols_to_scan = [(sid, id_to_name(sid)) for sid in prices_data.keys()]
    elif prices_data and isinstance(prices_data, list):
        symbols_to_scan = [(p.get("symbol"), id_to_name(p.get("symbol", ""))) for p in prices_data]

    if not symbols_to_scan:
        symbols_to_scan = all_symbols()

    setups = []
    for sid, sname in symbols_to_scan:
        feat_data, err = data_api.features_latest(sname, timeframe)
        if err or not feat_data:
            continue

        flags = []
        signal = "-"

        rsi = feat_data.get("rsi", feat_data.get("rsi_14", 50))
        adx = feat_data.get("adx", feat_data.get("adx_14", 0))
        macd = feat_data.get("macd", 0)
        macd_signal = feat_data.get("macd_signal", 0)
        macd_hist = feat_data.get("macd_histogram", feat_data.get("macd_hist", 0))
        bb_pctb = feat_data.get("bb_percent_b", feat_data.get("bb_pctb", 0.5))

        # RSI extremes
        if rsi and rsi > rsi_ob:
            flags.append(f"RSI Overbought ({rsi:.1f})")
            signal = "SELL"
        elif rsi and rsi < rsi_os:
            flags.append(f"RSI Oversold ({rsi:.1f})")
            signal = "BUY"

        # MACD cross detection (histogram sign change implies cross)
        macd_status = "-"
        if macd_hist:
            if abs(macd_hist) < abs(macd_signal) * 0.1 if macd_signal else abs(macd_hist) < 0.00005:
                flags.append("MACD Cross")
                macd_status = "CROSS"
            elif macd_hist > 0:
                macd_status = "[green]BULL[/green]"
            else:
                macd_status = "[red]BEAR[/red]"

        # ADX strong trend
        if adx and adx > adx_strong:
            flags.append(f"Strong Trend (ADX {adx:.1f})")

        # Bollinger Band squeeze
        bb_upper = feat_data.get("bb_upper", 0)
        bb_lower = feat_data.get("bb_lower", 0)
        bb_mid = feat_data.get("bb_middle", feat_data.get("bb_mid", feat_data.get("sma_20", 0)))
        if bb_upper and bb_lower and bb_mid:
            bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid else 0
            if bb_width < 0.01:
                flags.append("BB Squeeze")

        if flags:
            setups.append({
                "symbol": sname,
                "signal": signal,
                "rsi": rsi or 0,
                "macd_status": macd_status,
                "adx": adx or 0,
                "bb_pctb": bb_pctb or 0,
                "flags": flags,
            })

    if as_json:
        print_json(setups)
    else:
        scan_table(setups)

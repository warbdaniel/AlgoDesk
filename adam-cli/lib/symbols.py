"""Symbol mapping: resolve between cTrader IDs and names."""

import json
import os

from .config import load_config

_mapping = None

# Fallback mapping if file not found
DEFAULT_SYMBOLS = {
    "1": "EURUSD",
    "2": "GBPUSD",
    "3": "EURJPY",
    "4": "USDJPY",
    "5": "AUDUSD",
    "6": "USDCHF",
    "7": "GBPJPY",
    "8": "USDCAD",
    "9": "EURGBP",
    "10": "NZDUSD",
    "11": "EURCHF",
    "12": "AUDJPY",
    "13": "GBPCHF",
    "14": "EURAUD",
    "15": "EURCAD",
    "16": "GBPAUD",
    "17": "GBPCAD",
    "18": "GBPNZD",
    "19": "AUDCAD",
    "20": "AUDNZD",
    "21": "NZDJPY",
    "22": "NZDCAD",
    "23": "NZDCHF",
    "24": "CADCHF",
    "25": "CADJPY",
    "26": "AUDCHF",
    "27": "EURNZD",
    "28": "XAUUSD",
}


def _load_mapping():
    global _mapping
    if _mapping is not None:
        return _mapping

    cfg = load_config()
    path = cfg.get("symbol_mapping_path", "")

    if path and os.path.exists(path):
        with open(path, "r") as f:
            _mapping = json.load(f)
    else:
        _mapping = DEFAULT_SYMBOLS

    return _mapping


def id_to_name(symbol_id):
    """Resolve a symbol ID (e.g. '1') to name (e.g. 'EURUSD')."""
    m = _load_mapping()
    return m.get(str(symbol_id), str(symbol_id))


def name_to_id(name):
    """Resolve a symbol name (e.g. 'EURUSD') to ID (e.g. '1')."""
    m = _load_mapping()
    name_upper = name.upper()
    for sid, sname in m.items():
        if sname.upper() == name_upper:
            return sid
    return name


def resolve(symbol_input):
    """Accept either name or ID, return (id, name) tuple."""
    m = _load_mapping()
    s = str(symbol_input).strip()

    # Check if it's an ID
    if s in m:
        return s, m[s]

    # Check if it's a name
    s_upper = s.upper()
    for sid, sname in m.items():
        if sname.upper() == s_upper:
            return sid, sname

    # Unknown - return as-is
    return s, s


def all_symbols():
    """Return list of (id, name) tuples."""
    m = _load_mapping()
    return [(sid, sname) for sid, sname in sorted(m.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)]

"""HTTP client wrappers for all AlgoDesk internal APIs."""

import requests

from .config import get_service_url, get_default

TIMEOUT = None


def _timeout():
    global TIMEOUT
    if TIMEOUT is None:
        TIMEOUT = get_default("request_timeout") or 5
    return TIMEOUT


def _get(base_url, path, params=None):
    """GET request with error handling. Returns (data, error)."""
    try:
        r = requests.get(f"{base_url}{path}", params=params, timeout=_timeout())
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Connection refused"
    except requests.Timeout:
        return None, "Timeout"
    except requests.HTTPError as e:
        return None, f"HTTP {e.response.status_code}"
    except Exception as e:
        return None, str(e)


def _post(base_url, path, json_data=None):
    """POST request with error handling. Returns (data, error)."""
    try:
        r = requests.post(f"{base_url}{path}", json=json_data, timeout=_timeout())
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Connection refused"
    except requests.Timeout:
        return None, "Timeout"
    except requests.HTTPError as e:
        return None, f"HTTP {e.response.status_code}"
    except Exception as e:
        return None, str(e)


def _delete(base_url, path):
    """DELETE request with error handling. Returns (data, error)."""
    try:
        r = requests.delete(f"{base_url}{path}", timeout=_timeout())
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Connection refused"
    except requests.Timeout:
        return None, "Timeout"
    except requests.HTTPError as e:
        return None, f"HTTP {e.response.status_code}"
    except Exception as e:
        return None, str(e)


def _timed_get(url, path):
    """GET with response time measurement. Returns (data, error, elapsed_ms)."""
    try:
        r = requests.get(f"{url}{path}", timeout=_timeout())
        elapsed_ms = r.elapsed.total_seconds() * 1000
        r.raise_for_status()
        return r.json(), None, elapsed_ms
    except requests.ConnectionError:
        return None, "Connection refused", 0
    except requests.Timeout:
        return None, "Timeout", 0
    except requests.HTTPError as e:
        return None, f"HTTP {e.response.status_code}", 0
    except Exception as e:
        return None, str(e), 0


class RegimeAPI:
    def __init__(self):
        self.base = get_service_url("regime-detector")

    def health(self):
        return _timed_get(self.base, "/health")

    def regime(self, symbol, timeframe="15m"):
        return _post(self.base, "/regime", {"symbol": symbol, "timeframe": timeframe})


class FixAPI:
    def __init__(self):
        self.base = get_service_url("fix-api")

    def health(self):
        return _timed_get(self.base, "/health")

    def prices(self):
        return _get(self.base, "/prices")

    def price(self, symbol):
        return _get(self.base, f"/prices/{symbol}")

    def positions(self):
        return _get(self.base, "/positions")

    def orders(self, active_only=False):
        return _get(self.base, "/orders", {"active_only": active_only})

    def account(self):
        return _get(self.base, "/account")

    def risk_status(self):
        return _get(self.base, "/risk/status")

    def risk_violations(self, count=50):
        return _get(self.base, "/risk/violations", {"count": count})

    def candles(self, symbol, count=100):
        return _get(self.base, f"/candles/{symbol}", {"count": count})


class DataAPI:
    def __init__(self):
        self.base = get_service_url("data-pipeline")

    def health(self):
        return _timed_get(self.base, "/health")

    def symbols(self):
        return _get(self.base, "/symbols")

    def candles(self, symbol, interval="1h", limit=500):
        return _get(self.base, f"/candles/{symbol}/{interval}", {"limit": limit})

    def features(self, symbol, interval="1h", limit=200):
        return _get(self.base, f"/features/{symbol}/{interval}", {"limit": limit})

    def features_latest(self, symbol, interval="1h"):
        return _get(self.base, f"/features/{symbol}/{interval}/latest")

    def feature_names(self):
        return _get(self.base, "/features/names")

    def events(self, event_type=None, limit=50):
        params = {"limit": limit}
        if event_type:
            params["event_type"] = event_type
        return _get(self.base, "/events", params)

    def stats(self):
        return _get(self.base, "/stats")

    def ticks_latest(self, symbol):
        return _get(self.base, f"/ticks/{symbol}/latest")


class DashboardAPI:
    def __init__(self):
        self.base = get_service_url("dashboard")

    def health(self):
        return _timed_get(self.base, "/health")

    def trades(self, symbol=None, regime=None, limit=500, offset=0):
        params = {"limit": limit, "offset": offset}
        if symbol:
            params["symbol"] = symbol
        if regime:
            params["regime"] = regime
        return _get(self.base, "/api/trades", params)

    def analytics(self, symbol=None, regime=None):
        params = {}
        if symbol:
            params["symbol"] = symbol
        if regime:
            params["regime"] = regime
        return _get(self.base, "/api/analytics", params)

    def filters(self):
        return _get(self.base, "/api/filters")


class ClaudeAPI:
    def __init__(self):
        self.base = get_service_url("claude-api")

    def health(self):
        return _timed_get(self.base, "/health")

    def query(self, question):
        return _post(self.base, "/query", {"message": question})

    def query_stream(self, question):
        """Stream a query response. Returns a requests.Response for streaming."""
        try:
            r = requests.post(
                f"{self.base}/query/stream",
                json={"message": question},
                stream=True,
                timeout=60,
            )
            r.raise_for_status()
            return r, None
        except requests.ConnectionError:
            return None, "Connection refused"
        except requests.Timeout:
            return None, "Timeout"
        except Exception as e:
            return None, str(e)

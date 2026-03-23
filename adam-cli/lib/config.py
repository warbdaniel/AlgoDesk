"""Configuration loader for Adam CLI."""

import os
import yaml

_config = None
_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load_config():
    global _config
    if _config is None:
        with open(_config_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config


def get_service_url(service_name):
    cfg = load_config()
    return cfg["services"][service_name]["url"]


def get_default(key):
    cfg = load_config()
    return cfg["defaults"].get(key)


def get_alert(key):
    cfg = load_config()
    return cfg["alerts"].get(key)

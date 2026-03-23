#!/bin/bash
# Adam CLI shell wrapper
# Install to /usr/local/bin/adam
source /opt/trading-desk/venv/bin/activate
python3 /opt/trading-desk/adam-cli/adam.py "$@"

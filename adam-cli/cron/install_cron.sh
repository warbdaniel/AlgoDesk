#!/bin/bash
# Install Adam CLI cron jobs
# Run: bash /opt/trading-desk/adam-cli/cron/install_cron.sh

ADAM_DIR="/opt/trading-desk/adam-cli"
VENV="/opt/trading-desk/venv/bin/python3"
LOG_DIR="$ADAM_DIR/reports"

mkdir -p "$LOG_DIR/daily"

# Build crontab entries
CRON_ENTRIES=$(cat <<EOF
# Adam CLI - Health check every 5 minutes
*/5 * * * * $VENV $ADAM_DIR/cron/health_check.py >> $LOG_DIR/health_cron.log 2>&1

# Adam CLI - Regime scan every 15 minutes (market hours Mon-Fri)
*/15 * * * 1-5 $VENV $ADAM_DIR/cron/regime_scan.py >> $LOG_DIR/regime_cron.log 2>&1

# Adam CLI - Daily report at 23:55 UTC Mon-Fri
55 23 * * 1-5 $VENV $ADAM_DIR/cron/daily_report.py >> $LOG_DIR/daily_cron.log 2>&1
EOF
)

# Check if entries already exist
if crontab -l 2>/dev/null | grep -q "adam-cli"; then
    echo "Adam CLI cron jobs already installed. Updating..."
    # Remove old entries and add new
    crontab -l 2>/dev/null | grep -v "adam-cli\|Adam CLI" | { cat; echo "$CRON_ENTRIES"; } | crontab -
else
    # Append to existing crontab
    (crontab -l 2>/dev/null; echo ""; echo "$CRON_ENTRIES") | crontab -
fi

echo "Cron jobs installed:"
echo "$CRON_ENTRIES"
echo ""
echo "Verify with: crontab -l"

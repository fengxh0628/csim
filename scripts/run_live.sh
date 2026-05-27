#!/bin/bash
# Daily run: download data + update cache + generate signal + execute trades.
#
# Usage:
#   ./scripts/run_live.sh --config prod.yml
#   ./scripts/run_live.sh --config prod.yml --dry-run
#   ./scripts/run_live.sh --config prod.yml --skip-download

set -e

CSIM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CSIM_ROOT"

CONFIG="prod.yml"
DRY_RUN=""
SKIP_DOWNLOAD=""
MAX_RETRIES=3

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2;;
        --dry-run) DRY_RUN="--dry-run"; shift;;
        --skip-download) SKIP_DOWNLOAD="--skip-download"; shift;;
        *) shift;;
    esac
done

DATE=$(date -u +%Y%m%d)
LOG_DIR="$CSIM_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/live_${DATE}.log"

# Tee all output to both stdout and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== csim daily run: ${DATE} $(date -u +%H:%M:%S) UTC ==="

# Step 1: Download data + update cache (with retry)
if [ -z "$SKIP_DOWNLOAD" ]; then
    echo "[1/3] Downloading data and updating cache..."
    for i in $(seq 1 $MAX_RETRIES); do
        if python3 scripts/update_data.py --config "$CONFIG"; then
            break
        else
            if [ $i -eq $MAX_RETRIES ]; then
                echo "ERROR: daily.py failed after $MAX_RETRIES attempts"
                exit 1
            fi
            echo "  Retry $i/$MAX_RETRIES in 30s..."
            sleep 30
        fi
    done
else
    echo "[1/3] Skipping download"
fi

# Step 2: Run csim to generate signal (with checkpoint)
echo "[2/3] Generating signal..."
bin/csim "$CONFIG"

# Step 3: Execute trades
# Find the latest dump file
DUMP_DIR=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
ops = cfg.get('portfolio', {}).get('operations', [])
for op in ops:
    if op.get('moduleId') == 'opdump':
        print(op.get('dumpdir', 'dumps'))
        break
")
LATEST_CSV=$(ls -t "${DUMP_DIR}"/*/*.csv 2>/dev/null | head -1)

if [ -z "$LATEST_CSV" ]; then
    echo "ERROR: No signal CSV found in ${DUMP_DIR}/"
    exit 1
fi

echo "[3/3] Executing: ${LATEST_CSV}..."
python3 scripts/execute.py --positions "$LATEST_CSV" $DRY_RUN

echo "=== Done: $(date -u +%H:%M:%S) UTC ==="

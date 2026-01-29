#!/usr/bin/env bash
#
# Example: Run a TE-piRNA arms race simulation
#
# This script runs a small simulation (N=100, 500 generations) to verify
# the framework is working. Output is written to output/example_run/.
#
# Prerequisites:
#   - SLiM 5.x installed (https://messerlab.org/slim/)
#   - slim executable in PATH or ~/bin/slim
#
# Usage:
#   bash run_example.sh

set -euo pipefail

# Find SLiM executable
SLIM_BIN=""
for candidate in slim ~/bin/slim /usr/local/bin/slim /opt/homebrew/bin/slim; do
    if command -v "$candidate" &>/dev/null; then
        SLIM_BIN="$candidate"
        break
    fi
done

if [ -z "$SLIM_BIN" ]; then
    echo "ERROR: SLiM executable not found. Install from https://messerlab.org/slim/"
    exit 1
fi

echo "Using SLiM: $SLIM_BIN"
$SLIM_BIN -version

# Set up output directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output/example_run"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=== Running TE-piRNA Arms Race Simulation ==="
echo "  Population size:  100"
echo "  Generations:      500"
echo "  Output interval:  50"
echo "  Output dir:       $OUTPUT_DIR"
echo ""

# Run simulation
$SLIM_BIN \
    -d N=100 \
    -d SIM_GENERATIONS=500 \
    -d OUTPUT_INTERVAL=50 \
    -d "OUTPUT_DIR=\"${OUTPUT_DIR}/\"" \
    -s 42 \
    "$SCRIPT_DIR/slim/te_pirna_simulation.slim"

echo ""
echo "=== Output Files ==="
ls -lh "$OUTPUT_DIR/"

echo ""
echo "=== Population Summary ==="
cat "$OUTPUT_DIR/population_summary.tsv"

echo ""
echo "Done. See output in: $OUTPUT_DIR/"

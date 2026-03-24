#!/usr/bin/env bash
# record_demos.sh -- Run both demo scripts in dry-run mode and capture output.
#
# Usage:
#   ./scripts/record_demos.sh              # Capture terminal output to demos/
#   ./scripts/record_demos.sh --live       # Run live (real APIs + browser)
#
# Output:
#   demos/flight_search_output.txt         # Flight search demo terminal output
#   demos/research_report_output.txt       # Research report demo terminal output
#
# For actual GIF/video recordings, see docs/RECORDING_GUIDE.md.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEMOS_DIR="${PROJECT_ROOT}/demos"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"

mkdir -p "${DEMOS_DIR}"

MODE="--dry-run"
if [[ "${1:-}" == "--live" ]]; then
    MODE=""
    echo "Running in LIVE mode (real APIs + browser)"
else
    echo "Running in DRY-RUN mode (mocked data, no API calls)"
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")

# ---------------------------------------------------------------------------
# Demo 1: Flight Search
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Demo 1: Flight Search (AMS -> BCN)"
echo "========================================"
echo ""

FLIGHT_OUTPUT="${DEMOS_DIR}/flight_search_output.txt"

{
    echo "=== GraphBot Flight Search Demo ==="
    echo "Recorded: $(date)"
    echo "Mode: ${MODE:-live}"
    echo ""
    python "${SCRIPTS_DIR}/demo_flight_search.py" ${MODE} --verbose 2>&1
} | tee "${FLIGHT_OUTPUT}"

echo ""
echo "Flight search output saved to: ${FLIGHT_OUTPUT}"

# ---------------------------------------------------------------------------
# Demo 2: Research Report
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Demo 2: Research Report"
echo "========================================"
echo ""

RESEARCH_OUTPUT="${DEMOS_DIR}/research_report_output.txt"

{
    echo "=== GraphBot Research Report Demo ==="
    echo "Recorded: $(date)"
    echo "Mode: ${MODE:-live}"
    echo ""
    python "${SCRIPTS_DIR}/demo_research_report.py" ${MODE} --verbose 2>&1
} | tee "${RESEARCH_OUTPUT}"

echo ""
echo "Research report output saved to: ${RESEARCH_OUTPUT}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  All demos complete"
echo "========================================"
echo ""
echo "Output files:"
echo "  ${FLIGHT_OUTPUT}"
echo "  ${RESEARCH_OUTPUT}"
echo ""
echo "For GIF recordings, see docs/RECORDING_GUIDE.md"

#!/bin/bash
set -e

# Get the repository root directory
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/src"
NOTEBOOK="$SRC_DIR/demo_hessian.ipynb"
TEMP_NOTEBOOK="$SRC_DIR/demo_hessian_tmp.ipynb"

# Ensure cleanup happens on exit (success or failure)
cleanup() {
    if [ -f "$TEMP_NOTEBOOK" ]; then
        rm "$TEMP_NOTEBOOK"
    fi
}
trap cleanup EXIT

# Check if notebook exists
if [ ! -f "$NOTEBOOK" ]; then
    echo "Error: Notebook not found at $NOTEBOOK"
    exit 1
fi

cp "$NOTEBOOK" "$TEMP_NOTEBOOK"

# Run the notebook using the project's virtual environment
# We run in SRC_DIR so local imports/files resolve correctly
cd "$SRC_DIR"
"$REPO_ROOT/.venv/bin/python" -m jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    "$(basename "$TEMP_NOTEBOOK")"

echo "Notebook execution successful."

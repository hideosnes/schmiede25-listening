#!/usr/bin/env bash

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/venv"

echo "Project root: $PROJECT_ROOT"
if [ -d "$VENV_DIR" ]; then
    echo "Using existing venv at $VENV_DIR"
else
    echo "Attempting to create virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
    pip install -r "${PROJECT_ROOT}/requirements.txt"
else
    echo "No requirements.txt found. Add one and re-run this script."
fi

echo ""
echo "Setup complete. To activate the venv in your shell run:"
echo " source ${VENV_DIR}/bin/activate"
echo "Then run the app, e.g.:"
echo " python ${PROJECT_ROOT}/app.py --list"
echo "For a list of all available features run:"
echo " python ${PROJECT_ROOT}/app.py --help"
echo "If you are already 'in' the folder in the CLI, drop the/project/root/path/ and run e.g. 'source venv/bin/active'"
echo ""
echo "<3 MewMew <3 mit Bussi aufs Bauchi:"
echo " curl -sSL <add-raw-github-url-to-setup.sh>/scripts/setup.sh | bash"
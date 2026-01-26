#!/bin/bash
set -e
PYTHON_BIN="/usr/bin/python3.12"
echo "🔧 Starting Environment Repair..."
if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ CRITICAL: Python 3.12 not found ($PYTHON_BIN)."
    echo "   Run: sudo dnf install python3.12 python3.12-devel"
    exit 1
fi
if [ -d ".venv" ]; then rm -rf .venv; fi
echo "🔨 Creating .venv with Python 3.12..."
$PYTHON_BIN -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi
echo "✅ Repair Complete."

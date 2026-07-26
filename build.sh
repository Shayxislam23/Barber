#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
.venv/bin/python -m pip install -q requests pillow numpy
.venv/bin/python solve_lighting.py
mkdir -p public
cp submission_lighting.csv public/submission_lighting.csv
cat > public/index.html <<'HTML'
<html><body><h1>AI Challenge — Lighting</h1><p>Submission generated successfully.</p><a href="/submission_lighting.csv">Download submission_lighting.csv</a></body></html>
HTML

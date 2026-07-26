#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
.venv/bin/python -m pip install -q requests pillow numpy pandas scikit-learn
mkdir -p public
printf '<html><body>all packages installed</body></html>' > public/index.html

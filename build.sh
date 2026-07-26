#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
.venv/bin/python -m pip install -q requests
.venv/bin/python - <<'PY'
from pathlib import Path
import zipfile
import requests

out = Path('/tmp/lighting.zip')
extract = Path('/tmp/lighting_data')
dispatcher = requests.get('https://cloud.mail.ru/api/v2/dispatcher', timeout=60, headers={'User-Agent':'Mozilla/5.0'})
dispatcher.raise_for_status()
prefix = dispatcher.json()['body']['weblink_get'][0]['url'].rstrip('/')
url = f'{prefix}/GCsv/1BXmZPEBj'
with requests.get(url, stream=True, allow_redirects=True, timeout=(60,900), headers={'User-Agent':'Mozilla/5.0','Referer':'https://cloud.mail.ru/public/GCsv/1BXmZPEBj'}) as r:
    r.raise_for_status()
    with out.open('wb') as f:
        for chunk in r.iter_content(4*1024*1024):
            if chunk:
                f.write(chunk)
assert out.stat().st_size > 100_000_000 and zipfile.is_zipfile(out)
extract.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out) as zf:
    zf.extractall(extract)
exts = {'.png','.jpg','.jpeg','.webp','.bmp'}
images = [p for p in extract.rglob('*') if p.is_file() and p.suffix.lower() in exts]
print('images', len(images), flush=True)
assert len(images) >= 1700
Path('public').mkdir(exist_ok=True)
Path('public/index.html').write_text(f'<html><body>extract ok: {len(images)} images</body></html>')
PY

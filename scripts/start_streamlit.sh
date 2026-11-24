#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x ".venv/bin/streamlit" ]; then
  echo "[start_streamlit] Virtual environment missing; bootstrapping .venv..."
  python3 -m venv --without-pip .venv
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  else
    python3 -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')"
  fi
  .venv/bin/python /tmp/get-pip.py --no-cache-dir
  rm -f /tmp/get-pip.py
  .venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
  .venv/bin/pip install --no-cache-dir -r requirements.txt
fi

if [ ! -e "/usr/local/bin/pip3" ] && [ -w "/usr/local/bin" ]; then
  ln -sf "$PROJECT_ROOT/.venv/bin/pip3" /usr/local/bin/pip3 || true
  ln -sf "$PROJECT_ROOT/.venv/bin/pip" /usr/local/bin/pip || true
fi

exec .venv/bin/streamlit run streamlit_app/app.py \
  --server.port=3000 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.enableWebsocketCompression=false \
  --server.allowedHosts='*' \
  --browser.gatherUsageStats=false

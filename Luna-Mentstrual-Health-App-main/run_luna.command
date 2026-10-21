#!/bin/bash
# Double-click to set up (first time only) and start Luna on macOS.
cd "$(dirname "$0")"
PY=""
for v in 3.12 3.11 3.13 3.10; do
  for p in "python$v" "/opt/homebrew/bin/python$v" "$HOME/.homebrew/bin/python$v" "/usr/local/bin/python$v" \
           "/Library/Frameworks/Python.framework/Versions/$v/bin/python3"; do
    if command -v "$p" >/dev/null 2>&1; then PY="$(command -v "$p")"; break 2; fi
  done
done
if [ -z "$PY" ]; then
  for p in "$CONDA_PREFIX/bin/python3" "$HOME/anaconda3/bin/python3" "$HOME/opt/anaconda3/bin/python3" "$HOME/miniconda3/bin/python3" python3; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] <= (3,13)))'; then PY="$(command -v "$p")"; break; fi
  done
fi
[ -z "$PY" ] && { echo "Need Python 3.10-3.13:  brew install python@3.12"; read -p "Press Enter to close"; exit 1; }
echo "Using $PY"
[ -d .venv ] && [ ! -f .venv/.installed ] && rm -rf .venv
if [ ! -d .venv ]; then
  echo "First run: installing packages (5-10 min)..."
  "$PY" -m venv .venv && source .venv/bin/activate && pip install --upgrade pip -q && \
  pip install -r requirements-eval.txt && touch .venv/.installed || { echo "Install failed - copy the error to Claude."; read -p "Press Enter"; exit 1; }
else
  source .venv/bin/activate
fi
pip show beautifulsoup4 >/dev/null 2>&1 || pip install -q beautifulsoup4
[ -f knowledge/web_sources.json ] || { echo "Building the knowledge base (about 1 minute)..."; python ingest.py; }
[ -f .env ] || cp .env.example .env
streamlit run streamlit_app.py

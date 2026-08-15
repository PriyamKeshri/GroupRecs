#!/usr/bin/env bash
# Starts the FastAPI backend (background) and the Streamlit UI (foreground).
# Ctrl+C stops both. No --reload here -- background + auto-reload's
# subprocess model don't play nicely together; use two terminals with
# `uvicorn --reload` instead if you're actively editing src/api.py.
set -e

PY_EXE=$(python3 -c "import sys; print(sys.executable)")
MISSING=$(python3 -c "
import importlib.util
missing = [m for m in ('fastapi', 'uvicorn', 'streamlit') if importlib.util.find_spec(m) is None]
print(' '.join(missing))
")

if [ -n "$MISSING" ]; then
  echo "Missing package(s) for $PY_EXE: $MISSING"
  echo "This is often a different python3 than the one you installed deps into"
  echo "(e.g. a conda env not active in this shell). Fix with either:"
  echo "  pip install -r requirements.txt"
  echo "  # or, if you have multiple Pythons, be explicit:"
  echo "  $PY_EXE -m pip install -r requirements.txt"
  exit 1
fi

python3 -m uvicorn src.api:app --port 8000 &
UVICORN_PID=$!
trap 'kill $UVICORN_PID 2>/dev/null' EXIT

# Backgrounding a failed command doesn't trip `set -e`, so confirm uvicorn
# actually stayed up before trusting it -- otherwise this prints a false
# "API running" and launches the UI against a dead backend.
sleep 1
if ! kill -0 "$UVICORN_PID" 2>/dev/null; then
  echo "API failed to start -- see the error above."
  exit 1
fi

echo "API running at http://localhost:8000 (pid $UVICORN_PID)"
echo "Starting Streamlit UI at http://localhost:8501 ..."
streamlit run streamlit_app.py

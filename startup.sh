#!/bin/bash
set -e

echo "=== Fleet Dashboard — Startup ==="
cd /home/site/wwwroot

# ── 1. Activate Oryx-built venv if it exists ────────────────────────────
if [ -d "antenv" ]; then
    echo ">>> Activating Oryx virtual environment (antenv)…"
    source antenv/bin/activate
elif [ -d ".venv" ]; then
    echo ">>> Activating .venv…"
    source .venv/bin/activate
else
    echo ">>> WARNING: No virtual environment found."
    echo "    Set  SCM_DO_BUILD_DURING_DEPLOYMENT=true  in App Settings"
    echo "    and redeploy so Oryx creates the antenv venv."
    echo ">>> Falling back to pip install into /home/.local …"
    pip install --quiet --user -r requirements.txt
    export PATH="/home/.local/bin:$PATH"
fi

# ── 2. Verify critical packages before starting gunicorn ────────────────
python -c "import dash, flask, pandas; print('  [ok] Core packages available')" || {
    echo ">>> Package check failed. Installing requirements…"
    pip install --quiet -r requirements.txt
}

# ── 3. Verify solver cache exists ───────────────────────────────────────
if [ ! -f "optimization/results/solver_cache.pkl" ]; then
    echo ">>> ERROR: optimization/results/solver_cache.pkl not found."
    echo "    Run 'python main.py --solver all' locally and redeploy."
    exit 1
fi

# ── 4. Launch gunicorn ──────────────────────────────────────────────────
echo ">>> Starting gunicorn app:app on 0.0.0.0:8000 …"
exec gunicorn app:app \
    --bind 0.0.0.0:8000 \
    --timeout 600 \
    --workers 2 \
    --access-logfile - \
    --error-logfile -

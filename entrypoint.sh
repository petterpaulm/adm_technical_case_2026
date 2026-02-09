#!/bin/bash
# Debug / diagnostic script — NOT used as the startup command.
# Use  startup.sh  or the portal startup command instead.
# Run manually via SSH console:  bash entrypoint.sh

set -e

echo "=== FILES AT WWWROOT ==="
ls -la /home/site/wwwroot/

echo "=== LOOKING FOR WSGI.PY ==="
find /home/site/wwwroot -name "wsgi.py" 2>/dev/null || echo "NO wsgi.py FOUND"

echo "=== LOOKING FOR REQUIREMENTS.TXT ==="
find /home/site/wwwroot -name "requirements.txt" 2>/dev/null || echo "NO requirements.txt FOUND"

echo "=== CWD ==="
pwd

echo "=== PYTHON VERSION ==="
python --version

echo "=== INSTALLING DEPS ==="
cd /home/site/wwwroot
pip install -r requirements.txt 2>&1 | tail -5

echo "=== ADDING optimization/ TO PYTHONPATH ==="
export PYTHONPATH="/home/site/wwwroot/optimization:$PYTHONPATH"

echo "=== STARTING GUNICORN ==="
gunicorn wsgi:server --bind 0.0.0.0:8000 --timeout 600 --workers 2

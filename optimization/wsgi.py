"""
WSGI entry point for Azure App Service / gunicorn.

Loads the pre-computed solver cache and exposes the Dash app's
underlying Flask ``server`` so that gunicorn can serve it:

    gunicorn wsgi:server -b 0.0.0.0:8000 --timeout 600

This file lives at optimization/ level (next to main.py).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Ensure the optimization/ directory is on sys.path
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import dash_bootstrap_components as dbc
from dash import Dash

from dashboard import figures as figs
from dashboard.callbacks import register
from dashboard.layouts import build_layout
from src.config import RESULTS_DIR

# ── Load cached solver results ──────────────────────────────────────────
_CACHE_PATH = RESULTS_DIR / "solver_cache.pkl"

if not _CACHE_PATH.exists():
    raise FileNotFoundError(
        f"No cached results at {_CACHE_PATH}.  "
        "Run  python main.py --solver all  locally first, then redeploy."
    )

with open(_CACHE_PATH, "rb") as _f:
    solver_data: dict = pickle.load(_f)

# Pre-compute summary stats (same as run_dashboard does)
for _name, _data in solver_data.items():
    _data["stats"] = figs.compute_summary_stats(
        _data["routes"], _data["costs"], _data["schedules"],
    )

# ── Build the Dash app ──────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    assets_folder=str(_THIS_DIR / "dashboard" / "assets"),
    title="Fleet Analytics",
    update_title="Updating...",
    suppress_callback_exceptions=True,
)

solver_names = list(solver_data.keys())
default_solver = solver_names[0]

app.layout = build_layout(solver_data, default_solver)
register(app, solver_data)

# This is what gunicorn binds to
server = app.server

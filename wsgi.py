"""
WSGI entry point for Azure App Service / gunicorn.

Lives at the REPOSITORY ROOT (Exercise/) so that no --chdir is needed.
Gunicorn invocation:

    gunicorn wsgi:server --bind 0.0.0.0:8000 --timeout 600 --workers 2
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
_log = logging.getLogger("fleet-opt")

# ── Path setup ───────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent          # Exercise/
_OPT  = _ROOT / "optimization"                   # Exercise/optimization/

_log.info("wsgi.py  ROOT=%s  OPT=%s", _ROOT, _OPT)

# Put the optimization directory on sys.path so its packages resolve
if str(_OPT) not in sys.path:
    sys.path.insert(0, str(_OPT))

import dash_bootstrap_components as dbc           # noqa: E402
from dash import Dash                             # noqa: E402

from dashboard import figures as figs             # noqa: E402
from dashboard.callbacks import register          # noqa: E402
from dashboard.layouts import build_layout        # noqa: E402
from src.config import RESULTS_DIR                # noqa: E402

# ── Load cached solver results ───────────────────────────────────────────
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
    assets_folder=str(_OPT / "dashboard" / "assets"),
    title="Fleet Analytics",
    update_title="Updating...",
    suppress_callback_exceptions=True,
)

solver_names = list(solver_data.keys())
default_solver = solver_names[0]

app.layout = build_layout(solver_data, default_solver)
register(app, solver_data)

# ── Gunicorn binds to this ───────────────────────────────────────────────
server = app.server

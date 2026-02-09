"""
Azure App Service entry point.

Gunicorn startup:
    gunicorn app:app --bind 0.0.0.0:8000 --timeout 600 --workers 2
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path

# ── Logging (visible in Azure Log Stream) ────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
_log = logging.getLogger("adm-fleet")

# ── Path setup ───────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent            # /home/site/wwwroot
_OPT  = _ROOT / "optimization"                     # optimization/

_log.info("ROOT = %s", _ROOT)
_log.info("OPT  = %s  (exists=%s)", _OPT, _OPT.is_dir())

if str(_OPT) not in sys.path:
    sys.path.insert(0, str(_OPT))

os.chdir(_OPT)  # dashboard assets use relative paths

import dash_bootstrap_components as dbc            # noqa: E402
from dash import Dash                              # noqa: E402

from dashboard import figures as figs              # noqa: E402
from dashboard.callbacks import register           # noqa: E402
from dashboard.layouts import build_layout         # noqa: E402
from src.config import RESULTS_DIR                 # noqa: E402

# ── Load cached solver results ───────────────────────────────────────────
_CACHE = RESULTS_DIR / "solver_cache.pkl"
_log.info("Cache path = %s  (exists=%s)", _CACHE, _CACHE.exists())

if not _CACHE.exists():
    raise FileNotFoundError(
        f"solver_cache.pkl not found at {_CACHE}. "
        "Run  python main.py --solver all  locally first, then redeploy."
    )

with open(_CACHE, "rb") as fh:
    solver_data: dict = pickle.load(fh)

for _name, _sd in solver_data.items():
    _sd["stats"] = figs.compute_summary_stats(
        _sd["routes"], _sd["costs"], _sd["schedules"],
    )

# ── Build Dash app ──────────────────────────────────────────────────────
_dash = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    assets_folder=str(_OPT / "dashboard" / "assets"),
    title="ADM Fleet Analytics",
    update_title="Updating...",
    suppress_callback_exceptions=True,
)

_names = list(solver_data.keys())
_dash.layout = build_layout(solver_data, _names[0])
register(_dash, solver_data)

# ── This is what gunicorn binds to: app:app ─────────────────────────────
app = _dash.server

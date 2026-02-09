"""
Dash application factory - Fleet Analytics Theme.

Supports single-solver and multi-solver modes.  When ``solver_data``
contains more than one key, the dashboard shows a Solver Selector
dropdown and a cross-strategy comparison section.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Dash

from dashboard import figures as figs
from dashboard.callbacks import register
from dashboard.layouts import build_layout


def run_dashboard(
    solver_data: dict,
    *,
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Build and launch the fleet dashboard.

    Parameters
    ----------
    solver_data : dict
        Mapping of solver name -> dict with keys:
        routes, costs, schedules, metadata, mc_result, solver_name.
    """

    # Pre-compute summary stats for every solver
    for _name, data in solver_data.items():
        data["stats"] = figs.compute_summary_stats(
            data["routes"], data["costs"], data["schedules"],
        )

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        assets_folder="assets",
        title="Fleet Analytics",
        update_title="Updating...",
        suppress_callback_exceptions=True,
    )

    solver_names = list(solver_data.keys())
    default_solver = solver_names[0]

    app.layout = build_layout(solver_data, default_solver)
    register(app, solver_data)

    label = " / ".join(s.upper() for s in solver_names)
    print(f"\n  Fleet Dashboard ({label}) -> http://localhost:{port}\n")
    app.run(debug=debug, port=port)

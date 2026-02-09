"""
main.py -- Entry point for the Fleet Delivery Optimisation system.

Usage
-----
    python main.py                          Run default solver (ALNS)
    python main.py --solver greedy          Use the greedy baseline
    python main.py --solver colgen          Use column generation
    python main.py --solver pareto          Multi-objective Pareto
    python main.py --solver all             Run greedy + colgen + alns
    python main.py --solver all --dashboard Compare all strategies side by side
    python main.py --dashboard-only         Reload cached results, skip solvers

Pipeline
--------
  1. Load & validate data  (orders, items, trucks)
  2. Run selected optimisation solver(s)
  3. Compute costs & CO2
  4. Build HOS-compliant schedule
  5. Export results (CSV + console)
  6. Run Monte Carlo risk simulation (automatic when --dashboard is used)
  7. Cache all results to disk (results/solver_cache.pkl)
  8. Optionally launch the multi-solver Plotly Dash dashboard
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from src.config import RESULTS_DIR
from src.cost_engine import all_route_costs
from src.data_loader import load_all
from src.reporting import banner, export_all
from src.scheduler import build_schedule
from src.solvers.alns import ALNSSolver
from src.solvers.column_gen import ColumnGenSolver
from src.solvers.greedy import GreedySolver
from src.solvers.pareto import ParetoSolver

_CACHE_PATH = RESULTS_DIR / "solver_cache.pkl"

SOLVERS = {
    "greedy": GreedySolver,
    "alns": ALNSSolver,
    "colgen": ColumnGenSolver,
    "pareto": ParetoSolver,
}

# "all" runs these three (pareto is excluded -- it calls ALNS internally)
_ALL_SOLVERS = ["greedy", "colgen", "alns"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fleet delivery route optimisation")
    ap.add_argument(
        "--solver",
        choices=list(SOLVERS.keys()) + ["all"],
        default="alns",
        help=(
            "Optimisation strategy (default: alns). "
            "Use 'all' to run greedy + colgen + alns and compare them."
        ),
    )
    ap.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the interactive Plotly Dash dashboard after solving",
    )
    ap.add_argument(
        "--monte-carlo",
        action="store_true",
        help="Run Monte Carlo risk simulation (1 000 trials)",
    )
    ap.add_argument(
        "--dashboard-only",
        action="store_true",
        help=(
            "Skip solver execution -- reload the last saved results and "
            "launch the dashboard immediately. Requires a prior run."
        ),
    )
    return ap.parse_args()


# -- Single-solver pipeline ----------------------------------------------


def _run_solver(
    solver_name: str,
    orders,
    trucks,
    distances,
    *,
    run_mc: bool = False,
) -> dict:
    """Run one solver through the full pipeline and return results dict."""
    solver_cls = SOLVERS[solver_name]

    banner(f"OPTIMISING  [{solver_cls.name}]")
    t0 = time.perf_counter()
    solver = solver_cls()
    solution = solver.solve(orders, trucks, distances)
    routes = solution.routes
    print(f"  {len(routes)} routes generated")
    if routes:
        print(f"  Max stops : {max(r.num_stops for r in routes)}")
        print(f"  Heaviest  : {max(r.total_weight_kg for r in routes):,.0f} kg")

    # costs & carbon
    banner(f"COSTING  [{solver_name.upper()}]")
    costs = all_route_costs(routes)
    total_cost = sum(c["total_cost"] for c in costs)
    total_co2 = sum(c["co2_kg"] for c in costs)
    print(f"  Fleet cost : ${total_cost:,.2f}")
    print(f"  CO2 emitted: {total_co2:,.0f} kg  ({total_co2 / 1000:.1f} t)")

    # schedule
    banner(f"SCHEDULING  [{solver_name.upper()}]")
    schedules = build_schedule(routes, distances, costs)
    hos = sum(1 for s in schedules if s.needs_rest)
    print(f"  HOS rest breaks: {hos}/{len(schedules)}")
    if schedules:
        first = min(s.departure for s in schedules)
        last = max(s.arrival for s in schedules)
        print(f"  First departure: {first.strftime('%a %b %d, %Y %I:%M %p')}")
        print(f"  Last return    : {last.strftime('%a %b %d, %Y %I:%M %p')}")

    # export CSV (last solver overwrites)
    export_all(routes, costs, schedules)

    if solution.metadata:
        meta_path = RESULTS_DIR / "solution_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(solution.metadata, f, indent=2, default=str)
        print(f"  [ok] Metadata -> {meta_path.name}")

    # optional Monte Carlo
    mc_result = None
    if run_mc:
        banner(f"MONTE CARLO  [{solver_name.upper()}]")
        from src.monte_carlo import run_simulation

        mc_result = run_simulation(routes, costs)
        print(f"  Trials           : {mc_result.n_trials:,}")
        print(f"  Deterministic    : ${mc_result.deterministic_cost:,.0f}")
        print(f"  P50              : ${mc_result.cost_p50:,.0f}")
        print(f"  P95              : ${mc_result.cost_p95:,.0f}")
        print(f"  Robustness score : {mc_result.robustness_score:.0f}%")

    elapsed = time.perf_counter() - t0
    print(f"\n  {solver_name.upper()} finished in {elapsed:.1f}s")

    return {
        "solver_name": solution.solver_name or solver_name,
        "routes": routes,
        "costs": costs,
        "schedules": schedules,
        "metadata": solution.metadata,
        "mc_result": mc_result,
    }


# -- Persistence helpers -------------------------------------------------


def _save_cache(solver_data: dict[str, dict]) -> None:
    """Persist solver results to disk so the dashboard can reload later."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(solver_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [ok] Results cached -> {_CACHE_PATH.name}")


def _load_cache() -> dict[str, dict]:
    """Load previously saved solver results from disk."""
    if not _CACHE_PATH.exists():
        print("  ERROR: No cached results found.")
        print(f"         Expected file: {_CACHE_PATH}")
        print("         Run the solvers first (without --dashboard-only).")
        sys.exit(1)
    with open(_CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    solvers = ", ".join(s.upper() for s in data)
    print(f"  Loaded cached results for: {solvers}")
    return data


# -- Main ----------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()

    # -- fast path: reload cached results --------------------------------
    if args.dashboard_only:
        banner("LOADING CACHED RESULTS")
        solver_data = _load_cache()
        banner("Launching Dashboard")
        from dashboard.app import run_dashboard

        run_dashboard(solver_data=solver_data)
        return

    # -- step 1: data ----------------------------------------------------
    banner("STEP 1 -- Loading Data")
    items, orders, trucks, distances = load_all()
    print(f"  {len(items)} items, {len(orders)} order lines, {len(trucks)} trucks")
    dests = sorted({o.destination for o in orders})
    print(f"  {len(dests)} unique destinations")
    total_kg = sum(o.total_weight_kg for o in orders)
    print(f"  Total cargo: {total_kg:,.0f} kg")

    # -- step 2: determine solvers ---------------------------------------
    solver_list = _ALL_SOLVERS if args.solver == "all" else [args.solver]
    if len(solver_list) > 1:
        label = ", ".join(s.upper() for s in solver_list)
        banner(f"MULTI-SOLVER RUN: {label}")
        print("  Results from every strategy will appear in one dashboard.\n")

    # Monte Carlo runs automatically when the dashboard is requested
    run_mc = args.monte_carlo or args.dashboard

    # -- step 3: run each solver -----------------------------------------
    solver_data: dict[str, dict] = {}
    for sname in solver_list:
        solver_data[sname] = _run_solver(
            sname, orders, trucks, distances, run_mc=run_mc,
        )

    elapsed = time.perf_counter() - t0
    banner("PIPELINE COMPLETE")
    print(f"  Solvers : {', '.join(s.upper() for s in solver_list)}")
    print(f"  Total: {elapsed:.1f}s\n")

    # -- step 4: save results to disk ------------------------------------
    _save_cache(solver_data)

    # -- step 5: dashboard -----------------------------------------------
    if args.dashboard:
        banner("Launching Dashboard")
        from dashboard.app import run_dashboard

        run_dashboard(solver_data=solver_data)


if __name__ == "__main__":
    main()

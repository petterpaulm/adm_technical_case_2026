"""
Fleet Command Center -- Dash callbacks (multi-solver).

Handles:
  - Navigation (landing → dashboard, back, tile clicks)
  - Tab switching (Route / Cost / Drivers / Risk / Sustainability / Strategy)
  - Sidebar collapse/expand
  - Solver switching → KPI ribbon + route dropdown refresh
  - 20+ chart callbacks driven by solver-selector & route-dropdown
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dash import ALL, Input, Output, State, callback_context, no_update, html

from dashboard import figures as figs

if TYPE_CHECKING:
    from dash import Dash


def register(app: Dash, solver_data: dict) -> None:
    """Bind every callback to the app."""

    def _get(solver_name: str) -> dict:
        return solver_data.get(solver_name, next(iter(solver_data.values())))

    def _filter(selected_ids, routes, costs, schedules):
        if not selected_ids:
            return routes, costs, schedules
        idx = {sid for sid in selected_ids}
        fr = [r for r in routes if r.route_id in idx]
        fc = [c for r, c in zip(routes, costs, strict=False) if r.route_id in idx]
        fs = [s for s in schedules if s.route_id in idx]
        return fr, fc, fs

    # ------------------------------------------------------------------
    # NAVIGATION: tile click → open dashboard on specific tab
    # ------------------------------------------------------------------
    _tile_ids = [
        "tile-routes", "tile-cost", "tile-drivers",
        "tile-risk", "tile-sustainability", "tile-strategy",
    ]
    _tab_btn_ids = [
        "tab-btn-routes", "tab-btn-cost", "tab-btn-drivers",
        "tab-btn-risk", "tab-btn-sustainability", "tab-btn-strategy",
    ]
    _tab_names = [
        "routes", "cost", "drivers", "risk", "sustainability", "strategy",
    ]

    @app.callback(
        [
            Output("page-landing", "style"),
            Output("page-dashboard", "style"),
            Output("active-tab", "data"),
        ],
        [Input(tid, "n_clicks") for tid in _tile_ids] +
        [Input("btn-back", "n_clicks")],
        [State("active-tab", "data")],
        prevent_initial_call=True,
    )
    def _navigate(*args):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        # Back button
        if trigger == "btn-back":
            return {"display": "flex"}, {"display": "none"}, no_update

        # Tile click → go to dashboard
        for tile_id, tab_name in zip(_tile_ids, _tab_names):
            if trigger == tile_id:
                return {"display": "none"}, (
                    {"display": "flex", "flexDirection": "column",
                     "height": "100vh", "width": "100%"}
                ), tab_name

        return no_update, no_update, no_update

    # ------------------------------------------------------------------
    # TAB SWITCHING: navbar tab buttons
    # ------------------------------------------------------------------
    @app.callback(
        [Output("sidebar-tab-content", "children")] +
        [Output(tid, "className") for tid in _tab_btn_ids],
        [Input(tid, "n_clicks") for tid in _tab_btn_ids] +
        [Input("active-tab", "data")],
        prevent_initial_call=True,
    )
    def _switch_tab(*args):
        from dashboard.layouts import TAB_BUILDERS

        ctx = callback_context
        # Determine which tab to show
        tab = "routes"
        if ctx.triggered:
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger == "active-tab":
                # Came from tile click
                tab = args[-1] or "routes"
            else:
                for btn_id, tname in zip(_tab_btn_ids, _tab_names):
                    if trigger == btn_id:
                        tab = tname
                        break

        default_solver = list(solver_data.keys())[0]
        builder = TAB_BUILDERS.get(tab, TAB_BUILDERS["routes"])
        content = builder(solver_data, default_solver)

        # Set active class on correct tab button
        classes = []
        for tname in _tab_names:
            classes.append("nav-tab active" if tname == tab else "nav-tab")

        return [content] + classes

    # ------------------------------------------------------------------
    # SIDEBAR TOGGLE
    # ------------------------------------------------------------------
    @app.callback(
        [Output("dash-sidebar", "className"),
         Output("sidebar-toggle", "children")],
        Input("sidebar-toggle", "n_clicks"),
        State("sidebar-state", "data"),
        prevent_initial_call=True,
    )
    def _toggle_sidebar(n, state):
        if state == "open":
            return "dash-sidebar collapsed", "\u276E"
        return "dash-sidebar", "\u276F"

    @app.callback(
        Output("sidebar-state", "data"),
        Input("sidebar-toggle", "n_clicks"),
        State("sidebar-state", "data"),
        prevent_initial_call=True,
    )
    def _toggle_state(n, state):
        return "closed" if state == "open" else "open"

    # ------------------------------------------------------------------
    # Solver switch -- update KPIs, dropdown, table
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("sidebar-kpi", "children"),
            Output("route-dropdown", "options"),
            Output("route-dropdown", "value"),
            Output("route-table-container", "children"),
        ],
        Input("solver-selector", "value"),
    )
    def _switch_solver(solver_name):
        from dashboard.layouts import build_kpi_cards, build_route_table

        d = _get(solver_name)
        stats = d["stats"]
        routes = d["routes"]
        mc = d.get("mc_result")

        kpis = build_kpi_cards(stats, mc)
        opts = [
            {"label": f"Route {r.route_id} - "
                      f"{r.truck.driver if r.truck else '?'} "
                      f"({r.num_stops} stops, {r.total_weight_kg:,.0f} kg)",
             "value": r.route_id}
            for r in routes
        ]
        all_ids = [r.route_id for r in routes]
        table = build_route_table(routes)

        return kpis, opts, all_ids, table

    # -- Shared inputs for all chart callbacks
    _inputs = [Input("solver-selector", "value"), Input("route-dropdown", "value")]

    # -- 1. Route map
    @app.callback(Output("route-map", "figure"), _inputs)
    def _map(solver, selected):
        d = _get(solver)
        fr, _, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.route_map(fr, d["schedules"])

    # -- 2. Cost waterfall
    @app.callback(Output("cost-waterfall", "figure"), _inputs)
    def _waterfall(solver, selected):
        d = _get(solver)
        _, fc, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.cost_waterfall(fc)

    # -- 3. Cost bar
    @app.callback(Output("cost-bar", "figure"), _inputs)
    def _cost_bar(solver, selected):
        d = _get(solver)
        fr, fc, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.cost_breakdown_bar(fr, fc)

    # -- 4. Utilisation
    @app.callback(Output("utilisation", "figure"), _inputs)
    def _util(solver, selected):
        d = _get(solver)
        fr, _, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.utilisation_bars(fr)

    # -- 5. Distance vs Weight scatter
    @app.callback(Output("dist-weight", "figure"), _inputs)
    def _scatter(solver, selected):
        d = _get(solver)
        fr, fc, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.dist_weight_scatter(fr, fc)

    # -- 6. Driver radar
    @app.callback(Output("driver-radar", "figure"), _inputs)
    def _radar(solver, selected):
        d = _get(solver)
        fr, fc, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.driver_radar(fr, fc)

    # -- 7. Cost treemap
    @app.callback(Output("cost-treemap", "figure"), _inputs)
    def _treemap(solver, selected):
        d = _get(solver)
        fr, fc, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.cost_treemap(fr, fc)

    # -- 8. Gantt chart
    @app.callback(Output("gantt", "figure"), _inputs)
    def _gantt(solver, selected):
        d = _get(solver)
        _, _, fs = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.gantt_chart(fs)

    # -- 9. Distance heatmap
    @app.callback(Output("heatmap", "figure"), _inputs)
    def _heatmap(solver, selected):
        d = _get(solver)
        fr, fc, fs = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.driver_distance_heatmap(fr, fc, fs)

    # -- 10. Pareto front (multi-solver comparison or single-solver front)
    @app.callback(Output("pareto", "figure"), _inputs)
    def _pareto(solver, _selected):
        d = _get(solver)
        return figs.pareto_scatter(
            d["metadata"],
            solver_data=solver_data if len(solver_data) >= 2 else None,
        )

    # -- 26. ALNS operator adaptation
    @app.callback(Output("alns-operators", "figure"), _inputs)
    def _alns_ops(solver, _selected):
        # Prefer the ALNS solver's metadata if present
        if "alns" in solver_data:
            meta = solver_data["alns"]["metadata"]
        else:
            meta = _get(solver)["metadata"]
        return figs.alns_operator_weights(meta)

    # -- 11. Carbon by route
    @app.callback(Output("carbon-route", "figure"), _inputs)
    def _carbon_route(solver, selected):
        d = _get(solver)
        fr, fc, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.carbon_by_route(fr, fc)

    # -- 12. Carbon waterfall
    @app.callback(Output("carbon-waterfall", "figure"), _inputs)
    def _carbon_wf(solver, selected):
        d = _get(solver)
        fr, fc, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.carbon_waterfall(fr, fc)

    # -- 13. Cost Sankey
    @app.callback(Output("cost-sankey", "figure"), _inputs)
    def _sankey(solver, selected):
        d = _get(solver)
        fr, fc, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.cost_sankey(fr, fc)

    # -- 14. Efficiency frontier
    @app.callback(Output("efficiency-frontier", "figure"), _inputs)
    def _frontier(solver, selected):
        d = _get(solver)
        fr, fc, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.efficiency_frontier(fr, fc)

    # -- 15. Delivery timeline
    @app.callback(Output("delivery-timeline", "figure"), _inputs)
    def _timeline(solver, selected):
        d = _get(solver)
        _, _, fs = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.delivery_timeline(fs)

    # -- 16. Fuel price sensitivity (what-if)
    @app.callback(Output("fuel-sensitivity", "figure"), _inputs)
    def _fuel_sens(solver, selected):
        d = _get(solver)
        fr, fc, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.fuel_sensitivity_waterfall(fr, fc)

    # -- 17. Driver performance matrix
    @app.callback(Output("driver-matrix", "figure"), _inputs)
    def _drv_matrix(solver, selected):
        d = _get(solver)
        fr, fc, fs = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.driver_performance_matrix(fr, fc, fs)

    # -- 18. Hub & spoke radial network
    @app.callback(Output("network-radial", "figure"), _inputs)
    def _network(solver, selected):
        d = _get(solver)
        fr, _, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.route_network_radial(fr)

    # -- 21. Truck class efficiency
    @app.callback(Output("truck-class-eff", "figure"), _inputs)
    def _truck_eff(solver, selected):
        d = _get(solver)
        fr, fc, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.truck_class_efficiency(fr, fc)

    # -- 22. Economies of distance
    @app.callback(Output("econ-distance", "figure"), _inputs)
    def _econ_dist(solver, selected):
        d = _get(solver)
        fr, fc, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.economies_of_distance(fr, fc)

    # -- 23. HOS compliance
    @app.callback(Output("hos-compliance", "figure"), _inputs)
    def _hos(solver, selected):
        d = _get(solver)
        fr, fc, fs = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.hos_compliance(fr, fc, fs)

    # -- 24. Customer-destination heatmap
    @app.callback(Output("customer-heatmap", "figure"), _inputs)
    def _cust_heat(solver, selected):
        d = _get(solver)
        fr, _, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.customer_dest_heatmap(fr)

    # -- 25. Stops distribution
    @app.callback(Output("stops-distribution", "figure"), _inputs)
    def _stops_dist(solver, selected):
        d = _get(solver)
        fr, _, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.stops_distribution(fr)

    # -- 19. Demand density map
    @app.callback(Output("demand-density", "figure"), _inputs)
    def _demand(solver, selected):
        d = _get(solver)
        fr, _, _ = _filter(selected, d["routes"], d["costs"], d["schedules"])
        return figs.demand_heatmap_geo(fr)

    # -- 20. Fleet KPI gauges
    @app.callback(Output("fleet-gauges", "figure"), _inputs)
    def _gauges(solver, _selected):
        d = _get(solver)
        return figs.fleet_kpi_gauges(d["stats"], d.get("mc_result"))

    # -- 16. MC cost histogram
    @app.callback(Output("mc-histogram", "figure"), _inputs)
    def _mc_hist(solver, _selected):
        d = _get(solver)
        return figs.mc_cost_histogram(d.get("mc_result"))

    # -- 14. Risk tornado
    @app.callback(Output("risk-tornado", "figure"), _inputs)
    def _tornado(solver, _selected):
        d = _get(solver)
        return figs.risk_tornado(d.get("mc_result"))

    # -- 15. Robustness gauge
    @app.callback(Output("robustness-gauge", "figure"), _inputs)
    def _gauge(solver, _selected):
        d = _get(solver)
        return figs.robustness_gauge(d.get("mc_result"))

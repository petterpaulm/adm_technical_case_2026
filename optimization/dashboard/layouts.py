"""
Fleet Command Center -- CropMonitor-inspired layout.

Architecture:
  PAGE 1 - Landing page with branded hero + clickable tile grid
  PAGE 2 - Map-centric view with left sidebar panels that slide over the map

Navigation:
  dcc.Location tracks the URL.
  /            -> landing page
  /dashboard   -> map + sidebar analytics view

The sidebar has tabs (like CropMonitor's Rainfall / Temperature / Forecast
tabs) that switch between analytics categories: Routes, Cost, Drivers,
Risk, Sustainability, Strategy.

Each tab loads its own set of chart panels in a left-side slide-over panel
while the map fills the background.
"""

from __future__ import annotations

from dash import dash_table, dcc, html

from dashboard import figures as figs

# -- Corporate palette constants for inline styles
_NAVY    = "#00263A"
_GOLD    = "#D4A843"
_GREEN   = "#4A7C59"
_CRIMSON = "#C8102E"
_SKY     = "#5B9BD5"
_CREAM   = "#FAF7F2"
_SLATE   = "#6B7B8D"
_WHEAT   = "#E8D5B7"
_DTXT    = "#2D2D2D"
_WHITE   = "#FFFFFF"


# =====================================================================
# LANDING PAGE TILES
# =====================================================================

_TILES = [
    {
        "id": "tile-routes",
        "icon": "01",
        "title": "Route Network",
        "desc": "Interactive US map with all delivery routes, "
                "stops, and depot connections",
        "color": _NAVY,
        "tab": "routes",
    },
    {
        "id": "tile-cost",
        "icon": "02",
        "title": "Cost Analytics",
        "desc": "Waterfall, treemap, Sankey flow, and per-route "
                "cost decomposition",
        "color": _GOLD,
        "tab": "cost",
    },
    {
        "id": "tile-drivers",
        "icon": "03",
        "title": "Driver Intelligence",
        "desc": "Radar charts, performance matrix, workload "
                "heatmaps, and HOS timeline",
        "color": _GREEN,
        "tab": "drivers",
    },
    {
        "id": "tile-risk",
        "icon": "04",
        "title": "Monte Carlo Risk",
        "desc": "1,000-trial stochastic simulation with tornado "
                "sensitivity and robustness scoring",
        "color": _CRIMSON,
        "tab": "risk",
    },
    {
        "id": "tile-sustainability",
        "icon": "05",
        "title": "Sustainability",
        "desc": "CO2 emissions tracking, carbon waterfall, and "
                "demand density analysis",
        "color": "#2D8E8E",
        "tab": "sustainability",
    },
    {
        "id": "tile-strategy",
        "icon": "06",
        "title": "Strategy Comparison",
        "desc": "Cross-solver benchmarking: Greedy vs Column Gen "
                "vs ALNS -- cost, efficiency, Pareto",
        "color": _SKY,
        "tab": "strategy",
    },
]


def _tile_card(tile: dict, stats: dict) -> html.Div:
    return html.Div(
        className="landing-tile",
        id=tile["id"],
        n_clicks=0,
        **{"data-tab": tile["tab"]},
        children=[
            html.Div(className="tile-accent",
                     style={"background": tile["color"]}),
            html.Div(className="tile-body", children=[
                html.Div(className="tile-number", children=tile["icon"],
                         style={"color": tile["color"],
                                "borderColor": tile["color"]}),
                html.H3(tile["title"], className="tile-title"),
                html.P(tile["desc"], className="tile-desc"),
            ]),
            html.Div(className="tile-arrow", children="\u2192"),
        ],
    )


# =====================================================================
# KPI / HELPERS (kept for backward compatibility with callbacks)
# =====================================================================

_KPI_ICONS = {
    "Total Cost":  ("$", "icon-navy"),
    "Routes":      ("#", "icon-orange"),
    "Orders":      ("#", "icon-green"),
    "Trucks":      ("T", "icon-blue"),
    "Distance":    ("mi", "icon-purple"),
    "Avg Util":    ("%", "icon-green"),
    "Cost / Mile": ("$/mi", "icon-navy"),
    "HOS Issues":  ("!", "icon-red"),
    "CO2":         ("CO2", "icon-green"),
    "Robustness":  ("R", "icon-blue"),
}


def _kpi_card(title: str, value: str, subtitle: str = "") -> html.Div:
    icon_emoji, icon_cls = _KPI_ICONS.get(title, ("", "icon-navy"))
    return html.Div(className="kpi-card", children=[
        html.Div(className=f"kpi-icon {icon_cls}", children=icon_emoji),
        html.Div(className="kpi-content", children=[
            html.Span(value, className="kpi-value"),
            html.Span(title, className="kpi-label"),
            html.Span(subtitle, className="kpi-sub") if subtitle else None,
        ]),
    ])


def build_kpi_cards(stats: dict, mc_result=None) -> list:
    co2_kg = stats.get("total_co2_kg", 0)
    cards = [
        _kpi_card("Total Cost", f"${stats['total_cost']:,.0f}",
                  f"${stats['per_route']:,.0f} / route"),
        _kpi_card("Routes", str(stats["n_routes"]),
                  f"{stats['n_trucks']} trucks"),
        _kpi_card("Orders", str(stats["n_orders"]),
                  f"{stats['n_stops']} stops"),
        _kpi_card("Distance", f"{stats['total_dist']:,.0f} mi",
                  f"${stats['per_mile']:.2f} / mi"),
        _kpi_card("Avg Util", f"{stats['avg_util']:.0f}%",
                  "fleet capacity"),
        _kpi_card("HOS Issues", str(stats["n_hos"]),
                  "need rest stops"),
        _kpi_card("CO2", f"{co2_kg:,.0f} kg",
                  f"{co2_kg / 1000:.1f} tonnes"),
    ]
    if mc_result is not None:
        cards.append(
            _kpi_card("Robustness", f"{mc_result.robustness_score:.0f}%",
                      f"P95 ${mc_result.cost_p95:,.0f}"),
        )
    return cards


def build_hero_meta(solver_name: str, stats: dict) -> list:
    return [
        html.Span(f"{solver_name.upper()}", className="badge"),
        html.Span(f"{stats['n_orders']} orders across "
                  f"{stats['n_routes']} routes", className="badge"),
        html.Span("LIVE", className="badge status-live"),
    ]


def build_route_table(routes) -> dash_table.DataTable:
    rows = []
    for r in routes:
        drv = r.truck.driver if r.truck else "?"
        tid = r.truck.truck_id if r.truck else "?"
        cap = r.truck.capacity_kg if r.truck else 0
        util = r.total_weight_kg / cap * 100 if cap else 0
        cities = r.city_sequence or [s.city for s in r.stops]
        rows.append({
            "Route": r.route_id,
            "Driver": drv,
            "Truck": tid,
            "Stops": r.num_stops,
            "Weight (kg)": f"{r.total_weight_kg:,.0f}",
            "Utilisation": f"{util:.0f}%",
            "Cities": " > ".join(cities[:5]) + (" ..." if len(cities) > 5 else ""),
        })

    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in
                 ["Route", "Driver", "Truck", "Stops",
                  "Weight (kg)", "Utilisation", "Cities"]],
        data=rows,
        sort_action="native",
        filter_action="native",
        page_size=12,
        style_table={"overflowX": "auto", "borderRadius": "8px",
                     "border": "1px solid #D5D0C5"},
        style_header={
            "backgroundColor": _NAVY,
            "color": _CREAM,
            "fontWeight": "700",
            "fontSize": "11px",
            "padding": "8px 12px",
            "border": "none",
            "textTransform": "uppercase",
            "letterSpacing": "0.5px",
        },
        style_cell={
            "backgroundColor": _WHITE,
            "color": _DTXT,
            "fontSize": "12px",
            "padding": "8px 12px",
            "border": "none",
            "borderBottom": "1px solid #E5E0D5",
            "textAlign": "left",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": _CREAM},
        ],
        style_filter={
            "backgroundColor": _CREAM,
            "fontSize": "11px",
            "padding": "4px 8px",
        },
    )


# =====================================================================
# GRAPH PANEL WRAPPER
# =====================================================================

def _graph_panel(title: str, graph_id: str, subtitle: str = "",
                 full_width: bool = False) -> html.Div:
    cls = "sidebar-panel sidebar-panel--full" if full_width else "sidebar-panel"
    return html.Div(className=cls, children=[
        html.Div(className="sp-header", children=[
            html.H4(title, className="sp-title"),
            html.Span(subtitle, className="sp-subtitle") if subtitle else None,
        ]),
        html.Div(className="sp-body", children=[
            dcc.Graph(id=graph_id, config={"displayModeBar": False}),
        ]),
    ])


# =====================================================================
# SIDEBAR TAB CONTENT
# =====================================================================

def _tab_routes(solver_data, default_solver):
    return html.Div(className="sidebar-tab-content", children=[
        _graph_panel("Fleet Utilisation", "utilisation",
                     "Capacity vs 90% target"),
        _graph_panel("Distance vs Weight", "dist-weight",
                     "Bubble size = total cost"),
        _graph_panel("Route Complexity", "stops-distribution",
                     "Stops per route histogram"),
        _graph_panel("Efficiency Frontier", "efficiency-frontier",
                     "Lower-left = most efficient"),
        _graph_panel("Delivery Cadence", "delivery-timeline",
                     "Cumulative deliveries over time"),
        _graph_panel("Demand Heatmap", "customer-heatmap",
                     "Customer \u00d7 Destination weight matrix",
                     full_width=True),
        html.Div(id="route-table-container", className="sidebar-panel",
                 children=[build_route_table(
                     solver_data[default_solver]["routes"])]),
    ])


def _tab_cost(solver_data, default_solver):
    return html.Div(className="sidebar-tab-content", children=[
        _graph_panel("Cost Waterfall", "cost-waterfall",
                     "Fuel + labour + maintenance breakdown"),
        _graph_panel("Cost by Route", "cost-bar",
                     "Stacked cost per route"),
        _graph_panel("Cost Efficiency by Truck Class", "truck-class-eff",
                     "Cost per ton-mile \u2014 lower is better"),
        _graph_panel("Economies of Distance", "econ-distance",
                     "Cost/mile decreases with route length"),
        _graph_panel("Cost Treemap", "cost-treemap",
                     "Hierarchical cost decomposition"),
        _graph_panel("Cost Sankey", "cost-sankey",
                     "Routes \u2192 cost components \u2192 fleet total",
                     full_width=True),
        _graph_panel("Diesel Sensitivity", "fuel-sensitivity",
                     "What-if: \u00B120% diesel price swing"),
    ])


def _tab_drivers(solver_data, default_solver):
    return html.Div(className="sidebar-tab-content", children=[
        _graph_panel("Driver Radar", "driver-radar",
                     "Normalised 4-dimension comparison"),
        _graph_panel("Performance Matrix", "driver-matrix",
                     "Percentile-ranked KPIs (green = best)"),
        _graph_panel("Timeline (Gantt)", "gantt",
                     "HOS-compliant schedule per driver"),
        _graph_panel("HOS Compliance", "hos-compliance",
                     "Driving vs mandatory rest per route"),
        _graph_panel("Driver-Distance Heatmap", "heatmap",
                     "Mileage intensity by driver & week"),
        _graph_panel("Hub & Spoke Network", "network-radial",
                     "Radial depot-to-destination cargo flow"),
    ])


def _tab_risk(solver_data, default_solver):
    return html.Div(className="sidebar-tab-content", children=[
        _graph_panel("Cost Distribution", "mc-histogram",
                     "1,000 simulated fleet costs with P5/P95"),
        _graph_panel("Sensitivity Tornado", "risk-tornado",
                     "One-at-a-time factor impact"),
        _graph_panel("Robustness Score", "robustness-gauge",
                     "How stable is the solution?"),
        _graph_panel("Fleet KPI Gauges", "fleet-gauges",
                     "Cost \u2022 Util \u2022 CO2 \u2022 Robustness",
                     full_width=True),
    ])


def _tab_sustainability(solver_data, default_solver):
    return html.Div(className="sidebar-tab-content", children=[
        _graph_panel("CO2 by Route", "carbon-route",
                     "Per-route emissions vs fleet average"),
        _graph_panel("Emissions Waterfall", "carbon-waterfall",
                     "CO2 by truck class"),
        _graph_panel("Demand Density Map", "demand-density",
                     "Geographic hotspot analysis"),
    ])


def _tab_strategy(solver_data, default_solver):
    """Pre-computed cross-solver comparison (only if multi-solver)."""
    if len(solver_data) < 2:
        return html.Div(className="sidebar-tab-content", children=[
            html.P("Run with --solver all to compare strategies",
                   style={"color": _SLATE, "padding": "2rem",
                          "textAlign": "center"}),
        ])

    return html.Div(className="sidebar-tab-content", children=[
        html.Div(className="sidebar-panel", children=[
            html.Div(className="sp-header", children=[
                html.H4("Cost Breakdown", className="sp-title"),
            ]),
            html.Div(className="sp-body", children=[
                dcc.Graph(figure=figs.solver_cost_comparison(solver_data),
                          config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="sidebar-panel", children=[
            html.Div(className="sp-header", children=[
                html.H4("Efficiency Radar", className="sp-title"),
            ]),
            html.Div(className="sp-body", children=[
                dcc.Graph(figure=figs.solver_efficiency_radar(solver_data),
                          config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="sidebar-panel", children=[
            html.Div(className="sp-header", children=[
                html.H4("Routes & Distance", className="sp-title"),
            ]),
            html.Div(className="sp-body", children=[
                dcc.Graph(figure=figs.solver_route_comparison(solver_data),
                          config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="sidebar-panel", children=[
            html.Div(className="sp-header", children=[
                html.H4("Risk Violin", className="sp-title"),
            ]),
            html.Div(className="sp-body", children=[
                dcc.Graph(figure=figs.mc_solver_violin(solver_data),
                          config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="sidebar-panel", children=[
            html.Div(className="sp-header", children=[
                html.H4("Cost per Mile", className="sp-title"),
            ]),
            html.Div(className="sp-body", children=[
                dcc.Graph(figure=figs.cost_per_mile_box(solver_data),
                          config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="sidebar-panel", children=[
            html.Div(className="sp-header", children=[
                html.H4("Cost vs CO2 Trade-off", className="sp-title"),
            ]),
            html.Div(className="sp-body", children=[
                dcc.Graph(figure=figs.cost_co2_tradeoff(solver_data),
                          config={"displayModeBar": False}),
            ]),
        ]),
        _graph_panel("Cost vs Makespan", "pareto",
                     "Multi-objective Pareto trade-off"),
        _graph_panel("ALNS Operator Adaptation", "alns-operators",
                     "Destroy & repair selection weights"),
        html.Div(className="sidebar-panel", children=[
            html.Div(className="sp-header", children=[
                html.H4("MC Risk Summary", className="sp-title"),
            ]),
            html.Div(className="sp-body", children=[
                dcc.Graph(figure=figs.solver_mc_summary_table(solver_data),
                          config={"displayModeBar": False}),
            ]),
        ]),
        _build_comparison_table(solver_data),
    ])


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def build_layout(solver_data: dict, default_solver: str) -> html.Div:
    """Build the full SPA layout with URL routing."""
    solver_names = list(solver_data.keys())
    multi = len(solver_names) > 1
    default = solver_data[default_solver]
    stats = default["stats"]
    routes = default["routes"]
    mc_result = default.get("mc_result")

    route_options = [
        {"label": f"Route {r.route_id} - "
                  f"{r.truck.driver if r.truck else '?'} "
                  f"({r.num_stops} stops, {r.total_weight_kg:,.0f} kg)",
         "value": r.route_id}
        for r in routes
    ]
    all_ids = [r.route_id for r in routes]

    # Hidden solver selector for callbacks
    if multi:
        solver_dd = dcc.Dropdown(
            id="solver-selector",
            options=[{"label": s.upper(), "value": s} for s in solver_names],
            value=default_solver,
            clearable=False,
        )
    else:
        solver_dd = dcc.Dropdown(
            id="solver-selector",
            options=[{"label": default_solver, "value": default_solver}],
            value=default_solver,
        )

    # -- LANDING PAGE --
    landing = html.Div(id="page-landing", className="page-landing", children=[
        # Hero banner
        html.Div(className="landing-hero", children=[
            html.Div(className="landing-hero-bg"),
            html.Div(className="landing-hero-content", children=[

                html.H1("Fleet Command Center",
                         className="landing-title"),
                html.P("Capacitated Vehicle Routing \u2022 "
                       "HOS Scheduling \u2022 Monte Carlo Risk \u2022 "
                       f"{len(solver_names)} Solver Strategies",
                       className="landing-subtitle"),
                html.Div(className="landing-stats", children=[
                    html.Div(className="landing-stat", children=[
                        html.Span(f"{stats['n_orders']}", className="stat-num"),
                        html.Span("Orders", className="stat-label"),
                    ]),
                    html.Div(className="landing-stat", children=[
                        html.Span(f"{stats['n_routes']}", className="stat-num"),
                        html.Span("Routes", className="stat-label"),
                    ]),
                    html.Div(className="landing-stat", children=[
                        html.Span(f"{stats['n_trucks']}", className="stat-num"),
                        html.Span("Trucks", className="stat-label"),
                    ]),
                    html.Div(className="landing-stat", children=[
                        html.Span(f"${stats['total_cost']:,.0f}",
                                  className="stat-num"),
                        html.Span("Fleet Cost", className="stat-label"),
                    ]),
                ]),
            ]),
        ]),

        # Tile grid
        html.Div(className="landing-tiles", children=[
            _tile_card(t, stats) for t in _TILES
        ]),

        # Footer
        html.Footer(className="landing-footer", children=[
            html.P([
                "Fleet Optimisation \u2022 Built by ",
                html.Strong("Pedro Paulo da Cruz Mendes (petter.mendes@outlook.com) - AI / Cloud Architect Candidate"),
                " \u2022 ",
                html.Code(" / ".join(s.upper() for s in solver_names)),
            ]),
        ]),
    ])

    # -- DASHBOARD PAGE (map + sidebar) --
    dashboard = html.Div(id="page-dashboard", className="page-dashboard",
                         style={"display": "none"}, children=[
        # Top navbar
        html.Div(className="dash-navbar", children=[
            html.Div(className="dash-nav-left", children=[
                html.Button("\u2190  Back", id="btn-back",
                            className="nav-back-btn", n_clicks=0),

                html.Span("Fleet Command Center",
                           className="nav-app-name"),
            ]),
            html.Div(className="dash-nav-center", children=[
                # Tab buttons (like CropMonitor's Rainfall / Temperature tabs)
                html.Div(className="nav-tabs", children=[
                    html.Button("Routes", id="tab-btn-routes",
                                className="nav-tab active",
                                **{"data-tab": "routes"}, n_clicks=0),
                    html.Button("Cost", id="tab-btn-cost",
                                className="nav-tab",
                                **{"data-tab": "cost"}, n_clicks=0),
                    html.Button("Drivers", id="tab-btn-drivers",
                                className="nav-tab",
                                **{"data-tab": "drivers"}, n_clicks=0),
                    html.Button("Risk", id="tab-btn-risk",
                                className="nav-tab",
                                **{"data-tab": "risk"}, n_clicks=0),
                    html.Button("Carbon", id="tab-btn-sustainability",
                                className="nav-tab",
                                **{"data-tab": "sustainability"}, n_clicks=0),
                    html.Button("Strategy", id="tab-btn-strategy",
                                className="nav-tab",
                                **{"data-tab": "strategy"}, n_clicks=0),
                ]),
            ]),
            html.Div(className="dash-nav-right", children=[
                html.Div(className="nav-solver-wrap", children=[
                    html.Label("Strategy:", className="nav-solver-lbl"),
                    solver_dd,
                ]) if multi else html.Div(
                    style={"display": "none"}, children=[solver_dd],
                ),
            ]),
        ]),

        # Main canvas: map background + sidebar overlay
        html.Div(className="dash-canvas", children=[
            # Full-screen map
            html.Div(className="dash-map", children=[
                dcc.Graph(id="route-map",
                          config={"scrollZoom": True,
                                  "displayModeBar": False},
                          style={"height": "100%", "width": "100%"}),
            ]),

            # Lateral sidebar (CropMonitor-style slide-over)
            html.Div(id="dash-sidebar", className="dash-sidebar", children=[
                # Drag-to-resize handle (left edge)
                html.Div(id="sidebar-resize-handle",
                         className="sidebar-resize-handle"),

                # Sidebar header with KPIs
                html.Div(id="sidebar-kpi", className="sidebar-kpi",
                         children=build_kpi_cards(stats, mc_result)),

                # Sidebar controls
                html.Div(className="sidebar-controls", children=[
                    html.Label("Filter Routes:",
                               className="sidebar-ctrl-label"),
                    dcc.Dropdown(
                        id="route-dropdown",
                        options=route_options,
                        value=all_ids,
                        multi=True,
                        placeholder="Select routes...",
                        className="dropdown sidebar-dropdown",
                    ),
                ]),

                # Sidebar toggle button (right-side sidebar opens left)
                html.Button(
                    "\u276F", id="sidebar-toggle",
                    className="sidebar-toggle", n_clicks=0,
                ),

                # Tab content container (swapped by callback)
                html.Div(id="sidebar-tab-content",
                         className="sidebar-scroll",
                         children=[
                    _tab_routes(solver_data, default_solver),
                ]),
            ]),
        ]),
    ])

    # Hidden route-dropdown for landing page (prevent callback errors)
    hidden = html.Div(style={"display": "none"}, children=[
        dcc.Store(id="active-tab", data="routes"),
        dcc.Store(id="sidebar-state", data="open"),
    ])

    return html.Div(className="app-root", children=[
        dcc.Location(id="url", refresh=False),
        hidden,
        # Hidden stubs for callback IDs that no longer have visible elements
        html.Div(id="kpi-ribbon", style={"display": "none"}),
        html.Div(id="hero-meta", style={"display": "none"}),
        html.Div(id="map-header-subtitle", style={"display": "none"}),
        landing,
        dashboard,
    ])


# =====================================================================
# COMPARISON TABLE (reused by strategy tab)
# =====================================================================

def _build_comparison_table(solver_data: dict) -> html.Div:
    solver_names = list(solver_data.keys())
    metrics = [
        ("Total Cost",            "total_cost",  "${:,.0f}",  True),
        ("Routes",                "n_routes",    "{}",        True),
        ("Orders",                "n_orders",    "{}",        False),
        ("Total Distance (mi)",   "total_dist",  "{:,.0f}",   True),
        ("Avg Utilisation (%)",   "avg_util",    "{:.0f}%",   False),
        ("HOS Issues",            "n_hos",       "{}",        True),
        ("CO2 (kg)",              "total_co2_kg","{:,.0f}",   True),
        ("Cost / Mile",           "per_mile",    "${:.2f}",   True),
        ("Cost / Route",          "per_route",   "${:,.0f}",  True),
    ]

    th_style = {
        "backgroundColor": _NAVY, "color": _CREAM,
        "padding": "8px 12px", "textTransform": "uppercase",
        "fontSize": "10px", "letterSpacing": "0.5px", "fontWeight": "700",
        "borderBottom": f"2px solid {_GOLD}", "textAlign": "center",
    }
    td_style_base = {
        "padding": "6px 12px", "borderBottom": "1px solid #E5E0D5",
        "textAlign": "center", "fontSize": "12px",
    }

    header = html.Tr(
        [html.Th("Metric", style={**th_style, "textAlign": "left"})] +
        [html.Th(s.upper(), style=th_style) for s in solver_names]
    )

    rows = []
    for label, key, fmt, lower_better in metrics:
        vals = [solver_data[s]["stats"].get(key, 0) for s in solver_names]
        best = min(vals) if lower_better else max(vals)
        cells = [html.Td(label, style={**td_style_base,
                                        "fontWeight": "600",
                                        "textAlign": "left"})]
        for v in vals:
            style = dict(td_style_base)
            if v == best and len(set(vals)) > 1:
                style["fontWeight"] = "700"
                style["color"] = _GREEN
            try:
                cells.append(html.Td(fmt.format(v), style=style))
            except (ValueError, TypeError):
                cells.append(html.Td(str(v), style=style))
        rows.append(html.Tr(cells))

    return html.Div(className="sidebar-panel sidebar-panel--full", children=[
        html.Div(className="sp-header", children=[
            html.H4("KPI Comparison", className="sp-title"),
            html.Span("Green = best", className="sp-subtitle"),
        ]),
        html.Div(className="sp-body", style={"padding": "0"}, children=[
            html.Table(
                children=[html.Thead(header), html.Tbody(rows)],
                style={"width": "100%", "borderCollapse": "collapse",
                       "fontSize": "12px"},
            ),
        ]),
    ])


# Expose tab builders for callbacks
TAB_BUILDERS = {
    "routes":         _tab_routes,
    "cost":           _tab_cost,
    "drivers":        _tab_drivers,
    "risk":           _tab_risk,
    "sustainability": _tab_sustainability,
    "strategy":       _tab_strategy,
}

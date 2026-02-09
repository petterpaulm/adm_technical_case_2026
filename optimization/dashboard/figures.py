"""
Plotly figure factories -- corporate dashboard theme.

Every function returns a ``go.Figure`` styled with the corporate brand palette.
Cream background (#FAF7F2), horizontal grid only, left-aligned bold titles.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

import plotly.graph_objects as go

from src.config import CITY_COORDS, WAREHOUSE

# -- Corporate brand palette
_NAVY    = "#00263A"
_GOLD    = "#D4A843"
_GREEN   = "#4A7C59"
_CRIMSON = "#C8102E"
_SKY     = "#5B9BD5"
_WHEAT   = "#E8D5B7"
_SLATE   = "#6B7B8D"
_EARTH   = "#8B6914"
_TEAL    = "#2D8E8E"
_CREAM   = "#FAF7F2"
_LGREY   = "#E5E5E5"
_DTXT    = "#2D2D2D"
_WHITE   = "#FFFFFF"

# ordered colour sequence for multi-series charts
_PALETTE = [_NAVY, _GOLD, _GREEN, _CRIMSON, _SKY, _EARTH, _SLATE, _TEAL]

# Base layout
_LAYOUT = dict(
    font=dict(family="Segoe UI, Helvetica Neue, Arial", size=13, color=_DTXT),
    paper_bgcolor="rgba(0,0,0,0)",   # transparent -- card bg set by CSS
    plot_bgcolor=_CREAM,
    margin=dict(l=60, r=30, t=50, b=50),
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=_NAVY),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0,
        bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=_SLATE),
    ),
    bargap=0.25,
)


def _apply(fig: go.Figure, *, show_grid: bool = True) -> go.Figure:
    """Apply corporate branding to any figure."""
    fig.update_layout(**_LAYOUT)
    y_grid = _LGREY if show_grid else "rgba(0,0,0,0)"
    fig.update_xaxes(
        showgrid=False, linecolor=_NAVY, linewidth=1.5,
        ticks="outside", tickcolor=_NAVY,
        zeroline=False,
        tickfont=dict(color=_SLATE, size=11),
        title_font=dict(color=_NAVY, size=12, family="Segoe UI"),
    )
    fig.update_yaxes(
        showgrid=show_grid, gridcolor=y_grid, gridwidth=0.5,
        linecolor=_NAVY, linewidth=0,
        ticks="", zeroline=False,
        tickfont=dict(color=_SLATE, size=11),
        title_font=dict(color=_NAVY, size=12, family="Segoe UI"),
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# -- 1. Route network map


def route_map(routes, schedules=None) -> go.Figure:
    """Professional route network map with state boundaries, depot marker,
    animated-style route arcs, city labels, and a US flag accent."""
    fig = go.Figure()

    # ---- State boundary choropleth (invisible fill, visible borders) ----
    fig.add_trace(go.Choropleth(
        locationmode="USA-states",
        locations=[
            "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
            "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
            "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
            "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
            "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
        ],
        z=[0]*50,
        colorscale=[[0, "rgba(240,237,228,0.35)"], [1, "rgba(240,237,228,0.35)"]],
        showscale=False,
        marker_line_color="#B8AFA0",
        marker_line_width=1.0,
        hoverinfo="location",
        hoverlabel=dict(bgcolor="white", font_size=11),
    ))

    wh = CITY_COORDS[WAREHOUSE]

    # ---- Route lines with widths proportional to weight ----
    max_wt = max((r.total_weight_kg for r in routes), default=1) or 1

    for i, route in enumerate(routes):
        col = _PALETTE[i % len(_PALETTE)]
        cities = route.city_sequence or [s.city for s in route.stops]
        if not cities:
            continue

        full_path = [WAREHOUSE, *cities, WAREHOUSE]
        lats = [CITY_COORDS.get(c, (0, 0))[0] for c in full_path]
        lons = [CITY_COORDS.get(c, (0, 0))[1] for c in full_path]
        drv = route.truck.driver if route.truck else "?"
        wt = route.total_weight_kg
        line_w = max(1.8, wt / max_wt * 4.5)
        label = f"R{route.route_id} \u2022 {drv}"

        # Shadow line for depth effect
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines",
            line=dict(width=line_w + 2, color="rgba(0,38,58,0.08)"),
            hoverinfo="skip", showlegend=False,
        ))
        # Main route line
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines",
            line=dict(width=line_w, color=col),
            name=label, hoverinfo="skip", showlegend=True,
        ))

        # Stop markers with city abbreviations
        for ci, city in enumerate(cities):
            clat, clon = CITY_COORDS.get(city, (0, 0))
            short = city.split(",")[0][:12]
            fig.add_trace(go.Scattergeo(
                lat=[clat], lon=[clon],
                mode="markers+text",
                marker=dict(
                    size=9, color=col,
                    line=dict(width=1.5, color=_WHITE),
                    symbol="circle",
                ),
                text=[short] if ci == 0 else [],
                textposition="top right",
                textfont=dict(size=8.5, color=_NAVY, family="Segoe UI"),
                hoverinfo="text",
                hovertext=(
                    f"<b>{city}</b><br>"
                    f"Route {route.route_id} \u2022 {drv}<br>"
                    f"Cargo: {wt:,.0f} kg<br>"
                    f"Stop {ci + 1} of {len(cities)}"
                ),
                showlegend=False,
            ))

    # ---- Depot marker (Cincinnati HQ) -- US-flag-inspired tricolour ----
    # Outer ring (navy), middle ring (white), inner ring (crimson)
    fig.add_trace(go.Scattergeo(
        lat=[wh[0]], lon=[wh[1]], mode="markers",
        marker=dict(size=26, color=_NAVY, symbol="diamond",
                    line=dict(width=0)),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scattergeo(
        lat=[wh[0]], lon=[wh[1]], mode="markers",
        marker=dict(size=20, color=_WHITE, symbol="diamond",
                    line=dict(width=0)),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scattergeo(
        lat=[wh[0]], lon=[wh[1]],
        mode="markers+text",
        marker=dict(
            size=14, color=_CRIMSON, symbol="diamond",
            line=dict(width=0),
        ),
        text=["CINCINNATI HQ"], textposition="bottom center",
        textfont=dict(color=_NAVY, size=10, family="Segoe UI"),
        name="Depot", showlegend=False,
        hoverinfo="text",
        hovertext=(
            "<b>Cincinnati, OH</b><br>"
            "Warehouse & Depot<br>"
            "Origin / Destination for all routes"
        ),
    ))

    # ---- Geo styling with proper state boundaries ----
    fig.update_geos(
        scope="usa",
        showland=True, landcolor="#F5F2EB",
        showlakes=True, lakecolor="#D6E6F0",
        showocean=True, oceancolor="#E8EEF2",
        showcountries=True, countrycolor="#8C8478",
        countrywidth=1.5,
        showsubunits=True, subunitcolor="#C0B8AA", subunitwidth=0.8,
        showcoastlines=True, coastlinecolor="#8C8478", coastlinewidth=1.2,
        showrivers=True, rivercolor="#D6E6F0", riverwidth=0.5,
        bgcolor="rgba(0,0,0,0)",
        lonaxis=dict(range=[-128, -66]),
        lataxis=dict(range=[24, 50]),
        projection_type="albers usa",
    )

    # ---- Clean title annotation (top-right) ----
    fig.add_annotation(
        text="<b>US Route Network</b>  |  Fleet Analytics",
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        xanchor="right", yanchor="top",
        showarrow=False,
        font=dict(size=11, color=_NAVY, family="Segoe UI"),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor=_LGREY, borderwidth=1, borderpad=6,
    )

    # ---- Stats annotation (bottom-left) ----
    n_routes = len(routes)
    n_cities = len({c for r in routes for c in (r.city_sequence or [s.city for s in r.stops])})
    fig.add_annotation(
        text=(
            f"<b>{n_routes}</b> routes  \u2022  "
            f"<b>{n_cities}</b> destinations  \u2022  "
            f"Hub: Cincinnati, OH"
        ),
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        xanchor="left", yanchor="bottom",
        showarrow=False,
        font=dict(size=10, color=_SLATE),
        bgcolor="rgba(250,247,242,0.92)",
        bordercolor=_LGREY, borderwidth=1, borderpad=5,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, system-ui, sans-serif", color=_DTXT, size=11),
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor=_LGREY, borderwidth=1,
            font=dict(size=10, color=_NAVY),
            orientation="v",
            yanchor="top", y=0.95, xanchor="left", x=0.01,
            itemsizing="constant",
            tracegroupgap=2,
        ),
        dragmode="pan",
    )
    return fig


# -- 2. Gantt chart


def gantt_chart(schedules) -> go.Figure:
    """Premium Gantt timeline -- driver schedules with HOS compliance.

    Each activity type gets a distinct colour with gradient-like opacity.
    Rest breaks overlay with a striped-pattern feel.  Hovercards show
    city, timing, and duration.  Drivers are grouped and labelled with
    truck + route info for operational clarity.
    """
    act_cfg = {
        "loading":        {"color": _NAVY,    "symbol": "\u25B2", "label": "Loading"},
        "delivery":       {"color": _GREEN,   "symbol": "\u2714", "label": "Delivery"},
        "return":         {"color": _SKY,     "symbol": "\u21A9", "label": "Return"},
        "mandatory_rest": {"color": _CRIMSON, "symbol": "\u23F8", "label": "HOS Rest"},
    }
    fig = go.Figure()

    # Sort schedules by departure for visual consistency
    sorted_scheds = sorted(schedules, key=lambda s: s.departure)

    # -- Build shapes (rounded bars) and hover traces per driver ---
    for idx, sc in enumerate(sorted_scheds):
        drv_lbl = f"{sc.driver}  \u2022  T{sc.truck_id}  \u2022  R{sc.route_id}"
        alt_shade = _hex_to_rgba(_NAVY, 0.025) if idx % 2 == 0 else "rgba(0,0,0,0)"

        # Background band for row (alternating subtle shading)
        fig.add_hrect(
            y0=idx - 0.42, y1=idx + 0.42,
            fillcolor=alt_shade, line_width=0,
            layer="below",
        )

        for ev in sc.events:
            cfg = act_cfg.get(ev.activity, {"color": _SLATE, "symbol": "", "label": ev.activity})
            dur_hrs = (ev.depart - ev.arrive).total_seconds() / 3600
            fig.add_trace(go.Bar(
                y=[drv_lbl], x=[(ev.depart - ev.arrive).total_seconds() * 1000],
                base=[ev.arrive.isoformat()],
                orientation="h",
                marker=dict(
                    color=_hex_to_rgba(cfg["color"], 0.78),
                    line=dict(width=1.2, color=cfg["color"]),
                    cornerradius=4,
                ),
                width=0.55,
                name=cfg["label"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{cfg['symbol']} {ev.city}</b><br>"
                    f"<span style='color:{cfg['color']}'>\u2588</span> "
                    f"{cfg['label']}<br>"
                    f"{ev.arrive.strftime('%b %d, %H:%M')} \u2192 "
                    f"{ev.depart.strftime('%H:%M')}<br>"
                    f"Duration: <b>{dur_hrs:.1f} h</b><br>"
                    f"Driver: {sc.driver} &bull; Truck {sc.truck_id}"
                    "<extra></extra>"
                ),
            ))

        # Rest breaks -- distinctive dashed pattern
        for rest in sc.rest_breaks:
            dur_hrs = (rest["end"] - rest["start"]).total_seconds() / 3600
            fig.add_trace(go.Bar(
                y=[drv_lbl],
                x=[(rest["end"] - rest["start"]).total_seconds() * 1000],
                base=[rest["start"].isoformat()],
                orientation="h",
                marker=dict(
                    color=_hex_to_rgba(_CRIMSON, 0.25),
                    line=dict(width=1.5, color=_CRIMSON),
                    cornerradius=4,
                    pattern=dict(shape="/", fgcolor=_hex_to_rgba(_CRIMSON, 0.45),
                                 size=6, solidity=0.4),
                ),
                width=0.55,
                name="HOS Rest", showlegend=False,
                hovertemplate=(
                    "<b>\u23F8 Mandatory HOS Rest</b><br>"
                    f"{rest['start'].strftime('%b %d, %H:%M')} \u2192 "
                    f"{rest['end'].strftime('%H:%M')}<br>"
                    f"Duration: <b>{dur_hrs:.0f} h</b><br>"
                    f"Driver: {sc.driver}"
                    "<extra></extra>"
                ),
            ))

    # -- Legend traces (invisible points for a clean legend) ---
    for act, cfg in act_cfg.items():
        fig.add_trace(go.Bar(
            y=[None], x=[None],
            marker=dict(color=cfg["color"], cornerradius=3),
            name=f"{cfg['symbol']} {cfg['label']}",
            showlegend=True,
        ))

    # -- Layout polish ---
    n_drivers = len(sorted_scheds)
    fig.update_layout(
        barmode="stack",
        height=max(420, n_drivers * 58 + 80),
        xaxis=dict(
            type="date", title="",
            gridcolor=_hex_to_rgba(_NAVY, 0.06),
            gridwidth=0.5,
            dtick=86_400_000,  # one day ticks
            tickformat="%b %d\n%a",
            tickfont=dict(size=10, color=_SLATE),
            side="top",
            showline=True, linecolor=_LGREY, linewidth=1,
        ),
        yaxis=dict(
            title="", autorange="reversed",
            tickfont=dict(size=10.5, color=_NAVY, family="Segoe UI"),
            gridcolor="rgba(0,0,0,0)",
            dividercolor=_LGREY,
            showdividers=True,
        ),
        legend=dict(
            orientation="h", y=-0.08, x=0.5, xanchor="center",
            bgcolor="rgba(250,247,242,0.9)",
            bordercolor=_LGREY, borderwidth=1,
            font=dict(size=10.5, color=_NAVY),
            itemwidth=30,
        ),
        bargap=0.08,
    )
    return _apply(fig)


# -- 3. Cost waterfall


def cost_waterfall(costs) -> go.Figure:
    fuel = sum(c["fuel_cost"] for c in costs)
    labour = sum(c["labour_cost"] for c in costs)
    maint = sum(c["maint_cost"] for c in costs)
    total = fuel + labour + maint

    fig = go.Figure(go.Waterfall(
        x=["Fuel", "Labour", "Maintenance", "Total"],
        y=[fuel, labour, maint, total],
        measure=["relative", "relative", "relative", "total"],
        connector=dict(line=dict(color=_GOLD, width=1.5, dash="dot")),
        decreasing=dict(marker=dict(color=_SKY)),
        increasing=dict(marker=dict(color=_NAVY)),
        totals=dict(marker=dict(color=_GOLD,
                                line=dict(color=_NAVY, width=1.5))),
        texttemplate="$%{y:,.0f}",
        textposition="outside",
        textfont=dict(size=12, color=_NAVY),
        hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(height=370, yaxis_title="Cost ($)", showlegend=False)
    return _apply(fig)


# -- 4. Stacked cost bar


def cost_breakdown_bar(routes, costs) -> go.Figure:
    labels = [f"R{r.route_id}" for r in routes]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Fuel", x=labels,
                         y=[c["fuel_cost"] for c in costs],
                         marker_color=_NAVY))
    fig.add_trace(go.Bar(name="Labour", x=labels,
                         y=[c["labour_cost"] for c in costs],
                         marker_color=_GOLD))
    fig.add_trace(go.Bar(name="Maintenance", x=labels,
                         y=[c["maint_cost"] for c in costs],
                         marker_color=_GREEN))
    fig.update_layout(
        barmode="stack", xaxis_title="Route",
        yaxis_title="Cost ($)", height=370, bargap=0.18,
    )
    return _apply(fig)


# -- 5. Fleet utilisation


def utilisation_bars(routes) -> go.Figure:
    sorted_routes = sorted(routes, key=lambda r: r.total_weight_kg, reverse=True)
    labels, pcts, colors = [], [], []

    for r in sorted_routes:
        cap = r.truck.capacity_kg if r.truck else 10_000
        pct = min(r.total_weight_kg / cap * 100, 100) if cap else 0
        drv = r.truck.driver[:8] if r.truck else "?"
        labels.append(f"R{r.route_id} ({drv})")
        pcts.append(pct)
        colors.append(_GREEN if pct >= 90 else _NAVY if pct >= 70
                      else _GOLD if pct >= 50 else _SLATE)

    fig = go.Figure(go.Bar(
        y=labels, x=pcts, orientation="h",
        marker=dict(color=colors, cornerradius=3,
                    line=dict(width=0.5, color=_CREAM)),
        text=[f"{p:.0f}%" for p in pcts],
        textposition="inside",
        textfont=dict(size=10, color=_WHITE),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% capacity<extra></extra>",
    ))
    fig.add_vline(x=90, line_dash="dot", line_color=_CRIMSON, line_width=1.5,
                  annotation_text="90 % target",
                  annotation_font_color=_CRIMSON, annotation_font_size=10)
    fig.update_layout(
        height=max(340, len(routes) * 26),
        xaxis=dict(title="Capacity Utilisation (%)", range=[0, 105]),
        yaxis=dict(title="", autorange="reversed"),
    )
    return _apply(fig)


# -- 6. Distance vs Weight scatter


def dist_weight_scatter(routes, costs) -> go.Figure:
    dists = [c["distance_mi"] for c in costs]
    weights = [r.total_weight_kg for r in routes]
    tcosts = [c["total_cost"] for c in costs]
    drivers = [r.truck.driver if r.truck else "?" for r in routes]

    fig = go.Figure(go.Scatter(
        x=dists, y=weights, mode="markers",
        marker=dict(
            size=[max(12, t / 70) for t in tcosts],
            color=tcosts,
            colorscale=[
                [0, _WHEAT], [0.35, _GOLD],
                [0.65, _NAVY], [1.0, _CRIMSON],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="Cost ($)", font=dict(size=10, color=_SLATE)),
                tickfont=dict(size=9, color=_SLATE),
                thickness=12, len=0.6, borderwidth=0,
            ),
            line=dict(width=1, color="rgba(0,38,58,0.2)"),
            opacity=0.88,
        ),
        text=[
            f"<b>Route {r.route_id}</b><br>{d}<br>"
            f"{dists[i]:,.0f} mi - {weights[i]:,.0f} kg<br>${tcosts[i]:,.0f}"
            for i, (r, d) in enumerate(zip(routes, drivers, strict=False))
        ],
        hoverinfo="text",
    ))
    fig.update_layout(
        xaxis_title="Distance (mi)", yaxis_title="Cargo Weight (kg)", height=380,
    )
    return _apply(fig)


# -- 7. Driver workload radar


def driver_radar(routes, costs) -> go.Figure:
    driver_stats: dict[str, dict] = defaultdict(
        lambda: {"routes": 0, "distance": 0.0, "weight": 0.0, "cost": 0.0}
    )
    for r, c in zip(routes, costs, strict=False):
        drv = r.truck.driver if r.truck else "Unknown"
        driver_stats[drv]["routes"] += 1
        driver_stats[drv]["distance"] += c["distance_mi"]
        driver_stats[drv]["weight"] += r.total_weight_kg
        driver_stats[drv]["cost"] += c["total_cost"]

    categories = ["Routes", "Distance", "Weight", "Cost"]
    fig = go.Figure()

    max_r = max((d["routes"] for d in driver_stats.values()), default=1) or 1
    max_d = max((d["distance"] for d in driver_stats.values()), default=1) or 1
    max_w = max((d["weight"] for d in driver_stats.values()), default=1) or 1
    max_c = max((d["cost"] for d in driver_stats.values()), default=1) or 1

    for i, (drv, st) in enumerate(sorted(driver_stats.items())):
        vals = [
            st["routes"] / max_r * 100,
            st["distance"] / max_d * 100,
            st["weight"] / max_w * 100,
            st["cost"] / max_c * 100,
        ]
        vals.append(vals[0])
        col = _PALETTE[i % len(_PALETTE)]

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=[*categories, categories[0]],
            fill="toself",
            fillcolor=_hex_to_rgba(col, 0.12),
            line=dict(color=col, width=2),
            name=drv,
            hovertemplate=(
                f"<b>{drv}</b><br>"
                f"Routes: {st['routes']}<br>"
                f"Distance: {st['distance']:,.0f} mi<br>"
                f"Weight: {st['weight']:,.0f} kg<br>"
                f"Cost: ${st['cost']:,.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=_CREAM,
            radialaxis=dict(
                visible=True, range=[0, 110],
                gridcolor=_LGREY, linecolor="rgba(0,0,0,0)",
                tickfont=dict(size=8, color=_SLATE),
            ),
            angularaxis=dict(
                gridcolor=_LGREY, linecolor=_LGREY,
                tickfont=dict(size=10, color=_NAVY),
            ),
        ),
        height=400,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    return _apply(fig, show_grid=False)


# -- 8. Cost treemap


def cost_treemap(routes, costs) -> go.Figure:
    ids, labels, parents, values, colors = [], [], [], [], []

    fuel_total = sum(c["fuel_cost"] for c in costs)
    lab_total = sum(c["labour_cost"] for c in costs)
    maint_total = sum(c["maint_cost"] for c in costs)
    fleet_total = fuel_total + lab_total + maint_total

    ids.append("Fleet")
    labels.append(f"Fleet Total<br>${fleet_total:,.0f}")
    parents.append("")
    values.append(fleet_total)
    colors.append(_CREAM)

    for cat, val, col in [("Fuel", fuel_total, _NAVY),
                          ("Labour", lab_total, _GOLD),
                          ("Maintenance", maint_total, _GREEN)]:
        ids.append(cat)
        labels.append(f"{cat}<br>${val:,.0f}")
        parents.append("Fleet")
        values.append(val)
        colors.append(col)

    for r, c in zip(routes, costs, strict=False):
        rid = f"R{r.route_id}"
        drv = r.truck.driver if r.truck else "?"
        for cat, key, col in [("Fuel", "fuel_cost", "#003D5C"),
                              ("Labour", "labour_cost", _EARTH),
                              ("Maintenance", "maint_cost", "#3A6347")]:
            uid = f"{cat}-{rid}"
            ids.append(uid)
            labels.append(f"{rid} ({drv})<br>${c[key]:,.0f}")
            parents.append(cat)
            values.append(c[key])
            colors.append(col)

    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents,
        values=values,
        marker=dict(colors=colors, line=dict(width=1.5, color=_CREAM)),
        branchvalues="total",
        textfont=dict(size=11, color=_WHITE),
        hovertemplate="<b>%{label}</b><extra></extra>",
        pathbar=dict(textfont=dict(size=11, color=_SLATE)),
    ))
    fig.update_layout(height=420, margin=dict(l=5, r=5, t=30, b=5))
    return _apply(fig, show_grid=False)


# -- 9. Driver distance heatmap


def driver_distance_heatmap(routes, costs, schedules) -> go.Figure:
    grid: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    sched_map = {sc.route_id: sc for sc in schedules}

    for r, c in zip(routes, costs, strict=False):
        drv = r.truck.driver if r.truck else "?"
        sc = sched_map.get(r.route_id)
        wk = sc.departure.strftime("Wk %U") if sc else "Wk ??"
        grid[drv][wk] += c["distance_mi"]

    drivers_sorted = sorted(grid.keys())
    weeks_sorted = sorted({w for d in grid.values() for w in d})
    z = [[grid[drv].get(wk, 0) for wk in weeks_sorted] for drv in drivers_sorted]

    fig = go.Figure(go.Heatmap(
        z=z, x=weeks_sorted, y=drivers_sorted,
        colorscale=[
            [0, _CREAM], [0.25, _WHEAT],
            [0.5, _GOLD], [0.75, _NAVY], [1.0, _CRIMSON],
        ],
        colorbar=dict(
            title=dict(text="Miles", font=dict(size=10, color=_SLATE)),
            tickfont=dict(size=9, color=_SLATE),
            thickness=12, len=0.6, borderwidth=0,
        ),
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:,.0f} miles<extra></extra>",
        xgap=3, ygap=3,
    ))
    fig.update_layout(
        height=max(280, len(drivers_sorted) * 45),
        xaxis=dict(title="", side="top"),
        yaxis=dict(title="", autorange="reversed"),
    )
    return _apply(fig, show_grid=False)


# -- 10. Pareto front


def pareto_scatter(metadata: dict, solver_data: dict | None = None) -> go.Figure:
    """Pareto front chart.

    In multi-solver mode (solver_data provided with 2+ solvers) this plots
    each solver as a point on the cost-vs-makespan plane so the user can
    visually compare trade-offs.

    In single-solver mode it uses the 'pareto_front' key from metadata
    (populated by --solver pareto).
    """

    # -- multi-solver comparison mode ------------------------------------
    if solver_data and len(solver_data) >= 2:
        names, costs_pts, makespan_pts, routes_pts = [], [], [], []
        for sname, sdata in solver_data.items():
            st = sdata.get("stats", {})
            if not st:
                continue
            names.append(sname.upper())
            costs_pts.append(st["total_cost"])
            # makespan = max schedule span in hours
            scheds = sdata.get("schedules", [])
            if scheds:
                first = min(s.departure for s in scheds)
                last = max(s.arrival for s in scheds)
                span_hrs = (last - first).total_seconds() / 3600
            else:
                span_hrs = 0
            makespan_pts.append(round(span_hrs, 1))
            routes_pts.append(st.get("n_routes", 0))

        palette = [_GREEN, _GOLD, _NAVY, _CRIMSON, _SKY, _TEAL]
        marker_colors = [palette[i % len(palette)] for i in range(len(names))]

        fig = go.Figure(go.Scatter(
            x=costs_pts, y=makespan_pts,
            mode="markers+text",
            text=names,
            textposition="top center",
            textfont=dict(size=11, color=_NAVY, family="Georgia"),
            marker=dict(
                size=[max(16, r * 0.6) for r in routes_pts],
                color=marker_colors,
                line=dict(width=1.5, color=_NAVY),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Cost: $%{x:,.0f}<br>"
                "Makespan: %{y:,.0f} hrs<br>"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            xaxis_title="Total Fleet Cost ($)",
            yaxis_title="Makespan (hours)",
            height=380,
        )
        return _apply(fig)

    # -- single-solver Pareto front from metadata ------------------------
    front = metadata.get("pareto_front", [])
    if not front:
        fig = go.Figure()
        fig.add_annotation(
            text="Run with --solver pareto to generate the Pareto front",
            showarrow=False, font=dict(size=14, color=_SLATE),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
        fig.update_layout(height=380)
        return _apply(fig, show_grid=False)

    cs = [p["cost"] for p in front]
    ms = [p["makespan_hrs"] for p in front]
    us = [p.get("util_std", 0) for p in front]

    fig = go.Figure(go.Scatter(
        x=cs, y=ms, mode="markers+lines",
        marker=dict(
            size=[max(10, u * 180) for u in us],
            color=cs,
            colorscale=[[0, _GREEN], [0.5, _NAVY], [1.0, _CRIMSON]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Cost", font=dict(size=10, color=_SLATE)),
                tickfont=dict(size=9, color=_SLATE),
                thickness=12, borderwidth=0,
            ),
            line=dict(width=1, color="rgba(0,38,58,0.2)"),
        ),
        line=dict(width=1.5, dash="dot", color=_NAVY),
        hovertemplate=(
            "<b>Solution</b><br>Cost: $%{x:,.0f}<br>"
            "Makespan: %{y:.1f} hrs<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title="Total Cost ($)", yaxis_title="Makespan (hours)", height=380,
    )
    return _apply(fig)


# -- Summary stats


def compute_summary_stats(routes, costs, schedules) -> dict:
    total_cost = sum(c["total_cost"] for c in costs)
    total_dist = sum(c["distance_mi"] for c in costs)
    total_fuel = sum(c["fuel_cost"] for c in costs)
    total_labour = sum(c["labour_cost"] for c in costs)
    total_maint = sum(c["maint_cost"] for c in costs)
    total_co2 = sum(c["co2_kg"] for c in costs)
    n_routes = len(routes)
    n_orders = sum(len(r.all_orders) for r in routes)
    n_stops = sum(r.num_stops for r in routes)
    n_trucks = len({r.truck.truck_id for r in routes if r.truck})
    n_hos = sum(1 for s in schedules if s.needs_rest)

    utilisation_pcts = []
    for r in routes:
        cap = r.truck.capacity_kg if r.truck else 10_000
        utilisation_pcts.append(min(r.total_weight_kg / cap * 100, 100) if cap else 0)
    avg_util = statistics.mean(utilisation_pcts) if utilisation_pcts else 0

    return {
        "total_cost": total_cost,
        "total_dist": total_dist,
        "total_fuel": total_fuel,
        "total_labour": total_labour,
        "total_maint": total_maint,
        "total_co2_kg": total_co2,
        "n_routes": n_routes,
        "n_orders": n_orders,
        "n_stops": n_stops,
        "n_trucks": n_trucks,
        "n_hos": n_hos,
        "avg_util": avg_util,
        "per_mile": total_cost / total_dist if total_dist else 0,
        "per_route": total_cost / n_routes if n_routes else 0,
    }


# -- 11. Carbon emissions by route


def carbon_by_route(routes, costs) -> go.Figure:
    """Horizontal bar chart -- CO2 per route, sorted descending."""
    data = sorted(
        [(f"R{r.route_id}", c["co2_kg"]) for r, c in zip(routes, costs)],
        key=lambda x: x[1],
    )
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    avg = statistics.mean(vals) if vals else 0

    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h",
        marker=dict(
            color=[_TEAL if v <= avg else _CRIMSON for v in vals],
            cornerradius=3,
            line=dict(width=0.5, color=_CREAM),
        ),
        text=[f"{v:,.0f} kg" for v in vals],
        textposition="inside",
        textfont=dict(size=10, color=_WHITE),
        hovertemplate="<b>%{y}</b><br>%{x:,.0f} kg CO2<extra></extra>",
    ))
    fig.add_vline(
        x=avg, line_dash="dot", line_color=_NAVY, line_width=1.5,
        annotation_text=f"Avg {avg:,.0f} kg",
        annotation_font_color=_NAVY, annotation_font_size=10,
    )
    fig.update_layout(
        height=max(340, len(routes) * 26),
        xaxis=dict(title="CO2 Emissions (kg)"),
        yaxis=dict(title="", autorange="reversed"),
    )
    return _apply(fig)


# -- 12. Carbon waterfall


def carbon_waterfall(routes, costs) -> go.Figure:
    """Waterfall -- fleet CO2 split by truck class."""
    class_co2: dict[str, float] = defaultdict(float)
    for r, c in zip(routes, costs):
        lbl = f"{r.truck.length_m:.1f} m" if r.truck else "Unknown"
        class_co2[lbl] += c["co2_kg"]

    labels = [*class_co2.keys(), "Fleet Total"]
    values = [*class_co2.values(), sum(class_co2.values())]
    measures = ["relative"] * len(class_co2) + ["total"]

    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures,
        connector=dict(line=dict(color=_GOLD, width=1.5, dash="dot")),
        decreasing=dict(marker=dict(color=_TEAL)),
        increasing=dict(marker=dict(color=_CRIMSON)),
        totals=dict(marker=dict(color=_GREEN,
                                line=dict(color=_NAVY, width=1.5))),
        texttemplate="%{y:,.0f} kg",
        textposition="outside",
        textfont=dict(size=11, color=_NAVY),
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} kg CO2<extra></extra>",
    ))
    fig.update_layout(height=370, yaxis_title="CO2 (kg)", showlegend=False)
    return _apply(fig)


# -- 13. Monte Carlo cost histogram


def mc_cost_histogram(mc_result) -> go.Figure:
    """Distribution of total fleet cost across Monte Carlo trials."""
    if mc_result is None:
        fig = go.Figure()
        fig.add_annotation(
            text="<b>No Monte Carlo Data</b>",
            showarrow=False, font=dict(size=20, color=_NAVY),
            xref="paper", yref="paper", x=0.5, y=0.55,
        )
        fig.add_annotation(
            text="Re-run with  --monte-carlo  flag to generate risk analysis",
            showarrow=False, font=dict(size=12, color=_SLATE),
            xref="paper", yref="paper", x=0.5, y=0.40,
        )
        fig.update_layout(
            height=380,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return _apply(fig, show_grid=False)

    trials = mc_result.trial_costs
    det = mc_result.deterministic_cost
    p5, p95 = mc_result.cost_p5, mc_result.cost_p95

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=trials, nbinsx=50,
        marker=dict(color=_hex_to_rgba(_NAVY, 0.55),
                    line=dict(color=_NAVY, width=0.5)),
        hovertemplate="$%{x:,.0f}<br>%{y} trials<extra></extra>",
        name="Simulated cost",
    ))

    fig.add_vline(x=det, line_color=_GREEN, line_width=2, line_dash="solid",
                  annotation_text=f"Deterministic ${det:,.0f}",
                  annotation_font=dict(color=_GREEN, size=10))
    fig.add_vline(x=p5, line_color=_SKY, line_width=1.5, line_dash="dash",
                  annotation_text=f"P5 ${p5:,.0f}",
                  annotation_font=dict(color=_SKY, size=10))
    fig.add_vline(x=p95, line_color=_CRIMSON, line_width=1.5, line_dash="dash",
                  annotation_text=f"P95 ${p95:,.0f}",
                  annotation_font=dict(color=_CRIMSON, size=10))

    fig.update_layout(
        height=380,
        xaxis_title="Total Fleet Cost ($)",
        yaxis_title="Frequency",
        showlegend=False,
        bargap=0.03,
    )
    return _apply(fig)


# -- 14. Risk tornado chart


def risk_tornado(mc_result) -> go.Figure:
    """Tornado -- cost sensitivity to each risk factor."""
    if mc_result is None or not mc_result.sensitivity:
        fig = go.Figure()
        fig.add_annotation(
            text="<b>No Sensitivity Data</b>",
            showarrow=False, font=dict(size=20, color=_NAVY),
            xref="paper", yref="paper", x=0.5, y=0.55,
        )
        fig.add_annotation(
            text="Re-run with  --monte-carlo  flag to generate sensitivity analysis",
            showarrow=False, font=dict(size=12, color=_SLATE),
            xref="paper", yref="paper", x=0.5, y=0.40,
        )
        fig.update_layout(
            height=380,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return _apply(fig, show_grid=False)

    det = mc_result.deterministic_cost
    sens = mc_result.sensitivity

    sorted_factors = sorted(sens.keys(), key=lambda k: sens[k][1] - sens[k][0])

    labels, lows, highs = [], [], []
    for factor in sorted_factors:
        low, high = sens[factor]
        labels.append(factor)
        lows.append(low - det)
        highs.append(high - det)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=lows, orientation="h",
        marker_color=_TEAL, name="Low scenario",
        text=[f"${v:+,.0f}" for v in lows],
        textposition="inside",
        textfont=dict(size=10, color=_WHITE),
        hovertemplate="<b>%{y}</b><br>Low: $%{x:+,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=highs, orientation="h",
        marker_color=_CRIMSON, name="High scenario",
        text=[f"${v:+,.0f}" for v in highs],
        textposition="inside",
        textfont=dict(size=10, color=_WHITE),
        hovertemplate="<b>%{y}</b><br>High: $%{x:+,.0f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=_NAVY, line_width=1.5)

    fig.update_layout(
        barmode="overlay",
        height=max(300, len(labels) * 60),
        xaxis_title="Cost Deviation from Deterministic ($)",
        yaxis_title="",
    )
    return _apply(fig)


# -- 15. MC risk summary gauge


def robustness_gauge(mc_result) -> go.Figure:
    """Gauge chart -- solution robustness score."""
    if mc_result is None:
        fig = go.Figure()
        fig.add_annotation(
            text="<b>No Robustness Data</b>",
            showarrow=False, font=dict(size=20, color=_NAVY),
            xref="paper", yref="paper", x=0.5, y=0.55,
        )
        fig.add_annotation(
            text="Re-run with  --monte-carlo  flag",
            showarrow=False, font=dict(size=12, color=_SLATE),
            xref="paper", yref="paper", x=0.5, y=0.38,
        )
        fig.update_layout(
            height=280,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return _apply(fig, show_grid=False)

    score = mc_result.robustness_score
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number=dict(suffix="%", font=dict(size=36, color=_NAVY)),
        delta=dict(reference=80, valueformat=".0f", prefix="vs target: "),
        gauge=dict(
            axis=dict(range=[0, 100],
                      tickfont=dict(size=10, color=_SLATE)),
            bar=dict(color=_GREEN if score >= 80
                     else _GOLD if score >= 60
                     else _CRIMSON),
            bgcolor=_CREAM,
            borderwidth=1,
            bordercolor=_LGREY,
            steps=[
                dict(range=[0, 60], color=_hex_to_rgba(_CRIMSON, 0.15)),
                dict(range=[60, 80], color=_hex_to_rgba(_GOLD, 0.18)),
                dict(range=[80, 100], color=_hex_to_rgba(_GREEN, 0.15)),
            ],
            threshold=dict(
                line=dict(color=_NAVY, width=2),
                thickness=0.8, value=80,
            ),
        ),
        title=dict(text="Solution Robustness",
                   font=dict(size=14, color=_SLATE)),
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=20))
    return _apply(fig, show_grid=False)


# CROSS-SOLVER COMPARISON CHARTS
# These figures accept the full ``solver_data`` dict and are rendered
# once at layout time (no callbacks needed).


def solver_cost_comparison(solver_data: dict) -> go.Figure:
    """Grouped bar chart -- fuel / labour / maintenance by solver strategy."""
    names = [s.upper() for s in solver_data]
    fuel   = [solver_data[s]["stats"]["total_fuel"]   for s in solver_data]
    labour = [solver_data[s]["stats"]["total_labour"] for s in solver_data]
    maint  = [solver_data[s]["stats"]["total_maint"]  for s in solver_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Fuel", x=names, y=fuel,
        marker=dict(color=_CRIMSON, cornerradius=4),
        text=[f"${v:,.0f}" for v in fuel], textposition="outside",
        textfont=dict(size=10),
    ))
    fig.add_trace(go.Bar(
        name="Labour", x=names, y=labour,
        marker=dict(color=_GOLD, cornerradius=4),
        text=[f"${v:,.0f}" for v in labour], textposition="outside",
        textfont=dict(size=10),
    ))
    fig.add_trace(go.Bar(
        name="Maintenance", x=names, y=maint,
        marker=dict(color=_GREEN, cornerradius=4),
        text=[f"${v:,.0f}" for v in maint], textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Cost Components by Strategy</b>", x=0, font_size=16),
        yaxis_title="Cost ($)",
        yaxis_tickprefix="$",
        height=380,
    )
    return _apply(fig)


def solver_efficiency_radar(solver_data: dict) -> go.Figure:
    """Radar / spider chart -- normalised metrics per solver (lower = better)."""
    categories = ["Cost", "Distance", "Routes", "Wasted Capacity", "CO2"]

    # Collect raw values (invert utilisation so lower = better)
    raw: dict[str, list[float]] = {}
    for s in solver_data:
        st = solver_data[s]["stats"]
        raw[s] = [
            st["total_cost"],
            st["total_dist"],
            st["n_routes"],
            100 - st["avg_util"],       # wasted capacity %
            st["total_co2_kg"],
        ]

    # Normalise 0..1 (0 = best, 1 = worst)
    all_vals = list(raw.values())
    n_cats = len(categories)
    mins = [min(v[i] for v in all_vals) for i in range(n_cats)]
    maxs = [max(v[i] for v in all_vals) for i in range(n_cats)]

    fig = go.Figure()
    for idx, (name, vals) in enumerate(raw.items()):
        norm = []
        for j in range(n_cats):
            rng = maxs[j] - mins[j]
            norm.append((vals[j] - mins[j]) / rng if rng > 0 else 0.5)
        norm.append(norm[0])  # close polygon

        col = _PALETTE[idx % len(_PALETTE)]
        fig.add_trace(go.Scatterpolar(
            r=norm,
            theta=categories + [categories[0]],
            name=name.upper(),
            fill="toself",
            fillcolor=_hex_to_rgba(col, 0.18),
            line=dict(color=col, width=2.5),
            marker=dict(size=5, color=col),
        ))

    fig.update_layout(
        title=dict(text="<b>Efficiency Comparison</b>", x=0, font_size=16),
        polar=dict(
            bgcolor=_CREAM,
            radialaxis=dict(
                visible=True, range=[0, 1.05],
                showticklabels=False, gridcolor=_LGREY,
            ),
            angularaxis=dict(gridcolor=_LGREY, linecolor=_NAVY),
        ),
        height=380,
    )
    return _apply(fig, show_grid=False)


def solver_route_comparison(solver_data: dict) -> go.Figure:
    """Dual-metric bar chart -- route count (left axis) & total distance
    (right axis) by solver."""
    names   = [s.upper() for s in solver_data]
    n_routes = [solver_data[s]["stats"]["n_routes"]  for s in solver_data]
    tot_dist = [solver_data[s]["stats"]["total_dist"] for s in solver_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Routes", x=names, y=n_routes,
        marker=dict(color=_NAVY, cornerradius=4),
        text=[str(v) for v in n_routes], textposition="outside",
        textfont=dict(size=11, color=_NAVY),
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        name="Distance (mi)", x=names, y=tot_dist,
        mode="lines+markers+text",
        line=dict(color=_CRIMSON, width=2.5),
        marker=dict(size=10, color=_CRIMSON, symbol="diamond"),
        text=[f"{v:,.0f}" for v in tot_dist],
        textposition="top center",
        textfont=dict(size=10, color=_CRIMSON),
        yaxis="y2",
    ))
    fig.update_layout(
        title=dict(text="<b>Routes & Total Distance</b>", x=0, font_size=16),
        barmode="group",
        yaxis=dict(title="Number of Routes", side="left",
                   showgrid=True, gridcolor=_LGREY),
        yaxis2=dict(title="Total Distance (mi)", side="right",
                    overlaying="y", showgrid=False),
        height=380,
        legend=dict(orientation="h", y=1.12),
    )
    return _apply(fig)


# =====================================================================
# ADVANCED ANALYTICS -- "Winner Presentation" charts
# =====================================================================


def cost_sankey(routes, costs) -> go.Figure:
    """Sankey diagram -- money flow from each truck/driver through cost
    components (fuel, labour, maintenance) to the fleet total.

    Left nodes  = Routes (driver labels)
    Middle nodes = Cost categories
    Right node  = Fleet Total
    """
    route_labels = [
        f"R{r.route_id} ({r.truck.driver.split()[0] if r.truck else '?'})"
        for r in routes
    ]
    cat_labels = ["Fuel", "Labour", "Maintenance"]
    total_label = "Fleet Total"

    # node list: routes + categories + total
    labels = route_labels + cat_labels + total_label.split("|")
    n_routes = len(routes)
    fuel_idx = n_routes
    lab_idx = n_routes + 1
    maint_idx = n_routes + 2
    total_idx = n_routes + 3

    # Route colours from palette
    node_colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_routes)]
    node_colors += [_NAVY, _GOLD, _GREEN, _CRIMSON]

    # Links: route -> each cost category, then categories -> total
    sources, targets, values, link_colors = [], [], [], []
    for i, c in enumerate(costs):
        for cat_offset, key, col in [
            (0, "fuel_cost", _hex_to_rgba(_NAVY, 0.35)),
            (1, "labour_cost", _hex_to_rgba(_GOLD, 0.35)),
            (2, "maint_cost", _hex_to_rgba(_GREEN, 0.35)),
        ]:
            sources.append(i)
            targets.append(n_routes + cat_offset)
            values.append(c[key])
            link_colors.append(col)

    # Category -> Fleet Total
    fuel_total = sum(c["fuel_cost"] for c in costs)
    lab_total = sum(c["labour_cost"] for c in costs)
    maint_total = sum(c["maint_cost"] for c in costs)
    for cat_off, val, col in [
        (0, fuel_total, _hex_to_rgba(_NAVY, 0.5)),
        (1, lab_total, _hex_to_rgba(_GOLD, 0.5)),
        (2, maint_total, _hex_to_rgba(_GREEN, 0.5)),
    ]:
        sources.append(n_routes + cat_off)
        targets.append(total_idx)
        values.append(val)
        link_colors.append(col)

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=22,
            label=labels,
            color=node_colors,
            line=dict(color=_CREAM, width=0.5),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>",
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors,
            hovertemplate=(
                "%{source.label} -> %{target.label}<br>"
                "$%{value:,.0f}<extra></extra>"
            ),
        ),
    ))
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(size=11, color=_DTXT, family="Segoe UI"),
    )
    return _apply(fig, show_grid=False)


def mc_solver_violin(solver_data: dict) -> go.Figure:
    """Violin / box overlay comparing Monte Carlo cost distributions
    across all solvers that have MC results."""
    palette = [_GREEN, _GOLD, _NAVY, _CRIMSON, _SKY, _TEAL]
    fig = go.Figure()
    idx = 0

    for sname, sdata in solver_data.items():
        mc = sdata.get("mc_result")
        if mc is None or not mc.trial_costs:
            continue
        col = palette[idx % len(palette)]
        fig.add_trace(go.Violin(
            y=mc.trial_costs,
            name=sname.upper(),
            box_visible=True,
            meanline_visible=True,
            fillcolor=_hex_to_rgba(col, 0.25),
            line_color=col,
            marker=dict(color=col, size=2, opacity=0.15),
            points="outliers",
            spanmode="hard",
            hoverinfo="y+name",
        ))
        # Add P50 annotation
        fig.add_annotation(
            x=sname.upper(), y=mc.cost_p50,
            text=f"P50: ${mc.cost_p50:,.0f}",
            showarrow=True, arrowhead=0, ax=55, ay=0,
            font=dict(size=9, color=col),
            arrowcolor=col,
        )
        idx += 1

    if not fig.data:
        fig.add_annotation(
            text="No Monte Carlo data available",
            showarrow=False, font=dict(size=14, color=_SLATE),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
        fig.update_layout(height=380)
        return _apply(fig, show_grid=False)

    fig.update_layout(
        title=dict(text="<b>Cost Risk Distribution by Strategy</b>",
                   x=0, font_size=16),
        yaxis_title="Simulated Total Cost ($)",
        yaxis_tickprefix="$",
        height=420,
        showlegend=False,
        violingap=0.35,
        violinmode="group",
    )
    return _apply(fig)


def delivery_timeline(schedules) -> go.Figure:
    """Cumulative step chart -- orders dispatched over time, coloured by
    driver.  Shows the cadence & temporal packing of the fleet plan."""
    if not schedules:
        fig = go.Figure()
        fig.add_annotation(text="No schedule data", showarrow=False,
                           font=dict(size=14, color=_SLATE),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        fig.update_layout(height=380)
        return _apply(fig, show_grid=False)

    # Sort by departure, build cumulative stop count
    sorted_sched = sorted(schedules, key=lambda s: s.departure)

    # Build events: each delivery stop adds one to the cumulative count
    events = []
    for sc in sorted_sched:
        for ev in sc.events:
            if ev.activity == "delivery":
                events.append({
                    "time": ev.arrive,
                    "driver": sc.driver,
                    "city": ev.city,
                    "route_id": sc.route_id,
                })

    events.sort(key=lambda e: e["time"])

    cum_count = 0
    times, counts, labels = [], [], []
    for ev in events:
        cum_count += 1
        times.append(ev["time"])
        counts.append(cum_count)
        labels.append(
            f"<b>{ev['city']}</b><br>"
            f"{ev['driver']} (R{ev['route_id']})<br>"
            f"Delivery #{cum_count}<br>"
            f"{ev['time'].strftime('%b %d %H:%M')}"
        )

    fig = go.Figure()

    # Shaded area under line
    fig.add_trace(go.Scatter(
        x=times, y=counts,
        mode="lines",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(_NAVY, 0.08),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Main step line
    fig.add_trace(go.Scatter(
        x=times, y=counts,
        mode="lines+markers",
        line=dict(shape="hv", width=2.5, color=_NAVY),
        marker=dict(size=5, color=_GOLD,
                    line=dict(width=1, color=_NAVY)),
        text=labels, hoverinfo="text",
        name="Deliveries",
    ))

    # Milestone markers at 25%, 50%, 75%, 100%
    total = len(events)
    for pct in [0.25, 0.50, 0.75, 1.0]:
        idx = min(int(total * pct) - 1, total - 1)
        if idx < 0:
            continue
        fig.add_annotation(
            x=times[idx], y=counts[idx],
            text=f"{int(pct * 100)}%",
            showarrow=True, arrowhead=2,
            ax=0, ay=-28,
            font=dict(size=9, color=_CRIMSON, family="Segoe UI"),
            arrowcolor=_CRIMSON, arrowwidth=1.5,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=_CRIMSON, borderwidth=1,
            borderpad=3,
        )

    fig.update_layout(
        xaxis=dict(title="Time", type="date"),
        yaxis_title="Cumulative Deliveries",
        height=400,
        showlegend=False,
    )
    return _apply(fig)


def efficiency_frontier(routes, costs) -> go.Figure:
    """Scatter plot -- Cost per kg vs distance per stop for each route.
    Routes near the origin are most efficient.  A convex hull envelope
    highlights the efficient frontier."""
    if not routes or not costs:
        fig = go.Figure()
        fig.update_layout(height=380)
        return _apply(fig)

    x_vals, y_vals, sizes, colors, texts = [], [], [], [], []
    for r, c in zip(routes, costs, strict=False):
        wt = r.total_weight_kg or 1
        ns = r.num_stops or 1
        cost_per_kg = c["total_cost"] / wt
        dist_per_stop = c["distance_mi"] / ns
        util = min(r.total_weight_kg / (r.truck.capacity_kg if r.truck else 10000) * 100, 100)

        x_vals.append(dist_per_stop)
        y_vals.append(cost_per_kg)
        sizes.append(max(12, r.num_stops * 5))
        colors.append(util)
        drv = r.truck.driver if r.truck else "?"
        texts.append(
            f"<b>Route {r.route_id}</b><br>{drv}<br>"
            f"Cost/kg: ${cost_per_kg:.2f}<br>"
            f"Dist/stop: {dist_per_stop:.0f} mi<br>"
            f"Utilisation: {util:.0f}%<br>"
            f"Stops: {r.num_stops}"
        )

    fig = go.Figure()

    # Efficient frontier line (lower-left envelope)
    # Sort by x and trace the lower convex hull
    pts = sorted(zip(x_vals, y_vals), key=lambda p: (p[0], p[1]))
    hull_x, hull_y = [pts[0][0]], [pts[0][1]]
    for px, py in pts[1:]:
        if py <= hull_y[-1]:
            hull_x.append(px)
            hull_y.append(py)

    if len(hull_x) > 1:
        fig.add_trace(go.Scatter(
            x=hull_x, y=hull_y,
            mode="lines",
            line=dict(color=_GREEN, width=2, dash="dash"),
            name="Efficient Frontier",
            hoverinfo="skip",
        ))
        # Shaded area below frontier
        fig.add_trace(go.Scatter(
            x=hull_x, y=hull_y,
            mode="lines", line=dict(width=0),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(_GREEN, 0.08),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            colorscale=[[0, _CRIMSON], [0.5, _GOLD], [1.0, _GREEN]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Util %", font=dict(size=10, color=_SLATE)),
                tickfont=dict(size=9, color=_SLATE),
                thickness=12, len=0.6, borderwidth=0,
            ),
            line=dict(width=1, color="rgba(0,38,58,0.25)"),
            opacity=0.88,
        ),
        text=texts, hoverinfo="text",
        name="Routes",
    ))

    fig.update_layout(
        xaxis_title="Distance per Stop (mi)",
        yaxis_title="Cost per kg ($)",
        height=400,
        legend=dict(orientation="h", y=1.08),
    )
    return _apply(fig)


def cost_per_mile_box(solver_data: dict) -> go.Figure:
    """Box + strip chart comparing cost-per-mile distributions across
    solvers.  Each route is a data point.  Shows spread and efficiency."""
    palette = [_GREEN, _GOLD, _NAVY, _CRIMSON, _SKY, _TEAL]
    fig = go.Figure()

    for idx, (sname, sdata) in enumerate(solver_data.items()):
        cpm = [
            c["total_cost"] / c["distance_mi"]
            for c in sdata["costs"]
            if c["distance_mi"] > 0
        ]
        col = palette[idx % len(palette)]
        fig.add_trace(go.Box(
            y=cpm,
            name=sname.upper(),
            boxpoints="all",
            jitter=0.4,
            pointpos=-1.5,
            marker=dict(color=col, size=5, opacity=0.5,
                        line=dict(width=0.5, color=_NAVY)),
            line=dict(color=col, width=2),
            fillcolor=_hex_to_rgba(col, 0.15),
            hovertemplate="<b>%{x}</b><br>$%{y:.2f}/mi<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="<b>Cost per Mile Distribution</b>",
                   x=0, font_size=16),
        yaxis_title="Cost per Mile ($/mi)",
        yaxis_tickprefix="$",
        height=380,
        showlegend=False,
    )
    return _apply(fig)


def solver_mc_summary_table(solver_data: dict) -> go.Figure:
    """Table figure showing MC risk metrics side-by-side across solvers."""
    headers = ["Metric"]
    det_row = ["Deterministic Cost"]
    p50_row = ["P50 (Median)"]
    p95_row = ["P95 (Worst Case)"]
    rob_row = ["Robustness Score"]
    spread_row = ["P95-P5 Spread"]
    co2_row = ["CO2 P50 (kg)"]

    for sname, sdata in solver_data.items():
        mc = sdata.get("mc_result")
        headers.append(sname.upper())
        if mc:
            det_row.append(f"${mc.deterministic_cost:,.0f}")
            p50_row.append(f"${mc.cost_p50:,.0f}")
            p95_row.append(f"${mc.cost_p95:,.0f}")
            rob_row.append(f"{mc.robustness_score:.0f}%")
            spread_row.append(f"${mc.cost_p95 - mc.cost_p5:,.0f}")
            co2_row.append(f"{mc.co2_p50:,.0f}")
        else:
            for row in [det_row, p50_row, p95_row, rob_row, spread_row, co2_row]:
                row.append("--")

    fig = go.Figure(go.Table(
        header=dict(
            values=headers,
            fill_color=_NAVY,
            font=dict(color=_CREAM, size=12, family="Segoe UI"),
            align=["left"] + ["center"] * (len(headers) - 1),
            height=36,
            line=dict(width=0),
        ),
        cells=dict(
            values=[det_row, p50_row, p95_row, rob_row, spread_row, co2_row],
            fill_color=[[_CREAM, _WHITE] * 3],
            font=dict(color=_DTXT, size=12, family="Segoe UI"),
            align=["left"] + ["center"] * (len(headers) - 1),
            height=30,
            line=dict(width=0.5, color=_LGREY),
        ),
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=5, r=5, t=5, b=5),
    )
    return _apply(fig, show_grid=False)


# =====================================================================
# CUTTING-EDGE ANALYTICS -- Advanced Operations Research Charts
# =====================================================================


def fuel_sensitivity_waterfall(routes, costs) -> go.Figure:
    """Diesel price sensitivity -- what-if analysis showing fleet cost
    impact for -20%, -10%, +10%, +20% fuel price changes.

    This is the kind of analysis a CFO would use to hedge risk and
    negotiate fuel contracts.  Each scenario bar shows incremental
    cost vs deterministic baseline.
    """
    base_fuel = sum(c["fuel_cost"] for c in costs)
    base_total = sum(c["total_cost"] for c in costs)
    scenarios = [
        ("-20%", -0.20, _GREEN),
        ("-10%", -0.10, _TEAL),
        ("Base", 0.00, _NAVY),
        ("+10%", +0.10, _GOLD),
        ("+20%", +0.20, _CRIMSON),
    ]

    labels, totals, deltas, colors = [], [], [], []
    for lbl, pct, col in scenarios:
        delta_fuel = base_fuel * pct
        new_total = base_total + delta_fuel
        labels.append(lbl)
        totals.append(new_total)
        deltas.append(delta_fuel)
        colors.append(col)

    fig = go.Figure()

    # Bars for each scenario
    fig.add_trace(go.Bar(
        x=labels, y=totals,
        marker=dict(
            color=colors,
            cornerradius=5,
            line=dict(width=1, color=_NAVY),
        ),
        text=[f"${t:,.0f}" for t in totals],
        textposition="outside",
        textfont=dict(size=11, color=_NAVY, family="Segoe UI"),
        hovertemplate=(
            "<b>Diesel %{x}</b><br>"
            "Fleet Cost: $%{y:,.0f}<br>"
            "Delta: $%{customdata:+,.0f}<extra></extra>"
        ),
        customdata=deltas,
        showlegend=False,
    ))

    # Baseline reference line
    fig.add_hline(
        y=base_total, line_dash="dot", line_color=_SLATE, line_width=1.5,
        annotation_text=f"Baseline ${base_total:,.0f}",
        annotation_font=dict(size=10, color=_SLATE),
        annotation_position="top left",
    )

    # Impact annotation
    max_delta = base_fuel * 0.20
    fig.add_annotation(
        text=(
            f"<b>20% diesel swing = ${max_delta:,.0f} impact</b><br>"
            f"({max_delta / base_total * 100:.1f}% of fleet cost)"
        ),
        xref="paper", yref="paper",
        x=0.98, y=0.95, xanchor="right", yanchor="top",
        showarrow=False,
        font=dict(size=10, color=_NAVY),
        bgcolor="rgba(250,247,242,0.92)",
        bordercolor=_LGREY, borderwidth=1, borderpad=6,
    )

    fig.update_layout(
        height=400,
        xaxis_title="Diesel Price Scenario",
        yaxis_title="Total Fleet Cost ($)",
        yaxis_tickprefix="$",
    )
    return _apply(fig)


def route_network_radial(routes) -> go.Figure:
    """Radial network diagram -- Cincinnati hub at centre with spokes
    to each destination.  Spoke width = cargo weight, spoke colour = route.

    This shows the hub-and-spoke topology and relative volume flow to
    each destination in a single, striking visualisation.
    """
    # Aggregate weight by destination city
    city_weight: dict[str, float] = defaultdict(float)
    city_routes: dict[str, list] = defaultdict(list)
    for r in routes:
        cities = r.city_sequence or [s.city for s in r.stops]
        for c in cities:
            city_weight[c] += r.total_weight_kg
            city_routes[c].append(r.route_id)

    # Sort by weight descending for visual impact
    sorted_cities = sorted(city_weight.items(), key=lambda x: x[1], reverse=True)

    import math
    n = len(sorted_cities)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(height=420)
        return _apply(fig, show_grid=False)

    # Lay out cities in a circle
    angles = [2 * math.pi * i / n for i in range(n)]
    radius = 1.0
    max_w = max(city_weight.values()) or 1

    fig = go.Figure()

    # Spokes
    for i, (city, wt) in enumerate(sorted_cities):
        angle = angles[i]
        cx, cy = radius * math.cos(angle), radius * math.sin(angle)
        col = _PALETTE[i % len(_PALETTE)]
        width = max(1.5, wt / max_w * 12)

        # Spoke line
        fig.add_trace(go.Scatter(
            x=[0, cx], y=[0, cy],
            mode="lines",
            line=dict(width=width, color=_hex_to_rgba(col, 0.55)),
            hoverinfo="skip", showlegend=False,
        ))

        # City node
        node_size = max(14, wt / max_w * 40)
        short_name = city.split(",")[0]
        rids = city_routes[city]
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers+text",
            marker=dict(size=node_size, color=col,
                        line=dict(width=1.5, color=_WHITE)),
            text=[short_name],
            textposition="top center" if cy >= 0 else "bottom center",
            textfont=dict(size=8.5, color=_NAVY),
            hovertemplate=(
                f"<b>{city}</b><br>"
                f"Cargo: {wt:,.0f} kg<br>"
                f"Routes: {', '.join(str(r) for r in rids)}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # Central hub
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=24, color=_NAVY, symbol="circle",
                    line=dict(width=2, color=_GOLD)),
        text=["DEPOT"], textposition="bottom center",
        textfont=dict(size=10, color=_NAVY, family="Segoe UI"),
        hovertemplate="<b>Cincinnati, OH</b><br>Hub Depot<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        height=500,
        xaxis=dict(visible=False, range=[-1.6, 1.6]),
        yaxis=dict(visible=False, range=[-1.6, 1.6], scaleanchor="x"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return _apply(fig, show_grid=False)


def driver_performance_matrix(routes, costs, schedules) -> go.Figure:
    """Heatmap matrix -- rows = drivers, columns = KPI metrics.
    Cells are colour-coded by percentile rank (green = best, crimson = worst).

    KPIs: Avg Cost/Mile, Avg Utilisation, Deliveries/Hour, CO2/Mile, HOS events.
    """
    driver_data: dict[str, dict] = defaultdict(
        lambda: {"cost": 0, "dist": 0, "weight": 0, "cap": 0,
                 "deliveries": 0, "drive_hrs": 0, "co2": 0, "hos": 0}
    )
    sched_map = {sc.route_id: sc for sc in schedules}

    for r, c in zip(routes, costs, strict=False):
        drv = r.truck.driver if r.truck else "?"
        d = driver_data[drv]
        d["cost"] += c["total_cost"]
        d["dist"] += c["distance_mi"]
        d["weight"] += r.total_weight_kg
        d["cap"] += r.truck.capacity_kg if r.truck else 10000
        d["deliveries"] += r.num_stops
        d["co2"] += c["co2_kg"]
        sc = sched_map.get(r.route_id)
        if sc:
            d["drive_hrs"] += sc.work_hrs
            if sc.needs_rest:
                d["hos"] += 1

    if not driver_data:
        fig = go.Figure()
        fig.update_layout(height=380)
        return _apply(fig, show_grid=False)

    drivers = sorted(driver_data.keys())
    metrics = ["Cost/Mile", "Utilisation", "Deliv/Hour", "CO2/Mile", "HOS Events"]

    # Calculate metric values
    raw = []
    for drv in drivers:
        d = driver_data[drv]
        cpm = d["cost"] / d["dist"] if d["dist"] else 0
        util = d["weight"] / d["cap"] * 100 if d["cap"] else 0
        dph = d["deliveries"] / d["drive_hrs"] if d["drive_hrs"] else 0
        co2m = d["co2"] / d["dist"] if d["dist"] else 0
        raw.append([cpm, util, dph, co2m, d["hos"]])

    # Percentile normalise (0..1) -- for each metric, lower is better except util & dph
    z = []
    import numpy as _np
    arr = _np.array(raw)
    for j in range(len(metrics)):
        col_vals = arr[:, j]
        mn, mx = col_vals.min(), col_vals.max()
        rng = mx - mn if mx != mn else 1
        normed = (col_vals - mn) / rng
        # For utilisation and deliveries/hr, higher is better (invert)
        if j in (1, 2):
            normed = 1 - normed
        z.append(normed)
    z = _np.array(z).T.tolist()  # shape: drivers x metrics

    # Annotation text with actual values
    text = []
    for row in raw:
        text.append([
            f"${row[0]:.2f}/mi",
            f"{row[1]:.0f}%",
            f"{row[2]:.2f}/hr",
            f"{row[3]:.1f} kg/mi",
            f"{int(row[4])}",
        ])

    fig = go.Figure(go.Heatmap(
        z=z, x=metrics, y=drivers,
        text=text, texttemplate="%{text}",
        textfont=dict(size=11, color=_WHITE),
        colorscale=[
            [0, _GREEN], [0.3, _TEAL],
            [0.5, _GOLD], [0.7, "#D4843E"],
            [1.0, _CRIMSON],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Risk\nRank", font=dict(size=9, color=_SLATE)),
            tickvals=[0, 0.5, 1],
            ticktext=["Best", "Mid", "Worst"],
            thickness=12, len=0.6, borderwidth=0,
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{x}: %{text}<br>"
            "Rank: %{z:.0%}"
            "<extra></extra>"
        ),
        xgap=3, ygap=3,
    ))

    fig.update_layout(
        height=max(320, len(drivers) * 50 + 80),
        xaxis=dict(title="", side="top",
                   tickfont=dict(size=11, color=_NAVY)),
        yaxis=dict(title="", autorange="reversed",
                   tickfont=dict(size=11, color=_NAVY)),
    )
    return _apply(fig, show_grid=False)


def demand_heatmap_geo(routes) -> go.Figure:
    """Geographic choropleth bubble map -- shows demand density (total kg)
    at each destination.  Larger, warmer bubbles = higher demand.

    This reveals which regions are volume hotspots and helps planners
    rebalance fleet allocation or consider regional warehouses.
    """
    city_demand: dict[str, float] = defaultdict(float)
    city_visits: dict[str, int] = defaultdict(int)
    for r in routes:
        cities = r.city_sequence or [s.city for s in r.stops]
        per_stop_wt = r.total_weight_kg / max(len(cities), 1)
        for c in cities:
            city_demand[c] += per_stop_wt
            city_visits[c] += 1

    if not city_demand:
        fig = go.Figure()
        fig.update_layout(height=420)
        return _apply(fig, show_grid=False)

    # State boundaries base
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locationmode="USA-states",
        locations=[
            "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
            "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
            "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
            "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
            "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
        ],
        z=[0]*50,
        colorscale=[[0, "rgba(245,242,235,0.4)"], [1, "rgba(245,242,235,0.4)"]],
        showscale=False,
        marker_line_color="#C0B8AA",
        marker_line_width=0.8,
        hoverinfo="skip",
    ))

    max_demand = max(city_demand.values()) or 1
    cities_sorted = sorted(city_demand.items(), key=lambda x: x[1], reverse=True)

    lats, lons, sizes, colors, texts = [], [], [], [], []
    for city, demand in cities_sorted:
        lat, lon = CITY_COORDS.get(city, (0, 0))
        if lat == 0:
            continue
        visits = city_visits[city]
        lats.append(lat)
        lons.append(lon)
        sizes.append(max(10, demand / max_demand * 45))
        colors.append(demand)
        texts.append(
            f"<b>{city}</b><br>"
            f"Demand: {demand:,.0f} kg<br>"
            f"Visits: {visits}<br>"
            f"Avg/visit: {demand / visits:,.0f} kg"
        )

    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons,
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            colorscale=[
                [0, _hex_to_rgba(_SKY, 0.6)],
                [0.35, _hex_to_rgba(_GOLD, 0.7)],
                [0.7, _hex_to_rgba(_CRIMSON, 0.75)],
                [1.0, _CRIMSON],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="Demand (kg)", font=dict(size=9, color=_SLATE)),
                tickfont=dict(size=9, color=_SLATE),
                thickness=12, len=0.5, borderwidth=0,
            ),
            line=dict(width=1, color="rgba(0,38,58,0.2)"),
            opacity=0.85,
        ),
        text=texts, hoverinfo="text",
        showlegend=False,
    ))

    # Top 3 labels
    for city, demand in cities_sorted[:3]:
        lat, lon = CITY_COORDS.get(city, (0, 0))
        short = city.split(",")[0]
        fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon],
            mode="text",
            text=[f"<b>{short}</b>"],
            textfont=dict(size=9, color=_NAVY),
            textposition="top center",
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_geos(
        scope="usa",
        showland=True, landcolor="#F5F2EB",
        showlakes=True, lakecolor="#D6E6F0",
        showocean=True, oceancolor="#E8EEF2",
        showcoastlines=True, coastlinecolor="#8C8478",
        showsubunits=True, subunitcolor="#C0B8AA",
        bgcolor="rgba(0,0,0,0)",
        projection_type="albers usa",
    )

    fig.add_annotation(
        text="<b>Demand Density by Destination</b>",
        xref="paper", yref="paper",
        x=0.5, y=1.02, xanchor="center",
        showarrow=False,
        font=dict(size=12, color=_NAVY, family="Segoe UI"),
    )

    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return _apply(fig, show_grid=False)


def fleet_kpi_gauges(stats: dict, mc_result=None) -> go.Figure:
    """Small-multiples gauge grid -- 4 operational KPIs displayed as
    speedometer dials.  At a glance the operator sees cost efficiency,
    fleet utilisation, carbon intensity, and risk robustness."""
    from plotly.subplots import make_subplots

    kpis = [
        {
            "title": "Cost Efficiency",
            "value": stats["per_mile"],
            "suffix": "$/mi",
            "range": [0, max(stats["per_mile"] * 1.8, 5)],
            "thresholds": [
                (0, 0.4, _GREEN), (0.4, 0.7, _GOLD), (0.7, 1.0, _CRIMSON),
            ],
            "target": stats["per_mile"] * 0.85,
        },
        {
            "title": "Fleet Utilisation",
            "value": stats["avg_util"],
            "suffix": "%",
            "range": [0, 100],
            "thresholds": [
                (0, 0.5, _CRIMSON), (0.5, 0.8, _GOLD), (0.8, 1.0, _GREEN),
            ],
            "target": 90,
        },
        {
            "title": "Carbon Intensity",
            "value": stats["total_co2_kg"] / stats["total_dist"] if stats["total_dist"] else 0,
            "suffix": "kg/mi",
            "range": [0, max(stats["total_co2_kg"] / stats["total_dist"] * 2, 3) if stats["total_dist"] else 3],
            "thresholds": [
                (0, 0.4, _GREEN), (0.4, 0.7, _GOLD), (0.7, 1.0, _CRIMSON),
            ],
            "target": None,
        },
        {
            "title": "Robustness",
            "value": mc_result.robustness_score if mc_result else 0,
            "suffix": "%",
            "range": [0, 100],
            "thresholds": [
                (0, 0.5, _CRIMSON), (0.5, 0.8, _GOLD), (0.8, 1.0, _GREEN),
            ],
            "target": 80,
        },
    ]

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}] * 4],
        horizontal_spacing=0.06,
    )

    for i, kpi in enumerate(kpis):
        steps = []
        rng = kpi["range"][1] - kpi["range"][0]
        for lo, hi, col in kpi["thresholds"]:
            steps.append(dict(
                range=[kpi["range"][0] + lo * rng,
                       kpi["range"][0] + hi * rng],
                color=_hex_to_rgba(col, 0.15),
            ))

        val = kpi["value"]
        gauge_color = _GREEN if val / rng <= 0.4 else _GOLD if val / rng <= 0.7 else _CRIMSON
        # For utilisation and robustness, higher is better (invert colour logic)
        if kpi["title"] in ("Fleet Utilisation", "Robustness"):
            pct = val / 100
            gauge_color = _GREEN if pct >= 0.8 else _GOLD if pct >= 0.5 else _CRIMSON

        indicator = go.Indicator(
            mode="gauge+number",
            value=val,
            number=dict(
                suffix=f" {kpi['suffix']}",
                font=dict(size=18, color=_NAVY),
            ),
            gauge=dict(
                axis=dict(range=kpi["range"],
                          tickfont=dict(size=8, color=_SLATE)),
                bar=dict(color=gauge_color, thickness=0.7),
                bgcolor=_CREAM,
                borderwidth=0,
                steps=steps,
                threshold=dict(
                    line=dict(color=_NAVY, width=2),
                    thickness=0.75,
                    value=kpi["target"] or 0,
                ) if kpi["target"] else {},
            ),
            title=dict(text=kpi["title"],
                       font=dict(size=11, color=_SLATE)),
            domain=dict(row=0, column=i),
        )
        fig.add_trace(indicator, row=1, col=i + 1)

    fig.update_layout(
        height=250,
        margin=dict(l=15, r=15, t=45, b=15),
    )
    return _apply(fig, show_grid=False)


def cost_co2_tradeoff(solver_data: dict) -> go.Figure:
    """Cost vs CO2 trade-off scatter -- reveals which solver strategy
    achieves the best balance between economic and environmental goals.

    Bubble size = number of routes.  Colour = solver.
    An ideal-point annotation shows the theoretical optimum.
    """
    if not solver_data or len(solver_data) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need 2+ solver strategies for trade-off analysis",
            showarrow=False, font=dict(size=14, color=_SLATE),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
        fig.update_layout(height=380)
        return _apply(fig, show_grid=False)

    palette = [_GREEN, _GOLD, _NAVY, _CRIMSON, _SKY, _TEAL]
    names, costs_v, co2_v, routes_v = [], [], [], []

    for sname, sdata in solver_data.items():
        st = sdata.get("stats", {})
        if not st:
            continue
        names.append(sname.upper())
        costs_v.append(st["total_cost"])
        co2_v.append(st["total_co2_kg"])
        routes_v.append(st["n_routes"])

    fig = go.Figure()

    # Ideal point (min cost, min CO2) -- annotated
    ideal_cost = min(costs_v) * 0.95
    ideal_co2 = min(co2_v) * 0.95

    fig.add_trace(go.Scatter(
        x=[ideal_cost], y=[ideal_co2],
        mode="markers+text",
        marker=dict(size=14, color=_hex_to_rgba(_GREEN, 0.3),
                    symbol="star", line=dict(width=2, color=_GREEN)),
        text=["\u2605 Ideal"], textposition="top center",
        textfont=dict(size=9, color=_GREEN),
        hovertemplate="<b>Theoretical Ideal</b><br>Lowest cost + lowest CO2<extra></extra>",
        showlegend=False,
    ))

    # Solver bubbles
    for i, (name, cost, co2, nr) in enumerate(zip(names, costs_v, co2_v, routes_v)):
        col = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=[cost], y=[co2],
            mode="markers+text",
            marker=dict(
                size=max(20, nr * 1.2),
                color=_hex_to_rgba(col, 0.55),
                line=dict(width=2, color=col),
            ),
            text=[name], textposition="top center",
            textfont=dict(size=11, color=col, family="Georgia"),
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Cost: ${cost:,.0f}<br>"
                f"CO2: {co2:,.0f} kg<br>"
                f"Routes: {nr}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # Connecting lines to ideal point
    for i, (cost, co2) in enumerate(zip(costs_v, co2_v)):
        fig.add_trace(go.Scatter(
            x=[ideal_cost, cost], y=[ideal_co2, co2],
            mode="lines",
            line=dict(width=1, dash="dot", color=_LGREY),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        xaxis_title="Total Fleet Cost ($)",
        yaxis_title="Total CO2 Emissions (kg)",
        xaxis_tickprefix="$",
        height=420,
    )
    return _apply(fig)


# =====================================================================
#  ADDITIONAL CHARTS  (ported from fleet_analytics.ipynb)
# =====================================================================


def truck_class_efficiency(routes, costs) -> go.Figure:
    """Cost per ton-mile box plot by truck class."""
    from src.config import FUEL_ECONOMY_MPG

    # Map truck lengths to class names
    _CLASS = {16.06: "Class 8", 14.82: "Class 7", 13.98: "Class 6", 11.57: "Class 5"}
    _CLASS_CLR = {"Class 8": _NAVY, "Class 7": _GOLD, "Class 6": _GREEN, "Class 5": _CRIMSON}

    data: dict[str, list[float]] = {}
    for r, c in zip(routes, costs):
        if r.truck is None or c["distance_mi"] == 0 or r.total_weight_kg == 0:
            continue
        cls = _CLASS.get(r.truck.length_m, f"{r.truck.length_m:.1f} m")
        ton_mi = (r.total_weight_kg / 1000) * c["distance_mi"]
        cost_per_tm = c["total_cost"] / ton_mi if ton_mi else 0
        data.setdefault(cls, []).append(cost_per_tm)

    fig = go.Figure()
    for cls in ["Class 8", "Class 7", "Class 6", "Class 5"]:
        vals = data.get(cls, [])
        if not vals:
            continue
        fig.add_trace(go.Box(
            y=vals, name=cls,
            marker_color=_CLASS_CLR.get(cls, _SLATE),
            boxmean="sd", jitter=0.4, pointpos=-1.5, boxpoints="all",
            marker=dict(size=6, opacity=0.7),
            hovertemplate="$%{y:.4f}/ton-mile<extra></extra>",
        ))

    fig.update_layout(
        yaxis=dict(title="Cost per Ton-Mile ($)", tickprefix="$"),
        showlegend=False, height=420,
    )
    return _apply(fig)


def alns_operator_weights(metadata: dict) -> go.Figure:
    """Polar bar chart of ALNS destroy/repair operator weights."""
    from plotly.subplots import make_subplots

    op_w = metadata.get("operator_weights", {})
    destroy_w = op_w.get("destroy", {})
    repair_w = op_w.get("repair", {})

    if not destroy_w and not repair_w:
        fig = go.Figure()
        fig.add_annotation(text="No ALNS operator data available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color=_SLATE))
        fig.update_layout(height=380)
        return _apply(fig, show_grid=False)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=("Destroy Operators", "Repair Operators"),
    )

    d_names = list(destroy_w.keys())
    d_vals = [v * 100 for v in destroy_w.values()]
    d_cols = [_NAVY, _GOLD, _GREEN, _CRIMSON][:len(d_names)]

    fig.add_trace(go.Barpolar(
        r=d_vals, theta=d_names,
        marker=dict(color=d_cols, line=dict(color="white", width=2)),
        text=[f"{v:.0f}%" for v in d_vals],
        hovertemplate="<b>%{theta}</b><br>Weight: %{r:.1f}%<extra></extra>",
    ), row=1, col=1)

    r_names = list(repair_w.keys())
    r_vals = [v * 100 for v in repair_w.values()]
    r_cols = [_SKY, _EARTH, _TEAL][:len(r_names)]

    fig.add_trace(go.Barpolar(
        r=r_vals, theta=r_names,
        marker=dict(color=r_cols, line=dict(color="white", width=2)),
        text=[f"{v:.0f}%" for v in r_vals],
        hovertemplate="<b>%{theta}</b><br>Weight: %{r:.1f}%<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        height=420,
        polar=dict(bgcolor=_CREAM, radialaxis=dict(visible=True, ticksuffix="%")),
        polar2=dict(bgcolor=_CREAM, radialaxis=dict(visible=True, ticksuffix="%")),
        showlegend=False,
    )
    return _apply(fig, show_grid=False)


def economies_of_distance(routes, costs) -> go.Figure:
    """Cost-per-mile vs route distance with hyperbolic amortisation curve."""
    import numpy as np

    _CLASS = {16.06: "Class 8", 14.82: "Class 7", 13.98: "Class 6", 11.57: "Class 5"}
    _CLASS_CLR = {"Class 8": _NAVY, "Class 7": _GOLD, "Class 6": _GREEN, "Class 5": _CRIMSON}

    dists, cpms, classes = [], [], []
    for r, c in zip(routes, costs):
        if c["distance_mi"] == 0:
            continue
        d = c["distance_mi"]
        cpm = c["total_cost"] / d
        cls = _CLASS.get(r.truck.length_m, "Other") if r.truck else "Other"
        dists.append(d)
        cpms.append(cpm)
        classes.append(cls)

    fig = go.Figure()

    # Scatter by truck class
    for cls in ["Class 8", "Class 7", "Class 6", "Class 5"]:
        idx = [i for i, c in enumerate(classes) if c == cls]
        if not idx:
            continue
        fig.add_trace(go.Scatter(
            x=[dists[i] for i in idx], y=[cpms[i] for i in idx],
            mode="markers+text",
            marker=dict(size=12, color=_CLASS_CLR.get(cls, _SLATE),
                        line=dict(width=1, color="white")),
            text=[f"R{routes[i].route_id}" for i in idx],
            textposition="top center", textfont=dict(size=8),
            name=cls,
            hovertemplate=(
                "<b>Route %{customdata}</b><br>"
                "Distance: %{x:,.0f} mi<br>"
                "Cost/mile: $%{y:.2f}<extra></extra>"
            ),
            customdata=[routes[i].route_id for i in idx],
        ))

    # Hyperbolic trend line: cost_per_mile ~ a + b/distance
    if len(dists) > 3:
        x_arr = np.array(dists, dtype=float)
        y_arr = np.array(cpms, dtype=float)
        try:
            # Simple a + b/x fit via least squares
            A = np.column_stack([np.ones_like(x_arr), 1.0 / x_arr])
            params, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
            a, b = params
            x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
            fig.add_trace(go.Scatter(
                x=x_line, y=a + b / x_line,
                mode="lines", name="Amortisation curve",
                line=dict(color=_CRIMSON, width=2, dash="dash"),
                hoverinfo="skip",
            ))
        except Exception:
            pass

    fig.update_layout(
        xaxis=dict(title="Route Distance (miles)"),
        yaxis=dict(title="Cost per Mile ($)", tickprefix="$"),
        height=450,
    )
    return _apply(fig)


def hos_compliance(routes, costs, schedules) -> go.Figure:
    """Stacked horizontal bar: driving hours vs mandatory rest per route."""
    labels, drive, rest = [], [], []

    for r, c, s in zip(routes, costs, schedules):
        labels.append(f"R{r.route_id}")
        drive.append(c["driving_hrs"])
        rest_hrs = sum(
            (rb["end"] - rb["start"]).total_seconds() / 3600
            for rb in s.rest_breaks
        )
        rest.append(rest_hrs)

    # Sort by driving hours ascending
    order = sorted(range(len(drive)), key=lambda i: drive[i])
    labels = [labels[i] for i in order]
    drive = [drive[i] for i in order]
    rest = [rest[i] for i in order]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=drive, orientation="h",
        name="Driving", marker_color=_NAVY,
        hovertemplate="Driving: %{x:.1f}h<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=rest, orientation="h",
        name="Mandatory Rest", marker_color=_SLATE,
        hovertemplate="Rest: %{x:.1f}h<extra></extra>",
    ))
    fig.add_vline(x=14, line_dash="dot", line_color=_CRIMSON, line_width=2,
                  annotation_text="14h HOS limit",
                  annotation_position="top right",
                  annotation_font=dict(color=_CRIMSON, size=10))

    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Hours"),
        yaxis=dict(title="", tickfont=dict(size=9)),
        height=max(350, len(labels) * 22),
    )
    return _apply(fig)


def customer_dest_heatmap(routes) -> go.Figure:
    """Company x Destination weight heatmap."""
    # Collect all orders across routes
    companies: dict[str, dict[str, float]] = {}
    for r in routes:
        for s in r.stops:
            for o in s.orders:
                companies.setdefault(o.company, {})
                companies[o.company][o.destination] = (
                    companies[o.company].get(o.destination, 0) + o.total_weight_kg
                )

    if not companies:
        fig = go.Figure()
        fig.add_annotation(text="No order data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=350)
        return _apply(fig, show_grid=False)

    # Build matrix
    all_dests = sorted({d for co in companies.values() for d in co})
    all_comps = sorted(companies.keys(),
                       key=lambda c: -sum(companies[c].values()))
    z = [[companies[co].get(d, 0) for d in all_dests] for co in all_comps]

    fig = go.Figure(go.Heatmap(
        z=z, x=all_dests, y=all_comps,
        colorscale=[
            [0.0, _CREAM], [0.3, _WHEAT], [0.6, _GOLD],
            [0.85, _EARTH], [1.0, _NAVY],
        ],
        colorbar=dict(title="kg", tickformat=",.0f"),
        hovertemplate="<b>%{y}</b> \u2192 %{x}<br>%{z:,.0f} kg<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(title="", tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(title="", tickfont=dict(size=10), autorange="reversed"),
        height=max(380, len(all_comps) * 35 + 100),
    )
    return _apply(fig, show_grid=False)


def stops_distribution(routes) -> go.Figure:
    """Histogram of stops per route."""
    from collections import Counter

    counts = Counter(r.num_stops for r in routes)
    stops_sorted = sorted(counts.keys())
    vals = [counts[s] for s in stops_sorted]
    mode_s = max(counts, key=counts.get) if counts else 0

    colours = [_NAVY if s == mode_s else _GOLD for s in stops_sorted]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(s) for s in stops_sorted], y=vals,
        marker_color=colours,
        text=vals, textposition="outside",
        textfont=dict(size=14, color=_NAVY),
        hovertemplate="%{x} stops: %{y} routes<extra></extra>",
    ))

    if mode_s and counts[mode_s]:
        fig.add_annotation(
            x=str(mode_s), y=counts[mode_s] + 0.5,
            text=f"Mode: {mode_s} stop{'s' if mode_s > 1 else ''}",
            showarrow=True, arrowhead=2, arrowcolor=_CRIMSON,
            font=dict(color=_CRIMSON, size=12),
        )

    fig.update_layout(
        xaxis=dict(title="Stops per Route"),
        yaxis=dict(title="Number of Routes"),
        height=380, showlegend=False,
    )
    return _apply(fig)

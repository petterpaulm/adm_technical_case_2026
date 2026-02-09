"""
Console output and CSV export for optimisation results.

Exports
-------
  route_details.csv      -- one row per route with cost + schedule data
  order_assignments.csv  -- one row per order showing truck / route
  fleet_summary.csv      -- aggregate KPIs
  dispatch_timeline.csv  -- stop-by-stop chronological event log
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.config import RESULTS_DIR, WAREHOUSE
from src.cost_engine import fleet_summary
from src.models import Route
from src.scheduler import RouteSchedule

# -- console helpers -----------------------------------------------------


def banner(title: str, width: int = 72) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_summary(
    routes: list[Route],
    costs: list[dict[str, float]],
    schedules: list[RouteSchedule],
) -> None:
    s = fleet_summary(routes, costs)

    banner("FLEET DELIVERY OPTIMISATION -- RESULTS")
    print(f"  Routes planned       : {s['num_routes']}")
    print(f"  Orders fulfilled     : {s['total_orders']}")
    print(f"  Delivery stops       : {s['total_stops']}")
    print(f"  Cargo weight         : {s['total_weight_kg']:,.0f} kg")
    print(f"  Total distance       : {s['total_distance_mi']:,.0f} mi")
    print(f"  Fuel consumed        : {s['total_fuel_gal']:,.0f} gal")

    banner("COST BREAKDOWN")
    print(f"  Fuel                 : ${s['fuel_cost']:>12,.2f}")
    print(f"  Labour               : ${s['labour_cost']:>12,.2f}")
    print(f"  Maintenance          : ${s['maint_cost']:>12,.2f}")
    print(f"  {'-' * 42}")
    print(f"  TOTAL                : ${s['total_cost']:>12,.2f}")
    print(f"  Per mile             : ${s['cost_per_mile']:>12,.2f}")
    print(f"  Per route            : ${s['cost_per_route']:>12,.2f}")

    banner("CARBON EMISSIONS")
    print(f"  Total CO2            : {s['total_co2_kg']:>10,.0f} kg")
    print(f"  CO2 per route        : {s['co2_per_route']:>10,.0f} kg")
    print(f"  CO2 per mile         : {s['co2_per_mile']:>10,.3f} kg")

    banner("ROUTE DETAILS")
    cost_map = {r.route_id: c for r, c in zip(routes, costs)}
    sched_map = {sc.route_id: sc for sc in schedules}

    for route in sorted(routes, key=lambda r: r.route_id):
        c = cost_map.get(route.route_id, {})
        sc = sched_map.get(route.route_id)

        path = " -> ".join([WAREHOUSE, *route.city_sequence, WAREHOUSE])
        drv = f"{route.truck.driver} (Truck {route.truck.truck_id})" if route.truck else "--"
        dep = sc.departure.strftime("%a %b %d %I:%M %p") if sc else "--"
        ret = sc.arrival.strftime("%a %b %d %I:%M %p") if sc else "--"
        hos = " [HOS break]" if sc and sc.needs_rest else ""

        print(f"\n  Route {route.route_id:>3d}  │  {drv}")
        print(
            f"    Stops {route.num_stops}  │  {route.total_weight_kg:,.0f} kg"
            f"  │  {c.get('distance_mi', 0):,.0f} mi"
        )
        print(f"    {dep} -> {ret}{hos}")
        print(
            f"    ${c.get('total_cost', 0):,.2f}"
            f"  (fuel ${c.get('fuel_cost', 0):,.2f}"
            f" + labour ${c.get('labour_cost', 0):,.2f}"
            f" + maint ${c.get('maint_cost', 0):,.2f})"
            f"  │  CO2 {c.get('co2_kg', 0):,.0f} kg"
        )
        print(f"    {path}")

    print("\n" + "=" * 72)


# -- CSV exports ---------------------------------------------------------


def _csv(path: Path, header: list[str], rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  [ok] {path.name}")


def export_route_details(routes, costs, schedules, path=None):
    path = path or RESULTS_DIR / "route_details.csv"
    sm = {s.route_id: s for s in schedules}
    rows = []
    for r, c in zip(routes, costs):
        sc = sm.get(r.route_id)
        rows.append(
            [
                r.route_id,
                r.truck.truck_id if r.truck else "",
                r.truck.driver if r.truck else "",
                r.num_stops,
                round(r.total_weight_kg, 1),
                c["distance_mi"],
                c["driving_hrs"],
                c["work_hrs"],
                c["fuel_gal"],
                c["fuel_cost"],
                c["labour_cost"],
                c["maint_cost"],
                c["total_cost"],
                c["co2_kg"],
                sc.departure.isoformat() if sc else "",
                sc.arrival.isoformat() if sc else "",
                "Yes" if sc and sc.needs_rest else "No",
                " -> ".join([WAREHOUSE, *r.city_sequence, WAREHOUSE]),
            ]
        )
    _csv(
        path,
        [
            "Route",
            "Truck",
            "Driver",
            "Stops",
            "Weight_kg",
            "Dist_mi",
            "Drive_hrs",
            "Work_hrs",
            "Fuel_gal",
            "Fuel$",
            "Labour$",
            "Maint$",
            "Total$",
            "CO2_kg",
            "Departure",
            "Return",
            "HOS_break",
            "Cities",
        ],
        rows,
    )


def export_order_assignments(routes, path=None):
    path = path or RESULTS_DIR / "order_assignments.csv"
    rows = []
    for r in routes:
        for o in r.all_orders:
            rows.append(
                [
                    o.idx,
                    o.company,
                    o.item_id,
                    o.units,
                    o.destination,
                    round(o.total_weight_kg, 1),
                    r.route_id,
                    r.truck.truck_id if r.truck else "",
                    r.truck.driver if r.truck else "",
                ]
            )
    _csv(
        path,
        [
            "Order",
            "Company",
            "Item",
            "Units",
            "Destination",
            "Weight_kg",
            "Route",
            "Truck",
            "Driver",
        ],
        rows,
    )


def export_fleet_summary(routes, costs, path=None):
    path = path or RESULTS_DIR / "fleet_summary.csv"
    s = fleet_summary(routes, costs)
    rows = [[k.replace("_", " ").title(), v] for k, v in s.items()]
    _csv(path, ["Metric", "Value"], rows)


def export_dispatch_timeline(schedules, path=None):
    path = path or RESULTS_DIR / "dispatch_timeline.csv"
    rows = []
    for sc in schedules:
        for ev in sc.events:
            rows.append(
                [
                    sc.route_id,
                    sc.truck_id,
                    sc.driver,
                    ev.city,
                    ev.arrive.strftime("%Y-%m-%d %H:%M"),
                    ev.depart.strftime("%Y-%m-%d %H:%M"),
                    ev.activity,
                ]
            )
        for rest in sc.rest_breaks:
            rows.append(
                [
                    sc.route_id,
                    sc.truck_id,
                    sc.driver,
                    "-- REST --",
                    rest["start"].strftime("%Y-%m-%d %H:%M"),
                    rest["end"].strftime("%Y-%m-%d %H:%M"),
                    "mandatory_rest",
                ]
            )
    _csv(
        path,
        [
            "Route",
            "Truck",
            "Driver",
            "City",
            "Arrive",
            "Depart",
            "Activity",
        ],
        rows,
    )


def export_all(routes, costs, schedules):
    print_summary(routes, costs, schedules)
    banner("EXPORTING CSV FILES")
    export_route_details(routes, costs, schedules)
    export_order_assignments(routes)
    export_fleet_summary(routes, costs)
    export_dispatch_timeline(schedules)
    print(f"\n  All files -> {RESULTS_DIR}\n")

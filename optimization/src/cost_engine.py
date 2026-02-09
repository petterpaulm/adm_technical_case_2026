"""
Cost computation for individual routes and the fleet as a whole.

Three cost components per route:
    fuel        = (distance / mpg) x diesel_price
    labour      = total_work_hours x driver_hourly_rate
    maintenance = distance x per-mile rate

Work hours include driving + warehouse loading + stop overhead.
"""

from __future__ import annotations

from src.config import (
    AVG_SPEED_MPH,
    CO2_KG_PER_GALLON,
    DIESEL_PRICE_PER_GALLON,
    FUEL_ECONOMY_MPG,
    LOADING_MINUTES,
    MAINTENANCE_PER_MILE,
    STOP_OVERHEAD_MINUTES,
)
from src.models import Route


def _mpg_for(truck_length_m: float) -> float:
    return FUEL_ECONOMY_MPG.get(f"{truck_length_m:.2f}", 6.0)


def route_cost(route: Route) -> dict[str, float]:
    """Detailed cost breakdown for one route."""
    if route.truck is None:
        raise ValueError(f"Route {route.route_id} has no truck assigned")

    miles = route.total_distance_mi
    mpg = _mpg_for(route.truck.length_m)

    driving_hrs = miles / AVG_SPEED_MPH if AVG_SPEED_MPH else 0.0
    load_hrs = LOADING_MINUTES / 60.0
    stop_hrs = (route.num_stops * STOP_OVERHEAD_MINUTES) / 60.0
    work_hrs = driving_hrs + load_hrs + stop_hrs

    fuel_gal = miles / mpg if mpg else 0.0
    fuel = fuel_gal * DIESEL_PRICE_PER_GALLON
    labour = work_hrs * route.truck.hourly_rate
    maint = miles * MAINTENANCE_PER_MILE
    co2_kg = fuel_gal * CO2_KG_PER_GALLON

    return {
        "distance_mi": round(miles, 1),
        "driving_hrs": round(driving_hrs, 2),
        "load_hrs": round(load_hrs, 2),
        "stop_hrs": round(stop_hrs, 2),
        "work_hrs": round(work_hrs, 2),
        "fuel_gal": round(fuel_gal, 1),
        "fuel_cost": round(fuel, 2),
        "labour_cost": round(labour, 2),
        "maint_cost": round(maint, 2),
        "total_cost": round(fuel + labour + maint, 2),
        "co2_kg": round(co2_kg, 1),
    }


def all_route_costs(routes: list[Route]) -> list[dict[str, float]]:
    return [route_cost(r) for r in routes]


def fleet_summary(
    routes: list[Route],
    costs: list[dict[str, float]],
) -> dict[str, float]:
    """Aggregate fleet-level KPIs."""
    t_mi = sum(c["distance_mi"] for c in costs)
    t_fuel = sum(c["fuel_gal"] for c in costs)
    t_fc = sum(c["fuel_cost"] for c in costs)
    t_lc = sum(c["labour_cost"] for c in costs)
    t_mc = sum(c["maint_cost"] for c in costs)
    t_cost = sum(c["total_cost"] for c in costs)
    t_co2 = sum(c["co2_kg"] for c in costs)
    t_wt = sum(r.total_weight_kg for r in routes)
    t_ord = sum(len(r.all_orders) for r in routes)
    n = len(routes) or 1

    return {
        "num_routes": len(routes),
        "total_orders": t_ord,
        "total_stops": sum(r.num_stops for r in routes),
        "total_weight_kg": round(t_wt, 1),
        "total_distance_mi": round(t_mi, 1),
        "total_fuel_gal": round(t_fuel, 1),
        "fuel_cost": round(t_fc, 2),
        "labour_cost": round(t_lc, 2),
        "maint_cost": round(t_mc, 2),
        "total_cost": round(t_cost, 2),
        "total_co2_kg": round(t_co2, 1),
        "co2_per_route": round(t_co2 / n, 1),
        "co2_per_mile": round(t_co2 / t_mi, 3) if t_mi else 0.0,
        "cost_per_mile": round(t_cost / t_mi, 2) if t_mi else 0.0,
        "cost_per_route": round(t_cost / n, 2),
    }

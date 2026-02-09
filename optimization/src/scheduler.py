"""
HOS-compliant driver scheduling.

Simulates every route minute-by-minute to produce departure / arrival
timestamps, mandatory rest insertions, and the full event timeline.

HOS rule (from the problem statement):
    - A driver may work up to 14 consecutive hours (driving + loading
      + unloading + paperwork).
    - After 14 h of work a mandatory 10 h rest break is required.

Multi-trip handling:
    A truck that returns to the warehouse early can be re-dispatched on
    the next available route once the driver's rest requirement is met.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.config import (
    AVG_SPEED_MPH,
    FIRST_DEPARTURE,
    LOADING_MINUTES,
    MANDATORY_REST_HOURS,
    MAX_WORK_HOURS,
    STOP_OVERHEAD_MINUTES,
    WAREHOUSE,
)
from src.models import Route

# -- data structures -----------------------------------------------------


@dataclass
class StopEvent:
    city: str
    arrive: datetime
    depart: datetime
    activity: str  # "loading" | "delivery" | "return"


@dataclass
class RouteSchedule:
    route_id: int
    truck_id: str
    driver: str
    departure: datetime
    arrival: datetime
    work_hrs: float
    events: list[StopEvent] = field(default_factory=list)
    rest_breaks: list[dict[str, datetime]] = field(default_factory=list)
    needs_rest: bool = False


# -- helpers -------------------------------------------------------------


def _drive_hours(miles: float) -> float:
    return miles / AVG_SPEED_MPH if AVG_SPEED_MPH else 0.0


def _td(minutes: float) -> timedelta:
    return timedelta(minutes=minutes)


# -- main scheduler ------------------------------------------------------


def build_schedule(
    routes: list[Route],
    distances: dict,
    costs: list[dict[str, float]],
) -> list[RouteSchedule]:
    """
    Walk through every route chronologically, inserting HOS rest breaks
    when a driver's accumulated work hours would exceed the legal limit.
    """
    truck_available: dict[str, datetime] = {}
    schedules: list[RouteSchedule] = []

    # dispatch shortest routes first so trucks free up quickly
    indexed = sorted(enumerate(routes), key=lambda ir: ir[1].total_distance_mi)

    for orig_idx, route in indexed:
        if route.truck is None:
            continue

        tid = route.truck.truck_id
        drv = route.truck.driver
        earliest = truck_available.get(tid, FIRST_DEPARTURE)
        depart = max(earliest, FIRST_DEPARTURE)

        events: list[StopEvent] = []
        rests: list[dict[str, datetime]] = []
        now = depart
        worked = 0.0

        # warehouse loading
        load_end = now + _td(LOADING_MINUTES)
        events.append(StopEvent(WAREHOUSE, now, load_end, "loading"))
        worked += LOADING_MINUTES / 60
        now = load_end

        # deliver to each stop in optimised sequence
        cities = route.city_sequence or [s.city for s in route.stops]
        prev = WAREHOUSE
        for city in cities:
            dist = distances.get((prev, city), 0.0)
            drive_h = _drive_hours(dist)
            projected = worked + drive_h + STOP_OVERHEAD_MINUTES / 60

            if projected > MAX_WORK_HOURS:
                rest_start = now
                rest_end = rest_start + timedelta(hours=MANDATORY_REST_HOURS)
                rests.append({"start": rest_start, "end": rest_end})
                now = rest_end
                worked = 0.0

            arrive = now + timedelta(hours=drive_h)
            worked += drive_h
            stop_end = arrive + _td(STOP_OVERHEAD_MINUTES)
            worked += STOP_OVERHEAD_MINUTES / 60

            events.append(StopEvent(city, arrive, stop_end, "delivery"))
            now = stop_end
            prev = city

        # return leg
        ret_dist = distances.get((prev, WAREHOUSE), 0.0)
        ret_h = _drive_hours(ret_dist)
        if worked + ret_h > MAX_WORK_HOURS:
            rest_start = now
            rest_end = rest_start + timedelta(hours=MANDATORY_REST_HOURS)
            rests.append({"start": rest_start, "end": rest_end})
            now = rest_end
            worked = 0.0

        ret_arrive = now + timedelta(hours=ret_h)
        events.append(StopEvent(WAREHOUSE, ret_arrive, ret_arrive, "return"))

        work_total = costs[orig_idx]["work_hrs"] if orig_idx < len(costs) else worked

        schedules.append(
            RouteSchedule(
                route_id=route.route_id,
                truck_id=tid,
                driver=drv,
                departure=depart,
                arrival=ret_arrive,
                work_hrs=round(work_total, 2),
                events=events,
                rest_breaks=rests,
                needs_rest=len(rests) > 0,
            )
        )

        truck_available[tid] = ret_arrive + timedelta(hours=MANDATORY_REST_HOURS)

    schedules.sort(key=lambda s: s.departure)
    return schedules

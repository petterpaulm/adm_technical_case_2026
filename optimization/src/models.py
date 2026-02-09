"""
Domain models shared across the entire optimisation pipeline.

Every domain concept (Item, Order, Truck, Stop, Route, Solution) lives here
so that solvers, schedulers, and the dashboard all speak the same language.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


# -- catalogue item ------------------------------------------------------
@dataclass(slots=True)
class Item:
    item_id: int
    weight_lbs: float
    weight_kg: float
    warehouse: str


# -- order line ----------------------------------------------------------
@dataclass(slots=True)
class Order:
    idx: int
    company: str
    item_id: int
    units: int
    destination: str
    unit_weight_kg: float
    total_weight_kg: float


# -- truck + driver ------------------------------------------------------
@dataclass(slots=True)
class Truck:
    truck_id: str
    driver: str
    experience_yrs: int
    length_m: float
    capacity_kg: float
    top_speed_kmh: float
    hourly_rate: float
    fuel_tank_gal: float
    spare_tires: int
    age_yrs: float


# -- delivery stop ------------------------------------------------------
@dataclass
class Stop:
    city: str
    orders: list[Order] = field(default_factory=list)
    weight_kg: float = 0.0

    def add(self, order: Order) -> None:
        self.orders.append(order)
        self.weight_kg += order.total_weight_kg

    def remove(self, order: Order) -> None:
        self.orders.remove(order)
        self.weight_kg -= order.total_weight_kg


# -- route ---------------------------------------------------------------
@dataclass
class Route:
    route_id: int
    stops: list[Stop] = field(default_factory=list)
    truck: Truck | None = None
    total_weight_kg: float = 0.0
    total_distance_mi: float = 0.0
    city_sequence: list[str] = field(default_factory=list)

    @property
    def num_stops(self) -> int:
        return len(self.stops)

    @property
    def all_orders(self) -> list[Order]:
        return [o for s in self.stops for o in s.orders]

    def recalc_weight(self) -> None:
        self.total_weight_kg = sum(s.weight_kg for s in self.stops)


# -- complete solution ---------------------------------------------------
@dataclass
class Solution:
    """Encapsulates a full set of routes -- the thing every solver returns."""

    routes: list[Route] = field(default_factory=list)
    total_cost: float = 0.0
    total_distance: float = 0.0
    solver_name: str = ""
    metadata: dict = field(default_factory=dict)

    def deep_copy(self) -> Solution:
        return copy.deepcopy(self)

    @property
    def num_routes(self) -> int:
        return len(self.routes)

    def recalc_distance(self, distances: dict[tuple[str, str], float]) -> None:
        """Recompute total_distance from scratch using the distance lookup."""
        from src.distance import route_distance  # avoid circular import

        self.total_distance = 0.0
        for route in self.routes:
            route.total_distance_mi = route_distance(route.city_sequence, distances)
            self.total_distance += route.total_distance_mi

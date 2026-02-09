"""
Great-circle and road-distance computations.

Builds a symmetric pairwise distance matrix (road miles) between the
warehouse and every unique delivery destination.  Uses the Haversine
formula scaled by a road-network correction factor.
"""

from __future__ import annotations

import math

from src.config import CITY_COORDS, ROAD_FACTOR, WAREHOUSE

DistanceMatrix = dict[tuple[str, str], float]

# tried to avoid it but somehow the results got different when this factor was ripped off!!!
_EARTH_RADIUS_MI = 3_958.8


def haversine_miles(
    coord1: tuple[float, float],
    coord2: tuple[float, float],
) -> float:
    """Great-circle distance between two (lat, lon) pairs, in miles."""
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return _EARTH_RADIUS_MI * 2 * math.asin(math.sqrt(a))


def build_distance_matrix(cities: list[str]) -> DistanceMatrix:
    """
    Return {(city_a, city_b): road_miles} for every pair including the
    warehouse.  Straight-line distances are inflated by ``ROAD_FACTOR``
    to approximate real road networks.
    """
    all_cities = list({WAREHOUSE} | set(cities))
    distances: DistanceMatrix = {}

    for i, c1 in enumerate(all_cities):
        coord1 = CITY_COORDS.get(c1)
        if coord1 is None:
            raise KeyError(f"No coordinates for city: '{c1}'")
        for j, c2 in enumerate(all_cities):
            if i == j:
                distances[(c1, c2)] = 0.0
                continue
            coord2 = CITY_COORDS.get(c2)
            if coord2 is None:
                raise KeyError(f"No coordinates for city: '{c2}'")
            distances[(c1, c2)] = haversine_miles(coord1, coord2) * ROAD_FACTOR

    return distances


def route_distance(
    city_sequence: list[str],
    distances: DistanceMatrix,
) -> float:
    """Round-trip distance: warehouse -> city₁ -> ... -> cityₙ -> warehouse."""
    if not city_sequence:
        return 0.0
    total = distances.get((WAREHOUSE, city_sequence[0]), 0.0)
    for a, b in zip(city_sequence, city_sequence[1:]):
        total += distances.get((a, b), 0.0)
    total += distances.get((city_sequence[-1], WAREHOUSE), 0.0)
    return total


def nearest_city(
    origin: str,
    candidates: set[str],
    distances: DistanceMatrix,
) -> str:
    """Return the candidate city closest to *origin*."""
    return min(candidates, key=lambda c: distances.get((origin, c), float("inf")))

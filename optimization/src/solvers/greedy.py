"""
Greedy constructive heuristic -- the baseline solver.

Three phases executed in sequence:
    1. Bin-pack orders onto routes (First-Fit Decreasing by weight,
       with a geographic proximity tie-breaker).
    2. Merge short routes via Clarke-Wright savings.
    3. Optimise visit order within each route (Nearest-Neighbour + 2-opt).
    4. Assign routes to the cheapest feasible truck.

This solver runs in well under a second and produces a solid initial
solution that the ALNS meta-heuristic can then improve.
"""

from __future__ import annotations

from collections import defaultdict

from src.config import MAX_STOPS, WAREHOUSE
from src.distance import DistanceMatrix, route_distance
from src.models import Order, Route, Solution, Stop, Truck
from src.solvers.base import BaseSolver


class GreedySolver(BaseSolver):
    name = "greedy"

    def solve(
        self,
        orders: list[Order],
        trucks: list[Truck],
        distances: DistanceMatrix,
    ) -> Solution:
        max_cap = max(t.capacity_kg for t in trucks)

        routes = self._bin_pack(orders, max_cap, distances)
        routes = self._clarke_wright_merge(routes, distances, max_cap)
        for r in routes:
            self._tsp_optimise(r, distances)
        self._assign_trucks(routes, trucks)

        sol = Solution(routes=routes, solver_name=self.name)
        sol.recalc_distance(distances)
        return sol

    # -- phase 1: bin-pack -----------------------------------------------

    def _bin_pack(
        self,
        orders: list[Order],
        max_cap: float,
        distances: DistanceMatrix,
    ) -> list[Route]:
        # group orders by destination
        by_dest: dict[str, list[Order]] = defaultdict(list)
        for o in orders:
            by_dest[o.destination].append(o)

        # build demand chunks; split any destination that exceeds max_cap
        chunks: list[tuple[str, float, list[Order]]] = []
        for city, ords in by_dest.items():
            total_wt = sum(o.total_weight_kg for o in ords)
            if total_wt <= max_cap:
                chunks.append((city, total_wt, ords))
            else:
                chunks.extend(self._split_destination(city, ords, max_cap))
        chunks.sort(key=lambda x: x[1], reverse=True)  # FFD

        routes: list[Route] = []
        rid = 0

        for city, wt, ords in chunks:
            # try placing into an existing route (geographic proximity)
            best_route, best_extra = None, float("inf")
            for route in routes:
                if route.num_stops >= MAX_STOPS:
                    continue
                if route.total_weight_kg + wt > max_cap:
                    continue
                last = route.stops[-1].city if route.stops else WAREHOUSE
                extra = distances.get((last, city), float("inf"))
                if extra < best_extra:
                    best_extra, best_route = extra, route

            if best_route is not None:
                stop = Stop(city=city)
                for o in ords:
                    stop.add(o)
                best_route.stops.append(stop)
                best_route.recalc_weight()
            else:
                rid += 1
                new = Route(route_id=rid)
                stop = Stop(city=city)
                for o in ords:
                    stop.add(o)
                new.stops.append(stop)
                new.recalc_weight()
                routes.append(new)

        return routes

    @staticmethod
    def _split_destination(
        city: str,
        orders: list[Order],
        max_cap: float,
    ) -> list[tuple[str, float, list[Order]]]:
        """Break an oversized destination into truck-sized chunks."""
        sorted_ords = sorted(orders, key=lambda o: o.total_weight_kg, reverse=True)
        chunks: list[tuple[str, float, list[Order]]] = []
        cur: list[Order] = []
        cur_wt = 0.0
        for o in sorted_ords:
            if cur_wt + o.total_weight_kg > max_cap and cur:
                chunks.append((city, cur_wt, cur))
                cur, cur_wt = [], 0.0
            cur.append(o)
            cur_wt += o.total_weight_kg
        if cur:
            chunks.append((city, cur_wt, cur))
        return chunks

    # -- phase 2: Clarke-Wright savings merge ----------------------------

    @staticmethod
    def _clarke_wright_merge(
        routes: list[Route],
        distances: DistanceMatrix,
        max_cap: float,
    ) -> list[Route]:
        # compute all pairwise savings
        savings: list[tuple[float, int, int]] = []
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                ri, rj = routes[i], routes[j]
                if ri.total_weight_kg + rj.total_weight_kg > max_cap:
                    continue
                if ri.num_stops + rj.num_stops > MAX_STOPS:
                    continue
                last_i = ri.stops[-1].city if ri.stops else WAREHOUSE
                first_j = rj.stops[0].city if rj.stops else WAREHOUSE
                s = (
                    distances.get((WAREHOUSE, last_i), 0)
                    + distances.get((WAREHOUSE, first_j), 0)
                    - distances.get((last_i, first_j), 0)
                )
                if s > 0:
                    savings.append((s, i, j))
        savings.sort(reverse=True)

        merged_into: dict[int, int] = {}
        for _, i, j in savings:
            while i in merged_into:
                i = merged_into[i]
            while j in merged_into:
                j = merged_into[j]
            if i == j:
                continue
            ri, rj = routes[i], routes[j]
            if ri.total_weight_kg + rj.total_weight_kg > max_cap:
                continue
            if ri.num_stops + rj.num_stops > MAX_STOPS:
                continue
            ri.stops.extend(rj.stops)
            ri.recalc_weight()
            merged_into[j] = i

        surviving = [routes[k] for k in range(len(routes)) if k not in merged_into]
        for idx, r in enumerate(surviving, 1):
            r.route_id = idx
        return surviving

    # -- phase 3: TSP (NN + 2-opt) --------------------------------------

    def _tsp_optimise(self, route: Route, distances: DistanceMatrix) -> None:
        cities = [s.city for s in route.stops]
        if len(cities) <= 1:
            route.city_sequence = cities
            route.total_distance_mi = route_distance(cities, distances)
            return

        # nearest-neighbour construction
        unvisited = set(cities)
        tour: list[str] = []
        cur = WAREHOUSE
        while unvisited:
            nxt = min(unvisited, key=lambda c: distances.get((cur, c), float("inf")))
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt

        # 2-opt improvement
        tour = self._two_opt(tour, distances)

        # re-order stops to match tour
        city_to_stop = {s.city: s for s in route.stops}
        route.stops = [city_to_stop[c] for c in tour]
        route.city_sequence = tour
        route.total_distance_mi = route_distance(tour, distances)

    @staticmethod
    def _two_opt(tour: list[str], distances: DistanceMatrix) -> list[str]:
        best = list(tour)
        best_d = route_distance(best, distances)
        improved = True
        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    cand = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                    d = route_distance(cand, distances)
                    if d < best_d - 0.01:
                        best, best_d = cand, d
                        improved = True
                        break
                if improved:
                    break
        return best

    # -- phase 4: truck assignment ---------------------------------------

    @staticmethod
    def _assign_trucks(routes: list[Route], trucks: list[Truck]) -> None:
        """Spread routes across the fleet so deliveries run in parallel.

        Heaviest routes are assigned first.  Among trucks that have
        capacity, the one with the fewest routes already assigned wins
        (ties broken by hourly rate, lower is better).
        """
        routes_by_wt = sorted(routes, key=lambda r: r.total_weight_kg, reverse=True)
        usage: dict[str, int] = {t.truck_id: 0 for t in trucks}

        for route in routes_by_wt:
            feasible = [t for t in trucks if t.capacity_kg >= route.total_weight_kg]
            if not feasible:
                feasible = [max(trucks, key=lambda t: t.capacity_kg)]

            # prefer the truck with fewest assigned routes, then cheapest rate
            best = min(feasible, key=lambda t: (usage[t.truck_id], t.hourly_rate))
            route.truck = best
            usage[best.truck_id] += 1

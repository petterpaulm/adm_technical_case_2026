"""
Adaptive Large Neighbourhood Search (ALNS) for the Heterogeneous CVRP.

Based on Ropke & Pisinger (2006) "An Adaptive Large Neighborhood Search
Heuristic for the Pickup and Delivery Problem with Time Windows",
adapted for the capacitated fleet delivery problem with HOS constraints.

The algorithm repeatedly *destroys* part of a solution (removes orders
from their routes) then *repairs* it (re-inserts them in better
positions).  Operator probabilities adapt over time based on which
operators have historically yielded improvements.

Destroy operators
-----------------
  ① Random removal    -- diversification baseline
  ② Worst removal     -- evict the costliest-to-serve orders
  ③ Shaw removal      -- evict geographically related orders
  ④ Route removal     -- remove an entire under-performing route

Repair operators
----------------
  ① Greedy insertion  -- cheapest feasible position
  ② Regret-2 insert   -- maximise second-best regret
  ③ Perturbation ins  -- greedy + random noise (exploration)

Acceptance criterion
--------------------
  Simulated Annealing -- accept worse solutions with probability
  exp(−Δ / T), temperature cools geometrically each iteration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from src.config import (
    ALNS_COOLING,
    ALNS_DESTROY_RANGE,
    ALNS_ITERATIONS,
    ALNS_SEGMENT,
    ALNS_START_TEMP,
    MAX_STOPS,
    SEED,
    WAREHOUSE,
)
from src.distance import DistanceMatrix, route_distance
from src.models import Order, Route, Solution, Stop, Truck
from src.solvers.base import BaseSolver
from src.solvers.greedy import GreedySolver

random.seed(SEED)


# -- operator tracking ---------------------------------------------------


@dataclass
class _OpStats:
    """Track how well an operator is performing within a segment."""

    score: float = 0.0
    uses: int = 0
    weight: float = 1.0

    def update_weight(self, reaction_factor: float = 0.1) -> None:
        """Adjust weight proportionally to average score per use."""
        if self.uses > 0:
            avg = self.score / self.uses
            self.weight = max(0.05, self.weight * (1 - reaction_factor) + reaction_factor * avg)
        self.score = 0.0
        self.uses = 0


# scoring constants (how much credit an operator gets)
_SIGMA_BEST = 33  # new global best
_SIGMA_BETTER = 9  # improved current (not global best)
_SIGMA_ACCEPT = 1  # accepted but not better


# -- the solver ----------------------------------------------------------


class ALNSSolver(BaseSolver):
    """
    Adaptive Large Neighbourhood Search for heterogeneous fleet VRP.

    Parameters
    ----------
    iterations : int
        Total destroy-repair cycles to execute.
    start_temp : float
        SA starting temperature.
    cooling : float
        Geometric cooling factor applied every iteration.
    segment : int
        Number of iterations between weight updates.
    """

    name = "alns"

    def __init__(
        self,
        iterations: int = ALNS_ITERATIONS,
        start_temp: float = ALNS_START_TEMP,
        cooling: float = ALNS_COOLING,
        segment: int = ALNS_SEGMENT,
    ) -> None:
        self.iterations = iterations
        self.start_temp = start_temp
        self.cooling = cooling
        self.segment = segment

        # operator registries (populated in solve)
        self._destroy_ops: list[tuple[str, callable]] = []
        self._repair_ops: list[tuple[str, callable]] = []
        self._d_stats: dict[str, _OpStats] = {}
        self._r_stats: dict[str, _OpStats] = {}

    # -- public API ------------------------------------------------------

    def solve(
        self,
        orders: list[Order],
        trucks: list[Truck],
        distances: DistanceMatrix,
    ) -> Solution:
        max_cap = max(t.capacity_kg for t in trucks)

        # warm-start with the greedy solver
        current = GreedySolver().solve(orders, trucks, distances)
        best = current.deep_copy()

        # stash for use in operators
        self._distances = distances
        self._trucks = trucks
        self._max_cap = max_cap

        # register operators
        self._destroy_ops = [
            ("random", self._destroy_random),
            ("worst", self._destroy_worst),
            ("shaw", self._destroy_shaw),
            ("route", self._destroy_route),
        ]
        self._repair_ops = [
            ("greedy", self._repair_greedy),
            ("regret2", self._repair_regret2),
            ("perturb", self._repair_perturb),
        ]
        self._d_stats = {n: _OpStats() for n, _ in self._destroy_ops}
        self._r_stats = {n: _OpStats() for n, _ in self._repair_ops}

        temp = self.start_temp
        no_improve_count = 0

        for it in range(1, self.iterations + 1):
            candidate = current.deep_copy()

            # select operators via roulette-wheel
            d_name, d_fn = self._roulette(self._destroy_ops, self._d_stats)
            r_name, r_fn = self._roulette(self._repair_ops, self._r_stats)

            # destroy -> repair
            removed = d_fn(candidate)

            # rebuild city sequences after destroy — the destroy operators
            # remove stops but do not update city_sequence, leaving it stale.
            for r in candidate.routes:
                r.city_sequence = [s.city for s in r.stops]

            r_fn(candidate, removed)

            # clean up empty routes and recompute distances
            candidate.routes = [r for r in candidate.routes if r.stops]
            for r in candidate.routes:
                r.recalc_weight()
            self._reassign_trucks(candidate)
            candidate.recalc_distance(distances)

            # fast cost proxy: total distance (driver costs scale similarly)
            c_cost = candidate.total_distance
            cur_cost = current.total_distance
            best_cost = best.total_distance
            delta = c_cost - cur_cost

            # SA acceptance
            if delta < 0 or (temp > 1e-9 and random.random() < math.exp(-delta / temp)):
                if c_cost < best_cost:
                    best = candidate.deep_copy()
                    self._reward(d_name, r_name, _SIGMA_BEST)
                    no_improve_count = 0
                elif c_cost < cur_cost:
                    self._reward(d_name, r_name, _SIGMA_BETTER)
                    no_improve_count = 0
                else:
                    self._reward(d_name, r_name, _SIGMA_ACCEPT)
                    no_improve_count += 1
                current = candidate
            else:
                no_improve_count += 1

            temp *= self.cooling

            # adaptive weight update every segment
            if it % self.segment == 0:
                for s in self._d_stats.values():
                    s.update_weight()
                for s in self._r_stats.values():
                    s.update_weight()

        # attach metadata
        best.solver_name = self.name
        best.metadata = {
            "iterations": self.iterations,
            "final_temp": round(temp, 4),
            "operator_weights": {
                "destroy": {n: round(s.weight, 3) for n, s in self._d_stats.items()},
                "repair": {n: round(s.weight, 3) for n, s in self._r_stats.items()},
            },
        }
        return best

    # -- operator selection ----------------------------------------------

    @staticmethod
    def _roulette(ops, stats) -> tuple[str, callable]:
        weights = [stats[n].weight for n, _ in ops]
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0.0
        for (name, fn), w in zip(ops, weights):
            cumulative += w
            if r <= cumulative:
                stats[name].uses += 1
                return name, fn
        name, fn = ops[-1]
        stats[name].uses += 1
        return name, fn

    def _reward(self, d_name: str, r_name: str, sigma: float) -> None:
        self._d_stats[d_name].score += sigma
        self._r_stats[r_name].score += sigma

    # -- destroy operators -----------------------------------------------

    def _removal_count(self, solution: Solution) -> int:
        """How many stops to rip out this iteration."""
        total_stops = sum(r.num_stops for r in solution.routes)
        lo = max(1, int(total_stops * ALNS_DESTROY_RANGE[0]))
        hi = max(lo + 1, int(total_stops * ALNS_DESTROY_RANGE[1]))
        return random.randint(lo, min(hi, total_stops))

    def _destroy_random(self, sol: Solution) -> list[Order]:
        """Remove k random orders from random routes."""
        k = self._removal_count(sol)
        all_orders = [(r, s, o) for r in sol.routes for s in r.stops for o in s.orders]
        if not all_orders:
            return []
        chosen = random.sample(all_orders, min(k, len(all_orders)))
        removed: list[Order] = []
        for route, stop, order in chosen:
            stop.remove(order)
            removed.append(order)
            if not stop.orders:
                route.stops.remove(stop)
        return removed

    def _destroy_worst(self, sol: Solution) -> list[Order]:
        """Remove the orders that contribute the most to route distance."""
        k = self._removal_count(sol)
        dist = self._distances
        scored: list[tuple[float, Route, Stop, Order]] = []

        for route in sol.routes:
            seq = route.city_sequence or [s.city for s in route.stops]
            base_dist = route_distance(seq, dist)
            for stop in route.stops:
                for order in stop.orders:
                    # marginal cost: distance without this stop's city
                    reduced = [c for c in seq if c != stop.city]
                    new_dist = route_distance(reduced, dist) if reduced else 0.0
                    saving = base_dist - new_dist
                    # higher saving = this order is costly to serve
                    scored.append((saving, route, stop, order))

        scored.sort(key=lambda x: x[0], reverse=True)
        removed: list[Order] = []
        seen: set[int] = set()
        for _, route, stop, order in scored:
            if len(removed) >= k:
                break
            if order.idx in seen:
                continue
            seen.add(order.idx)
            stop.remove(order)
            removed.append(order)
            if not stop.orders:
                route.stops.remove(stop)
        return removed

    def _destroy_shaw(self, sol: Solution) -> list[Order]:
        """Remove geographically related orders (nearby destinations)."""
        k = self._removal_count(sol)
        dist = self._distances
        all_orders = [(r, s, o) for r in sol.routes for s in r.stops for o in s.orders]
        if not all_orders:
            return []

        # pick a seed order at random
        seed_r, seed_s, seed_o = random.choice(all_orders)
        seed_city = seed_o.destination

        # score every other order by relatedness (distance + weight similarity)
        relatedness: list[tuple[float, Route, Stop, Order]] = []
        for r, s, o in all_orders:
            if o.idx == seed_o.idx:
                continue
            geo = dist.get((seed_city, o.destination), 9999)
            wt_diff = abs(seed_o.total_weight_kg - o.total_weight_kg)
            score = geo + wt_diff * 0.5  # lower = more related
            relatedness.append((score, r, s, o))
        relatedness.sort(key=lambda x: x[0])

        removed: list[Order] = [seed_o]
        seed_s.remove(seed_o)
        if not seed_s.orders:
            seed_r.stops.remove(seed_s)

        for _, route, stop, order in relatedness:
            if len(removed) >= k:
                break
            if order not in stop.orders:
                continue
            stop.remove(order)
            removed.append(order)
            if not stop.orders:
                route.stops.remove(stop)
        return removed

    def _destroy_route(self, sol: Solution) -> list[Order]:
        """Remove an entire under-performing route."""
        if not sol.routes:
            return []
        # pick the route with the worst distance-to-weight ratio
        worst = max(
            sol.routes,
            key=lambda r: r.total_distance_mi / max(r.total_weight_kg, 1),
        )
        removed = list(worst.all_orders)
        sol.routes.remove(worst)
        return removed

    # -- repair operators ------------------------------------------------

    def _best_insertion_cost(
        self,
        order: Order,
        sol: Solution,
    ) -> list[tuple[float, Route, int]]:
        """
        For a given order, find the cheapest feasible insertion in every
        route.  Returns a sorted list of (cost_delta, route, position).
        """
        dist = self._distances
        options: list[tuple[float, Route, int]] = []

        for route in sol.routes:
            if route.total_weight_kg + order.total_weight_kg > self._max_cap:
                continue

            seq = route.city_sequence or [s.city for s in route.stops]
            base_dist = route_distance(seq, dist)
            city = order.destination

            # existing stop for this city?
            for i, stop in enumerate(route.stops):
                if stop.city == city:
                    options.append((0.0, route, i))
                    break
            else:
                if route.num_stops >= MAX_STOPS:
                    continue
                # try inserting city at every position in the sequence
                for pos in range(len(seq) + 1):
                    new_seq = [*seq[:pos], city, *seq[pos:]]
                    delta = route_distance(new_seq, dist) - base_dist
                    options.append((delta, route, pos))

        # also consider opening a brand-new route
        new_dist = dist.get((WAREHOUSE, order.destination), 0) + dist.get(
            (order.destination, WAREHOUSE), 0
        )
        options.append((new_dist, None, 0))  # None signals "new route"

        options.sort(key=lambda x: x[0])
        return options

    def _insert_order(
        self,
        order: Order,
        route: Route | None,
        pos: int,
        sol: Solution,
    ) -> None:
        """Physically insert an order into the solution."""
        if route is None:
            # open a new route
            new_id = max((r.route_id for r in sol.routes), default=0) + 1
            stop = Stop(city=order.destination)
            stop.add(order)
            new_route = Route(route_id=new_id, stops=[stop])
            new_route.city_sequence = [order.destination]
            new_route.recalc_weight()
            sol.routes.append(new_route)
            return

        # check if the city already has a stop on this route
        for stop in route.stops:
            if stop.city == order.destination:
                stop.add(order)
                route.recalc_weight()
                return

        # insert a brand-new stop at the given position
        stop = Stop(city=order.destination)
        stop.add(order)
        seq = route.city_sequence or [s.city for s in route.stops]
        route.stops.insert(pos, stop)
        route.city_sequence = [*seq[:pos], order.destination, *seq[pos:]]
        route.recalc_weight()

    def _repair_greedy(self, sol: Solution, removed: list[Order]) -> None:
        """Insert each removed order at its cheapest feasible position."""
        random.shuffle(removed)
        for order in removed:
            options = self._best_insertion_cost(order, sol)
            if options:
                _cost, route, pos = options[0]
                self._insert_order(order, route, pos, sol)

    def _repair_regret2(self, sol: Solution, removed: list[Order]) -> None:
        """
        Regret-2 heuristic: always insert the order whose difference
        between best and second-best insertion cost is largest.
        This prevents painting ourselves into a corner.
        """
        pool = list(removed)
        while pool:
            best_regret, best_order, best_opt = -float("inf"), None, None
            for order in pool:
                options = self._best_insertion_cost(order, sol)
                if len(options) >= 2:
                    regret = options[1][0] - options[0][0]
                elif options:
                    regret = options[0][0]
                else:
                    regret = 1e9
                if regret > best_regret:
                    best_regret = regret
                    best_order = order
                    best_opt = options[0] if options else None

            if best_order is None:
                break
            pool.remove(best_order)
            if best_opt:
                self._insert_order(best_order, best_opt[1], best_opt[2], sol)

    def _repair_perturb(self, sol: Solution, removed: list[Order]) -> None:
        """Greedy insertion with random noise on costs (exploration)."""
        random.shuffle(removed)
        for order in removed:
            options = self._best_insertion_cost(order, sol)
            # add Gaussian noise to the first few options
            noised = [
                (cost + random.gauss(0, max(abs(cost) * 0.3, 10)), route, pos)
                for cost, route, pos in options[:10]
            ]
            noised.sort(key=lambda x: x[0])
            if noised:
                _, route, pos = noised[0]
                self._insert_order(order, route, pos, sol)

    # -- helpers ---------------------------------------------------------

    def _reassign_trucks(self, sol: Solution) -> None:
        """Load-balanced re-assignment: spread routes across feasible trucks."""
        usage: dict[str, int] = {t.truck_id: 0 for t in self._trucks}
        for route in sorted(sol.routes, key=lambda r: r.total_weight_kg, reverse=True):
            feasible = [t for t in self._trucks if t.capacity_kg >= route.total_weight_kg]
            if not feasible:
                feasible = [max(self._trucks, key=lambda t: t.capacity_kg)]
            best = min(feasible, key=lambda t: (usage[t.truck_id], t.hourly_rate))
            route.truck = best
            usage[best.truck_id] += 1

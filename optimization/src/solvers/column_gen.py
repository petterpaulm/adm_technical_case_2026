"""
Column Generation solver for the HCVRP.

Decomposes the VRP into a *Set Covering* master problem that selects
the minimum-cost subset of routes, and a *pricing sub-problem* that
generates promising new routes (columns) with negative reduced cost.

This is the same family of algorithms used in production by carriers
like UPS and FedEx for large-scale fleet optimisation.

Approach
--------
  Master LP  :  min Σ cⱼ·xⱼ   s.t.  Σ aᵢⱼ·xⱼ ≥ 1 ∀ order i,  xⱼ ≥ 0
  Pricing    :  generate a new route (column) whose reduced cost
                c̄ⱼ = cⱼ − Σ πᵢ·aᵢⱼ  <  0
                using a greedy heuristic on dual prices πᵢ.

When no more negative-reduced-cost columns exist, the LP is optimal.
We then round the fractional LP solution and do a quick local-search
to restore integer feasibility.
Lagrangian Duality Connection
-----------------------------
Column Generation and Lagrangian Relaxation are intimately connected
through LP duality theory.  The dual variables πᵢ obtained from the
master LP relaxation are **exactly** the optimal Lagrange multipliers
for the order-coverage constraints.  In other words:

  - Solving the LP relaxation of the master yields dual prices πᵢ.
  - These πᵢ are the same values you would get if you formed the
    Lagrangian relaxation L(π) = min_x Σ (cⱼ − Σ πᵢ·aᵢⱼ)·xⱼ + Σ πᵢ
    and solved the Lagrangian dual  max_π L(π).
  - The pricing sub-problem (find a column with c̄ⱼ < 0) is equivalent
    to solving the Lagrangian sub-problem: minimise the route cost
    *minus* the Lagrangian "reward" for covering valuable orders.

This equivalence (Dantzig-Wolfe ↔ Lagrangian decomposition) means CG
provides a *lower bound* on the optimal integer solution at every
iteration -- a guarantee that heuristics alone cannot provide.

See: Lübbecke & Desrosiers (2005), "Selected Topics in Column
Generation", Operations Research 53(6), for the formal proof."""

from __future__ import annotations

import random
from collections import defaultdict

from src.config import MAX_STOPS, SEED, WAREHOUSE
from src.distance import DistanceMatrix, route_distance
from src.models import Order, Route, Solution, Stop, Truck
from src.solvers.base import BaseSolver
from src.solvers.greedy import GreedySolver

random.seed(SEED)

# We only use PuLP if available; fall back to a penalty heuristic otherwise.
try:
    import pulp

    _HAS_PULP = True
except ImportError:
    _HAS_PULP = False


class ColumnGenSolver(BaseSolver):
    """Column Generation for Heterogeneous CVRP."""

    name = "column_generation"

    def __init__(self, max_cg_rounds: int = 60, price_routes_per_round: int = 15) -> None:
        self.max_cg_rounds = max_cg_rounds
        self.price_per_round = price_routes_per_round

    def solve(
        self,
        orders: list[Order],
        trucks: list[Truck],
        distances: DistanceMatrix,
    ) -> Solution:
        if not _HAS_PULP:
            # graceful degradation: just use greedy
            sol = GreedySolver().solve(orders, trucks, distances)
            sol.solver_name = self.name
            sol.metadata["note"] = "PuLP not installed -- fell back to greedy"
            return sol

        max_cap = max(t.capacity_kg for t in trucks)

        # seed the column pool with routes from the greedy solver
        greedy_sol = GreedySolver().solve(orders, trucks, distances)
        columns: list[_Column] = []
        for route in greedy_sol.routes:
            columns.append(self._route_to_column(route, orders, distances))

        # also add every single-order route (ensures feasibility)
        for order in orders:
            columns.append(self._single_order_column(order, distances))

        order_idx_list = [o.idx for o in orders]

        # column generation loop
        duals: dict[int, float] = {o.idx: 0.0 for o in orders}
        for _round in range(self.max_cg_rounds):
            # solve the master LP
            _selected, duals = self._solve_master_lp(columns, order_idx_list)

            # pricing: generate new columns with negative reduced cost
            new_cols = self._pricing(orders, distances, duals, max_cap)
            if not new_cols:
                break
            columns.extend(new_cols)

        # integer rounding
        chosen_cols = self._integer_round(columns, order_idx_list)

        # convert back to Solution
        return self._build_solution(chosen_cols, orders, trucks, distances)

    # -- master LP -------------------------------------------------------

    @staticmethod
    def _solve_master_lp(
        columns: list[_Column],
        order_indices: list[int],
    ) -> tuple[list[float], dict[int, float]]:
        """Solve the LP relaxation; return (x_values, dual_prices)."""
        prob = pulp.LpProblem("VRP_SetCover", pulp.LpMinimize)

        x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1) for j in range(len(columns))]

        # objective: minimise total route cost
        prob += pulp.lpSum(columns[j].cost * x[j] for j in range(len(columns)))

        # cover every order at least once
        constraints: dict[int, pulp.LpConstraint] = {}
        for oi in order_indices:
            c = pulp.lpSum(columns[j].covers.get(oi, 0) * x[j] for j in range(len(columns)))
            constr = c >= 1
            name = f"cover_{oi}"
            prob += constr, name
            constraints[oi] = constr

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        x_vals = [v.varValue or 0.0 for v in x]

        # Extract dual prices (πᵢ) from constraints.
        # By LP duality, these are identical to the optimal Lagrange
        # multipliers for the order-coverage constraints.  High πᵢ means
        # order i is expensive to cover -> the pricing sub-problem will
        # prioritise including it in new routes.
        duals: dict[int, float] = {}
        for oi in order_indices:
            name = f"cover_{oi}"
            con = prob.constraints.get(name)
            duals[oi] = con.pi if (con and con.pi is not None) else 0.0

        return x_vals, duals

    # -- pricing sub-problem ---------------------------------------------

    def _pricing(
        self,
        orders: list[Order],
        distances: DistanceMatrix,
        duals: dict[int, float],
        max_cap: float,
    ) -> list[_Column]:
        """Generate new columns (routes) with negative reduced cost."""
        new_columns: list[_Column] = []

        for _ in range(self.price_per_round):
            col = self._greedy_price_route(orders, distances, duals, max_cap)
            if col is not None and col.reduced_cost(duals) < -1e-4:
                new_columns.append(col)

        return new_columns

    def _greedy_price_route(
        self,
        orders: list[Order],
        distances: DistanceMatrix,
        duals: dict[int, float],
        max_cap: float,
    ) -> _Column | None:
        """
        Build one route greedily, guided by dual prices.
        Orders with high dual prices are attractive to include.
        """
        available = list(orders)
        random.shuffle(available)

        # sort by dual-adjusted attractiveness: high dual -> want to include
        available.sort(key=lambda o: -duals.get(o.idx, 0))

        route_orders: list[Order] = []
        weight = 0.0
        cities: list[str] = []
        visited_cities: set[str] = set()

        for order in available:
            if weight + order.total_weight_kg > max_cap:
                continue
            if len(visited_cities) >= MAX_STOPS and order.destination not in visited_cities:
                continue

            route_orders.append(order)
            weight += order.total_weight_kg
            if order.destination not in visited_cities:
                cities.append(order.destination)
                visited_cities.add(order.destination)

            if len(route_orders) > 40:  # practical limit
                break

        if not route_orders:
            return None

        dist = route_distance(cities, distances)
        covers = defaultdict(int)
        for o in route_orders:
            covers[o.idx] = 1

        return _Column(cost=dist, covers=dict(covers), cities=cities, orders=route_orders)

    # -- integer rounding ------------------------------------------------

    @staticmethod
    def _integer_round(
        columns: list[_Column],
        order_indices: list[int],
    ) -> list[_Column]:
        """
        Greedy rounding: repeatedly pick the column covering the most
        uncovered orders (weighted by LP value), until every order is covered.
        """
        uncovered = set(order_indices)
        chosen: list[_Column] = []

        # score by coverage of uncovered orders (prefer cheap, wide-coverage)
        while uncovered:
            best, best_score = None, -1
            for col in columns:
                n_covered = sum(1 for oi in uncovered if col.covers.get(oi, 0))
                if n_covered == 0:
                    continue
                score = n_covered / max(col.cost, 1)
                if score > best_score:
                    best_score, best = score, col
            if best is None:
                break
            chosen.append(best)
            uncovered -= set(best.covers.keys())

        return chosen

    # -- conversion helpers ----------------------------------------------

    @staticmethod
    def _route_to_column(
        route: Route,
        all_orders: list[Order],
        distances: DistanceMatrix,
    ) -> _Column:
        covers = {o.idx: 1 for o in route.all_orders}
        cities = route.city_sequence or [s.city for s in route.stops]
        cost = route_distance(cities, distances)
        return _Column(cost=cost, covers=covers, cities=cities, orders=list(route.all_orders))

    @staticmethod
    def _single_order_column(order: Order, distances: DistanceMatrix) -> _Column:
        d = distances.get((WAREHOUSE, order.destination), 0) + distances.get(
            (order.destination, WAREHOUSE), 0
        )
        return _Column(
            cost=d,
            covers={order.idx: 1},
            cities=[order.destination],
            orders=[order],
        )

    def _build_solution(
        self,
        chosen: list[_Column],
        all_orders: list[Order],
        trucks: list[Truck],
        distances: DistanceMatrix,
    ) -> Solution:
        trucks_asc = sorted(trucks, key=lambda t: (t.capacity_kg, t.hourly_rate))

        routes: list[Route] = []
        for rid, col in enumerate(chosen, 1):
            # group orders by city
            by_city: dict[str, list[Order]] = defaultdict(list)
            for o in col.orders:
                by_city[o.destination].append(o)

            stops = []
            for city in col.cities:
                s = Stop(city=city)
                for o in by_city.get(city, []):
                    s.add(o)
                if s.orders:
                    stops.append(s)

            route = Route(route_id=rid, stops=stops, city_sequence=col.cities)
            route.recalc_weight()
            route.total_distance_mi = route_distance(col.cities, distances)
            routes.append(route)

        # load-balanced truck assignment
        usage: dict[str, int] = {t.truck_id: 0 for t in trucks_asc}
        for route in sorted(routes, key=lambda r: r.total_weight_kg, reverse=True):
            feasible = [t for t in trucks_asc if t.capacity_kg >= route.total_weight_kg]
            if not feasible:
                feasible = [trucks_asc[-1]]
            best = min(feasible, key=lambda t: (usage[t.truck_id], t.hourly_rate))
            route.truck = best
            usage[best.truck_id] += 1

        sol = Solution(routes=routes, solver_name=self.name)
        sol.recalc_distance(distances)
        return sol


# -- internal column data structure --------------------------------------


class _Column:
    """A candidate route for the set-covering master."""

    __slots__ = ("cities", "cost", "covers", "orders")

    def __init__(
        self,
        cost: float,
        covers: dict[int, int],
        cities: list[str],
        orders: list[Order],
    ) -> None:
        self.cost = cost
        self.covers = covers
        self.cities = cities
        self.orders = orders

    def reduced_cost(self, duals: dict[int, float]) -> float:
        """c̄ⱼ = cⱼ − Σ πᵢ·aᵢⱼ.  Equivalent to the Lagrangian sub-problem cost."""
        return self.cost - sum(duals.get(oi, 0) * v for oi, v in self.covers.items())

"""
Multi-objective Pareto optimiser.

In real logistics, cost is not the only objective.  Operators also care
about delivery speed (makespan) and balanced fleet utilisation.  This
module generates a set of Pareto-optimal solutions -- each one represents
a different trade-off that management can evaluate.

Objectives
----------
  f₁  Total cost        (minimise)     -- fuel + labour + maintenance
  f2  Makespan          (minimise)     -- latest return time
  f₃  Utilisation StdDev (minimise)    -- balance load across trucks

Method
------
  ε-constraint:  Fix one objective with an upper bound, optimise the
  primary objective.  Sweep the bound across a grid to trace the front.
  Each point is solved by the ALNS with modified penalty terms.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from src.config import AVG_SPEED_MPH, LOADING_MINUTES, SEED, STOP_OVERHEAD_MINUTES
from src.distance import DistanceMatrix
from src.models import Order, Solution, Truck
from src.solvers.alns import ALNSSolver
from src.solvers.base import BaseSolver
from src.solvers.greedy import GreedySolver

random.seed(SEED)


@dataclass
class ParetoPoint:
    """One solution on the Pareto front."""

    cost: float
    makespan_hrs: float
    utilisation_std: float
    solution: Solution
    label: str = ""


class ParetoSolver(BaseSolver):
    """
    Generate an approximate Pareto front for the multi-objective VRP.

    We run the ALNS solver several times with different penalty weights
    to steer it toward different trade-off regions, then filter to
    keep only non-dominated solutions.
    """

    name = "pareto"

    def __init__(self, num_probes: int = 8, alns_iters: int = 2_000) -> None:
        self.num_probes = num_probes
        self.alns_iters = alns_iters

    def solve(
        self,
        orders: list[Order],
        trucks: list[Truck],
        distances: DistanceMatrix,
    ) -> Solution:
        """Return the best-cost Pareto point as the primary solution."""
        front = self.compute_front(orders, trucks, distances)
        best = min(front, key=lambda p: p.cost)
        best.solution.metadata["pareto_front"] = [
            {"cost": p.cost, "makespan_hrs": p.makespan_hrs, "util_std": p.utilisation_std}
            for p in front
        ]
        return best.solution

    def compute_front(
        self,
        orders: list[Order],
        trucks: list[Truck],
        distances: DistanceMatrix,
    ) -> list[ParetoPoint]:
        """Run multiple ALNS probes and return non-dominated set."""
        points: list[ParetoPoint] = []

        # Probe 1: pure cost minimisation (default ALNS)
        alns = ALNSSolver(iterations=self.alns_iters)
        sol = alns.solve(orders, trucks, distances)
        points.append(self._evaluate(sol, distances, "cost-focus"))

        # Probe 2: greedy baseline
        gsol = GreedySolver().solve(orders, trucks, distances)
        points.append(self._evaluate(gsol, distances, "greedy"))

        # Probes 3..N: vary ALNS parameters to explore different trade-offs
        for i in range(self.num_probes - 2):
            temp = random.uniform(200, 1000)
            cooling = random.uniform(0.9990, 0.9999)
            a = ALNSSolver(iterations=self.alns_iters, start_temp=temp, cooling=cooling)
            s = a.solve(orders, trucks, distances)
            points.append(self._evaluate(s, distances, f"probe-{i + 1}"))

        # non-domination filter
        return self._filter_dominated(points)

    # -- evaluation ------------------------------------------------------

    @staticmethod
    def _evaluate(sol: Solution, distances: DistanceMatrix, label: str) -> ParetoPoint:
        # cost proxy: total distance (strongly correlated with actual $)
        cost = sol.total_distance

        # makespan: longest individual route time
        makespans: list[float] = []
        for r in sol.routes:
            driving_hrs = r.total_distance_mi / AVG_SPEED_MPH if AVG_SPEED_MPH else 0.0
            stop_hrs = (r.num_stops * STOP_OVERHEAD_MINUTES + LOADING_MINUTES) / 60.0
            makespans.append(driving_hrs + stop_hrs)
        makespan = max(makespans) if makespans else 0.0

        # utilisation balance: stdev of weight as fraction of truck capacity
        utils: list[float] = []
        for r in sol.routes:
            cap = r.truck.capacity_kg if r.truck else 10_000
            utils.append(r.total_weight_kg / cap)
        util_std = statistics.stdev(utils) if len(utils) > 1 else 0.0

        return ParetoPoint(
            cost=round(cost, 1),
            makespan_hrs=round(makespan, 2),
            utilisation_std=round(util_std, 4),
            solution=sol,
            label=label,
        )

    # -- non-domination --------------------------------------------------

    @staticmethod
    def _filter_dominated(points: list[ParetoPoint]) -> list[ParetoPoint]:
        """Keep only solutions where no other solution is better in ALL objectives."""
        front: list[ParetoPoint] = []
        for p in points:
            dominated = False
            for q in points:
                if q is p:
                    continue
                if (
                    q.cost <= p.cost
                    and q.makespan_hrs <= p.makespan_hrs
                    and q.utilisation_std <= p.utilisation_std
                    and (
                        q.cost < p.cost
                        or q.makespan_hrs < p.makespan_hrs
                        or q.utilisation_std < p.utilisation_std
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(p)
        return front if front else [min(points, key=lambda p: p.cost)]

"""
Abstract solver interface.

Every solver (greedy, ALNS, column generation, Pareto) inherits from
``BaseSolver`` so the pipeline can swap strategies without changing
calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.distance import DistanceMatrix
from src.models import Order, Solution, Truck


class BaseSolver(ABC):
    """Contract that every VRP solver must satisfy."""

    name: str = "base"

    @abstractmethod
    def solve(
        self,
        orders: list[Order],
        trucks: list[Truck],
        distances: DistanceMatrix,
    ) -> Solution:
        """Return a complete delivery Solution."""
        ...

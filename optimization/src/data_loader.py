"""
CSV data loading, cleaning, and normalisation.

Reads the three source files (items, orders, trucks), joins item weights
onto order lines, normalises city names, and exposes a one-shot loader
that returns everything needed by the solvers.
"""

from __future__ import annotations

import re

import pandas as pd

from src.config import ITEMS_CSV, LBS_TO_KG, ORDERS_CSV, TRUCKS_CSV
from src.distance import DistanceMatrix, build_distance_matrix
from src.models import Item, Order, Truck

# -- helpers -------------------------------------------------------------


def _clean_city(raw: str) -> str:
    """Collapse whitespace and strip quotes so city names match config keys."""
    return re.sub(r"\s+", " ", str(raw).strip())


# -- loaders -------------------------------------------------------------


def load_items() -> dict[int, Item]:
    df = pd.read_csv(ITEMS_CSV)
    df.columns = df.columns.str.strip()
    items: dict[int, Item] = {}
    for _, row in df.iterrows():
        iid = int(row["ItemId"])
        wt = float(row["weight (pounds)"])
        items[iid] = Item(
            item_id=iid,
            weight_lbs=wt,
            weight_kg=wt * LBS_TO_KG,
            warehouse=row["warehouse origin"].strip(),
        )
    return items


def load_trucks() -> list[Truck]:
    df = pd.read_csv(TRUCKS_CSV, sep=";")
    df.columns = df.columns.str.strip()
    trucks: list[Truck] = []
    for _, row in df.iterrows():
        rate = float(str(row["Driver Hourly Rate"]).replace("$", "").strip())
        trucks.append(
            Truck(
                truck_id=str(row["Truck ID"]).zfill(6),
                driver=row["Driver Name"].strip(),
                experience_yrs=int(row["Years Experience"]),
                length_m=float(row["Truck Type (length in m)"]),
                capacity_kg=float(row["Weight Capacity (kg)"]),
                top_speed_kmh=float(row["Top Speed (km/h)"]),
                hourly_rate=rate,
                fuel_tank_gal=float(row["Fuel Tank Size (gallon)"]),
                spare_tires=int(row["Spare Tires"]),
                age_yrs=float(row["Truck Age (years)"]),
            )
        )
    return trucks


def load_orders(items: dict[int, Item]) -> list[Order]:
    df = pd.read_csv(ORDERS_CSV)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    orders: list[Order] = []
    for idx, row in df.iterrows():
        iid = int(row["Item"])
        units = int(row["Number of Units"])
        dest = _clean_city(str(row["Destination"]))
        company = str(row["Company"]).strip()

        item = items.get(iid)
        if item is None:
            raise ValueError(f"Row {idx}: unknown item {iid}")

        orders.append(
            Order(
                idx=int(idx),
                company=company,
                item_id=iid,
                units=units,
                destination=dest,
                unit_weight_kg=item.weight_kg,
                total_weight_kg=units * item.weight_kg,
            )
        )
    return orders


def unique_destinations(orders: list[Order]) -> list[str]:
    return sorted({o.destination for o in orders})


# -- one-shot convenience -----------------------------------------------


def load_all() -> tuple[dict[int, Item], list[Order], list[Truck], DistanceMatrix]:
    """Load items -> orders -> trucks -> distance matrix, all in one call."""
    items = load_items()
    orders = load_orders(items)
    trucks = load_trucks()
    dests = unique_destinations(orders)
    distances = build_distance_matrix(dests)
    return items, orders, trucks, distances

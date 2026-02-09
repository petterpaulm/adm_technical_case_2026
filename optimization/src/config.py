"""
Centralised configuration for the fleet delivery optimisation system.

Keeps every tuneable parameter, file path, and reference dataset (city
coordinates) in one place so the rest of the codebase stays clean.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# -- paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent  # CSVs sit one level above optimization/
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

ORDERS_CSV = DATA_DIR / "orders.csv"
ITEMS_CSV = DATA_DIR / "item_info.csv"
TRUCKS_CSV = DATA_DIR / "trucks.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -- unit conversions ----------------------------------------------------
LBS_TO_KG: float = 0.453592
KM_TO_MILES: float = 0.621371
MILES_TO_KM: float = 1.60934

# -- scheduling / HOS ---------------------------------------------------
FIRST_DEPARTURE = datetime(2026, 2, 3, 8, 0, 0)  # Mon Feb 3 2026, 08:00
MAX_WORK_HOURS: float = 14.0  # consecutive work before mandatory rest
MANDATORY_REST_HOURS: float = 10.0
LOADING_MINUTES: int = 30  # truck loading at warehouse
STOP_OVERHEAD_MINUTES: int = 35  # unload (20 min) + paperwork (15 min)

# -- fuel & cost ---------------------------------------------------------
FUEL_ECONOMY_MPG: dict[str, float] = {
    "16.06": 5.5,  # Class 8 heavy
    "14.82": 6.0,  # Class 7
    "13.98": 6.5,  # Class 6
    "11.57": 7.5,  # Class 5 medium
}
DIESEL_PRICE_PER_GALLON: float = 3.75
MAINTENANCE_PER_MILE: float = 0.15

# -- carbon / sustainability ---------------------------------------------
CO2_KG_PER_GALLON: float = 10.18  # EPA diesel emission factor (kg CO2/gal)

# -- Monte Carlo risk simulation -----------------------------------------
MC_TRIALS: int = 1_000
MC_TRAVEL_TIME_STD: float = 0.20   # ±20 % travel-time uncertainty
MC_BREAKDOWN_PROB: float = 0.05    # 5 % chance of truck breakdown per route
MC_BREAKDOWN_DELAY_HRS: float = 4.0  # mean delay when a breakdown occurs
MC_FUEL_PRICE_STD: float = 0.15   # ±15 % fuel-price volatility

# -- routing -------------------------------------------------------------
ROAD_FACTOR: float = 1.30  # haversine -> road-distance multiplier
AVG_SPEED_MPH: float = 55.0  # realistic loaded-truck highway average
MAX_STOPS: int = 8  # practical limit per route

# -- solver hyper-parameters ---------------------------------------------
SEED: int = 42
ALNS_ITERATIONS: int = 6_000
ALNS_SEGMENT: int = 100  # weight update every N iterations
ALNS_START_TEMP: float = 500.0  # simulated-annealing start temperature
ALNS_COOLING: float = 0.9997  # geometric cooling rate
ALNS_DESTROY_RANGE: tuple[float, float] = (0.10, 0.35)  # fraction of stops to remove

# -- warehouse -----------------------------------------------------------
WAREHOUSE = "Cincinnati, OH"
WAREHOUSE_COORDS = (39.1031, -84.5120)

# -- city coordinates ----------------------------------------------------
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Cincinnati, OH": (39.1031, -84.5120),
    "Albany, NY": (42.6526, -73.7562),
    "Baton Rouge, LA": (30.4515, -91.1871),
    "Boston, MA": (42.3601, -71.0589),
    "Chicago, IL": (41.8781, -87.6298),
    "Colorado Springs, CO": (38.8339, -104.8214),
    "Newport, KY": (39.0914, -84.4958),
    "Oshkosh, WI": (44.0247, -88.5426),
    "Philadelphia, PA": (39.9526, -75.1652),
    "Pittsburgh, PA": (40.4406, -79.9959),
    "Pittsburg, PA": (40.4406, -79.9959),
    "Portland, MA": (43.6591, -70.2568),
    "Portland, OR": (45.5152, -122.6784),
    "Santa Claus, IN": (38.1200, -86.9141),
    "Washington, DC": (38.9072, -77.0369),
    "Cambridge, MA": (42.3736, -71.1097),
    "Charlotte, NC": (35.2271, -80.8431),
    "Decatur, IL": (39.8403, -88.9548),
    "Lexington, SC": (33.9815, -81.2368),
    "Lincoln, NE": (40.8136, -96.7026),
    "Los Angeles, CA": (34.0522, -118.2437),
    "Orlando, FL": (28.5383, -81.3792),
    "Davis, CA": (38.5449, -121.7405),
    "Detroit, MI": (42.3314, -83.0458),
    "Hilton Head, SC": (32.2163, -80.7526),
    "Las Vegas, NV": (36.1699, -115.1398),
    "Lexington, KY": (38.0406, -84.5037),
    "Memphis, TN": (35.1495, -90.0490),
    "Memphis TN": (35.1495, -90.0490),
    "Tampa, FL": (27.9506, -82.4572),
    "Toronto, Canada": (43.6532, -79.3832),
    "Albany, ID": (42.0966, -114.2598),
    "Columbus, OH": (39.9612, -82.9988),
    "Forks, WA": (47.9504, -124.3855),
    "Miami, FL": (25.7617, -80.1918),
    "San Francisco, CA": (37.7749, -122.4194),
    "Seattle, WA": (47.6062, -122.3321),
    "Salt Lake City, UT": (40.7608, -111.8910),
    "Dallas, TX": (32.7767, -96.7970),
    "Grand Rapids, MI": (42.9634, -85.6681),
    "Key West, FL": (24.5551, -81.7800),
    "Nashville, TN": (36.1627, -86.7816),
    "Omaha, NE": (41.2565, -95.9345),
    "Salem, MA": (42.5195, -70.8967),
    "Birmingham, AL": (33.5186, -86.8104),
    "Denver, CO": (39.7392, -104.9903),
    "St Louis, MO": (38.6270, -90.1994),
    "Buffalo, NY": (42.8864, -78.8784),
    "New York City, NY": (40.7128, -74.0060),
    "Louisville, KY": (38.2527, -85.7585),
    "Indianapolis, IN": (39.7684, -86.1581),
    "Spokane, WA": (47.6588, -117.4260),
    "Frankfort, KY": (38.2009, -84.8733),
}

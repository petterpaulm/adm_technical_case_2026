# Fleet Delivery Optimisation -- Technical Documentation

> A four-solver vehicle routing engine with HOS-compliant scheduling,
> Monte Carlo risk simulation, and a 21-chart interactive dashboard.
>
> This document covers the mathematics, algorithms, and engineering
> decisions in detail. For the executive summary and deployment guide,
> see the [root README](../README.md).

---

## Table of Contents

1. [Problem Statement](#1--problem-statement)
2. [Input Data](#2--input-data)
3. [Distance Model](#3--distance-model)
4. [Cost Model](#4--cost-model)
5. [Hours-of-Service (HOS) Scheduling](#5--hours-of-service-hos-scheduling)
6. [Solver 1 -- Greedy Constructive Heuristic](#6--solver-1----greedy-constructive-heuristic)
7. [Solver 2 -- Adaptive Large Neighbourhood Search (ALNS)](#7--solver-2----adaptive-large-neighbourhood-search-alns)
8. [Solver 3 -- Column Generation](#8--solver-3----column-generation)
9. [Solver 4 -- Multi-Objective Pareto Optimiser](#9--solver-4----multi-objective-pareto-optimiser)
10. [Key Assumptions and Simplifications](#10--key-assumptions-and-simplifications)
11. [Carbon Emissions Tracking](#11--carbon-emissions-tracking)
12. [Monte Carlo Risk Simulation](#12--monte-carlo-risk-simulation)
13. [How to Run](#13--how-to-run)
14. [Interactive Dashboard and Visualisations](#14--interactive-dashboard-and-visualisations)
15. [Project Structure](#15--project-structure)
16. [References](#16--references)

---

## 1 -- Problem Statement

The company operates a fleet of **15 trucks** stationed at a single warehouse in
**Cincinnati, OH**. The fleet must deliver **300 order lines** (spanning
29 unique products) to roughly **50 destinations** across North America.

This is a variant of the **Capacitated Vehicle Routing Problem (CVRP)**,
one of the most studied problems in combinatorial optimisation
(Toth & Vigo, 2014). The goal is to find a set of routes that:

| Objective | Meaning |
|:---|:---|
| Minimise total cost | Fuel + driver labour + truck maintenance |
| Respect truck capacities | Each route's cargo does not exceed the assigned truck's weight limit |
| Limit route complexity | At most 8 delivery stops per route |
| Obey HOS regulations | Drivers must rest after 14 h of continuous work |

The CVRP is **NP-hard** (Lenstra & Rinnooy Kan, 1981), meaning no known
algorithm solves all instances in polynomial time. We built a portfolio
of four solvers that range from fast heuristics to mathematical-programming
approaches, giving the user explicit control over the speed-quality trade-off.

---

## 2 -- Input Data

Three CSV files describe the problem instance.

### 2.1 Items (`item_info.csv`)

| Column | Description |
|:---|:---|
| `ItemId` | Unique product identifier |
| `Weight Per Unit (lbs)` | Unit weight in US pounds |
| `Origin` | Always Cincinnati, OH (single-depot assumption) |

All weights are converted to kilograms on load:

$$w_{\text{kg}} = w_{\text{lbs}} \times 0.453592$$

### 2.2 Orders (`orders.csv`)

| Column | Description |
|:---|:---|
| `Company` | Customer identifier (CompanyA, CompanyB, ...) |
| `ItemId` | Foreign key into `item_info.csv` |
| `Number of Units` | Quantity ordered |
| `Destination` | Delivery city (e.g. Portland, OR) |

An order's total weight:

$$W_{\text{order}} = \text{units} \times w_{\text{kg}}$$

### 2.3 Trucks (`trucks.csv`)

| Column | Description |
|:---|:---|
| `Vehicle ID` | 6-digit zero-padded identifier |
| `Driver Name` | Assigned driver |
| `Vehicle Length (m)` | Used to look up fuel economy |
| `Weight Capacity (kg)` | Maximum payload |
| `Driver Hourly Rate` | Labour cost per hour |

The fleet comprises four vehicle classes:

| Class | Length (m) | Capacity (kg) | Fuel Economy (MPG) | Count |
|:---|:---:|:---:|:---:|:---:|
| Class 8 | 16.06 | 10,000 | 5.5 | 5 |
| Class 7 | 14.82 | 7,500 | 6.0 | 4 |
| Class 6 | 13.98 | 6,000 | 6.5 | 3 |
| Class 5 | 11.57 | 4,000 | 7.5 | 3 |

Smaller trucks are more fuel-efficient but carry less. The optimiser
balances these trade-offs automatically.

---

## 3 -- Distance Model

We need the distance between any two cities. To keep the solution
self-contained (no external routing API), we compute distances analytically.

### 3.1 Haversine Formula

The Haversine formula calculates the great-circle distance between two
points on a sphere. It is the standard geodesic distance for transportation
planning when road-network data is unavailable.

Given two points $(\phi_1, \lambda_1)$ and $(\phi_2, \lambda_2)$ in radians:

$$d = 2R \arcsin\!\left(\sqrt{\sin^2\!\left(\frac{\phi_2 - \phi_1}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\!\left(\frac{\lambda_2 - \lambda_1}{2}\right)}\right)$$

where $R = 3{,}958.8$ miles is the Earth's mean radius.

Think of it as stretching a string along the surface of a globe between
two cities -- the Haversine gives you the length of that string.

### 3.2 Road-Distance Correction

Real roads do not follow the surface of a perfect sphere. They curve
around mountains, follow highways, and detour through interchanges.
The standard practice is to apply a circuity factor (Ballou et al., 2002):

$$d_{\text{road}}(i, j) = d_{\text{haversine}}(i, j) \times \kappa$$

We use $\kappa = 1.30$, meaning roads are assumed to be 30% longer than
the straight-line distance. The US national average circuity factor is
typically in the range 1.2 to 1.4 for intercity freight.

### 3.3 Route Distance

A route visits an ordered sequence of cities
$c_1, c_2, \ldots, c_n$ starting and ending at the warehouse $W$:

$$D_{\text{route}} = d(W, c_1) + \sum_{k=1}^{n-1} d(c_k, c_{k+1}) + d(c_n, W)$$

This is a round-trip distance: the truck must return empty to
Cincinnati after completing all deliveries.

### 3.4 Distance Matrix

Before solving, we pre-compute a full symmetric distance matrix
$\mathbf{D} \in \mathbb{R}^{(m+1) \times (m+1)}$ where $m$ is the number
of unique cities (including the warehouse). This makes distance lookups
$O(1)$ during optimisation, avoiding expensive trigonometric recomputations.

---

## 4 -- Cost Model

Every route incurs three types of cost.

### 4.1 Fuel Cost

Each truck class has a characteristic fuel economy $\eta$ in miles per
gallon, determined by vehicle length:

$$\text{fuel\_gallons} = \frac{D_{\text{route}}}{\eta}$$

$$C_{\text{fuel}} = \text{fuel\_gallons} \times P_{\text{fuel}}$$

where $P_{\text{fuel}} = \$3.75\text{/gal}$.

### 4.2 Labour Cost

Drivers are paid for all work hours, not just driving:

$$t_{\text{drive}} = \frac{D_{\text{route}}}{v_{\text{avg}}}$$

$$t_{\text{load}} = \frac{30}{60} = 0.5 \;\text{h} \quad\text{(warehouse loading)}$$

$$t_{\text{stops}} = \frac{n_{\text{stops}} \times 35}{60} \;\text{h} \quad\text{(20 min unload + 15 min paperwork per stop)}$$

$$t_{\text{work}} = t_{\text{drive}} + t_{\text{load}} + t_{\text{stops}}$$

$$C_{\text{labour}} = t_{\text{work}} \times r_{\text{driver}}$$

where $v_{\text{avg}} = 55$ mph and $r_{\text{driver}}$ is the driver's
hourly rate ($22 to $40/h depending on experience).

### 4.3 Maintenance Cost

A mileage-based estimate:

$$C_{\text{maint}} = D_{\text{route}} \times \$0.15\text{/mi}$$

### 4.4 Total Cost

$$C_{\text{route}} = C_{\text{fuel}} + C_{\text{labour}} + C_{\text{maint}}$$

$$C_{\text{fleet}} = \sum_{\text{all routes}} C_{\text{route}}$$

The solvers optimise total distance as a proxy for cost during the search
(since distance is the dominant cost driver). The full three-component cost
is computed after routes are finalised.

---

## 5 -- Hours-of-Service (HOS) Scheduling

The US Federal Motor Carrier Safety Administration (FMCSA) mandates rest
rules for commercial truck drivers. We implement a simplified but
representative version:

| Rule | Value | Description |
|:---|:---|:---|
| Maximum work window | 14 hours | A driver may work for at most 14 consecutive hours |
| Mandatory rest | 10 hours | After reaching the 14h limit, the driver must rest for 10 hours |

### 5.1 Scheduling Algorithm

The scheduler simulates a timeline starting from Monday, 3 February 2026
at 08:00 AM and processes routes in order of increasing distance (shortest
first, so trucks free up quickly for subsequent routes).

For each route:

1. **Truck availability check** -- if the truck is still on a previous
   route, the new route starts after the truck returns + 10h rest.
2. **Warehouse loading** -- 30 minutes.
3. **For each delivery stop:**
   - Compute driving time: $t = d / 55$.
   - **HOS check:** if adding this driving time would push total work
     hours past 14h, insert a 10-hour mandatory rest break, then
     reset the work-hour counter.
   - Advance the clock by driving time.
   - Add 35 minutes for unloading + paperwork.
4. **Return leg** -- same HOS check before driving back to Cincinnati.
5. **Turnaround** -- after arrival, the truck rests for 10 hours before
   it can be dispatched again.

The output is a `RouteSchedule` with a full event timeline (loading,
delivery, return, rest breaks) and timestamps for every activity.

### 5.2 Simplifications

We implement only the 14-hour/10-hour rule. The real FMCSA regulations
also include an 11-hour driving sub-limit, a mandatory 30-minute break
after 8 hours, and weekly 60/70-hour caps. These are omitted for scope
but the architecture supports adding them.

---

## 6 -- Solver 1 -- Greedy Constructive Heuristic

**Runtime:** ~0.03 seconds | **Quality:** Good baseline

Builds routes from scratch using a four-phase pipeline. Fast and
deterministic -- ideal as a starting point or warm-start for meta-heuristics.

### Phase 1 -- First-Fit Decreasing Bin Packing

A classic bin-packing approach (Johnson, 1974) adapted for vehicle routing:

1. Group all orders by destination city.
2. Split any city whose total weight exceeds the largest truck's capacity
   into smaller chunks using First-Fit Decreasing (sort orders by weight,
   assign each to the first chunk that fits).
3. Sort all chunks descending by weight (heaviest first -- this is the
   "Decreasing" in FFD and is proven to produce better packings).
4. Insert each chunk into an existing route if:
   - The route can still carry the weight: $W_{\text{route}} + W_{\text{chunk}} \leq C_{\text{max}}$
   - The route has room for another stop: $n_{\text{stops}} < 8$
   - Among feasible routes, choose the one whose last stop is
     geographically closest to the chunk's city (proximity tie-break).
5. If no existing route can accommodate the chunk, open a new route.

### Phase 2 -- Clarke-Wright Savings Merge

The Clarke-Wright savings algorithm (Clarke & Wright, 1964) is one of the
earliest and most celebrated VRP heuristics. The key insight:

If two routes currently make separate round-trips to the warehouse, we can
save distance by linking them into one route that visits both sets of stops
in sequence.

For every pair of routes $(i, j)$, compute the savings:

$$s_{ij} = d(W, \text{last}_i) + d(W, \text{first}_j) - d(\text{last}_i, \text{first}_j)$$

$d(W, \text{last}_i)$ is the distance route $i$ currently spends returning
to the warehouse, and $d(W, \text{first}_j)$ is the distance route $j$
spends leaving the warehouse. If we merge them, we replace these two legs
with a single direct leg $d(\text{last}_i, \text{first}_j)$. The
difference is our savings.

We sort all savings in descending order and greedily merge pairs as long as
capacity and stop-count constraints are satisfied.

### Phase 3 -- TSP Optimisation (Nearest Neighbour + 2-Opt)

After merging, each route's stops may be in a sub-optimal order. We solve
a mini Travelling Salesman Problem for each route.

**Nearest-Neighbour Construction:** Starting at the warehouse, repeatedly
visit the nearest unvisited city until all stops are visited. $O(n^2)$.

**2-Opt Improvement** (Croes, 1958): Pick two edges in the tour, reverse
the segment between them, accept if shorter. Repeat until no improving swap
exists. This eliminates crossing edges -- two edges that cross can always
be uncrossed to produce a shorter tour.

```
Before (crossing):       After (uncrossed):
A ---- C                 A ---- B
  \  /                       |
   \/                        |
  /  \                       |
B ---- D                 C ---- D
```

### Phase 4 -- Load-Balanced Truck Assignment

Routes are assigned to trucks using a load-balanced strategy:

1. Sort routes by weight (heaviest first).
2. For each route, find all trucks with sufficient capacity.
3. Among feasible trucks, pick the one with the fewest routes already
   assigned (tie-break: lowest hourly rate to minimise labour cost).

---

## 7 -- Solver 2 -- Adaptive Large Neighbourhood Search (ALNS)

**Runtime:** ~10 min | **Quality:** State-of-the-art metaheuristic

ALNS (Ropke & Pisinger, 2006) is the gold-standard metaheuristic for
vehicle routing. It works by repeatedly destroying part of a solution and
repairing it, guided by simulated annealing for diversification.

### 7.1 Core Idea

Think of remodelling a house: instead of rebuilding from scratch every
time, you demolish one room and rebuild it better. Over thousands of
iterations, the whole house improves.

### 7.2 Simulated Annealing Acceptance

Not every repair improves the solution. To avoid getting stuck in local
optima, we accept worse solutions with a probability that decreases over
time:

$$P(\text{accept}) = \begin{cases} 1 & \text{if } \Delta \leq 0 \;\;(\text{improvement})\\[4pt] e^{-\Delta / T} & \text{if } \Delta > 0 \;\;(\text{deterioration}) \end{cases}$$

where $\Delta = f(\text{new}) - f(\text{current})$ and $T$ is the
temperature. It cools geometrically:

$$T_{k+1} = T_k \times \alpha, \quad \alpha = 0.9997$$

Starting temperature: $T_0 = 500$. At high temperatures the search explores
broadly (accepts worse solutions about half the time at $T_0 = 500$). As
$T \to 0$, only improvements are accepted.

### 7.3 Adaptive Operator Selection

ALNS maintains a portfolio of destroy and repair operators. Each operator
has a weight updated every 100 iterations based on performance:

$$w \leftarrow \max\!\left(0.05, \; w \cdot (1 - \rho) + \rho \cdot \frac{\text{score}}{\text{uses}}\right)$$

where $\rho = 0.1$ is the reaction factor. Operators are selected by
roulette-wheel selection (probability proportional to weight).

| Event | Score |
|:---|:---|
| New global best solution | 33 |
| Improved current (but not global best) | 9 |
| Accepted (but not an improvement) | 1 |

### 7.4 Destroy Operators

Each destroy operator removes $k$ orders, where
$k \in [\lceil 0.10 \cdot S \rceil, \lfloor 0.35 \cdot S \rfloor]$
and $S$ is the total number of stops across all routes.

| # | Operator | Strategy |
|:---|:---|:---|
| 1 | Random Removal | Remove $k$ orders uniformly at random. Simple but effective for diversification. |
| 2 | Worst Removal | Score each order by its marginal distance contribution. Remove the $k$ costliest. |
| 3 | Shaw Removal | Pick a seed order. Score others by relatedness: $\text{rel}(i,j) = d(c_i, c_j) + 0.5 \cdot \|w_i - w_j\|$. Remove the $k$ most related. Creates a hole that is easy to repair differently. (Shaw, 1997) |
| 4 | Route Removal | Remove the entire worst route: $\arg\max_r D_r / W_r$. Forces major restructuring. |

### 7.5 Repair Operators

| # | Operator | Strategy |
|:---|:---|:---|
| 1 | Greedy Insertion | Insert each removed order at the cheapest feasible position across all routes. |
| 2 | Regret-2 Insertion | Insert the order with the largest regret first. Regret = $\Delta_2 - \Delta_1$ (second-best minus best insertion cost). High regret means the order has only one good placement -- insert it before that slot is taken. (Potvin & Rousseau, 1993) |
| 3 | Noisy Greedy | Same as greedy, but adds Gaussian noise: $\Delta' = \Delta + \mathcal{N}(0, \max(0.3 |\Delta|, 10))$. Injects randomness to explore alternative placements. |

### 7.6 Insertion Cost Calculation

When inserting an order for city $c$ into a route with stops
$s_1, s_2, \ldots, s_n$, we evaluate every position $p$:

$$\Delta_p = d(s_{p-1}, c) + d(c, s_p) - d(s_{p-1}, s_p)$$

If city $c$ already appears in the route, insertion cost is zero -- we
simply add the order to the existing stop.

Opening a new route costs $\Delta_{\text{new}} = 2 \cdot d(W, c)$.

### 7.7 Algorithm Pseudocode

```
solution <- Greedy()                    // warm-start
best <- solution
T <- 500

FOR iter = 1 TO 6,000:
    destroy_op <- roulette_select(destroy_operators)
    repair_op  <- roulette_select(repair_operators)

    removed_orders <- destroy_op(solution)
    candidate <- repair_op(solution, removed_orders)

    delta <- cost(candidate) - cost(solution)

    IF delta < 0  OR  random() < exp(-delta / T):
        solution <- candidate
        IF cost(solution) < cost(best):
            best <- solution
            reward destroy_op, repair_op with SCORE_BEST (33)
        ELSE:
            reward with SCORE_BETTER (9)
    ELSE:
        reward with SCORE_ACCEPT (1)

    T <- T * 0.9997

    IF iter mod 100 == 0:
        update_all_operator_weights()

RETURN best
```

---

## 8 -- Solver 3 -- Column Generation

**Runtime:** ~42 seconds | **Quality:** Near-optimal (LP relaxation bound)

Column Generation (CG) is a mathematical programming technique that solves
large-scale linear programs by generating variables (columns) on the fly.
Introduced by Dantzig & Wolfe (1960), it is the backbone of most commercial
VRP solvers used in industry today.

### 8.1 Why Column Generation?

A brute-force formulation of the VRP would need one binary variable for
every possible route -- an astronomically large number. CG avoids this by
starting with a small set of routes and adding only those that could improve
the solution.

### 8.2 Master Problem (Set Covering LP)

Let $\mathcal{R}$ be the pool of known feasible routes. For each route
$j \in \mathcal{R}$:

- $c_j$ = total distance of route $j$
- $a_{ij} = 1$ if route $j$ covers order $i$, else $0$
- $x_j \in [0, 1]$ = decision variable

$$\min \sum_{j \in \mathcal{R}} c_j \cdot x_j$$

subject to:

$$\sum_{j \in \mathcal{R}} a_{ij} \cdot x_j \geq 1 \quad \forall \;\text{order } i$$

$$0 \leq x_j \leq 1 \quad \forall j \in \mathcal{R}$$

Choose a combination of routes (fractions allowed in the LP relaxation)
that covers every order at minimum total distance.

### 8.3 Pricing Sub-Problem

After solving the master LP, we obtain dual prices $\pi_i$ for each order
constraint. A new route $j$ is worth adding only if its reduced cost is
negative:

$$\bar{c}_j = c_j - \sum_{i} \pi_i \cdot a_{ij} < 0$$

$c_j$ is the route's actual cost. $\sum \pi_i a_{ij}$ is the value the
master LP attributes to covering those orders. If the value exceeds the
cost, the route is profitable.

We generate new routes greedily, prioritising orders with high dual prices
(they are under-served in the current solution).

### 8.4 CG Iteration

```
pool <- Greedy routes + {single-order routes}    // ensure feasibility

FOR round = 1 TO 60:
    Solve master LP over current pool -> x*, pi*

    new_columns <- price(pi*, up to 15 routes)
    IF no column has reduced cost < -1e-4:
        BREAK                                    // optimality reached

    pool <- pool + new_columns

x_LP* <- final LP solution (fractional)
```

### 8.5 Integer Rounding

The LP relaxation gives fractional $x_j$ values. We need integer routes.
We apply greedy set covering:

1. While uncovered orders remain:
   - For each route, compute: $\text{score}_j = |\text{uncovered orders covered by } j| / \max(c_j, 1)$
   - Select the route with the highest score.
   - Mark its orders as covered.

This is a well-known approximation for the Set Cover problem with a
guaranteed $O(\ln n)$ ratio (Chvatal, 1979).

### 8.6 Connection to Lagrangian Relaxation

Column Generation and Lagrangian Relaxation are connected through LP
duality theory. The Lagrangian relaxation of the master problem:

$$L(\pi) = \min_{x \geq 0} \sum_j c_j x_j + \sum_i \pi_i\!\left(1 - \sum_j a_{ij} x_j\right)$$

Rearranging:

$$L(\pi) = \sum_i \pi_i + \min_{x \geq 0} \sum_j \underbrace{\left(c_j - \sum_i \pi_i a_{ij}\right)}_{\bar{c}_j \text{ (reduced cost)}} x_j$$

The inner minimisation is the pricing sub-problem. When the LP has an
optimal solution, its dual variables are identical to the optimal Lagrange
multipliers (Lubbecke & Desrosiers, 2005). This means CG automatically
solves the Lagrangian dual, and the LP objective is a valid lower bound on
the optimal integer solution.

---

## 9 -- Solver 4 -- Multi-Objective Pareto Optimiser

**Runtime:** Variable | **Quality:** Explores the full cost-makespan trade-off

Real logistics decisions involve conflicting objectives. The Pareto solver
finds solutions where no objective can be improved without worsening
another -- the Pareto front.

### 9.1 Three Objectives

| # | Objective | Formula | Goal |
|:---|:---|:---|:---|
| $f_1$ | Total cost | $\sum D_r$ | Minimise |
| $f_2$ | Makespan | $\max_r T_r$ | Minimise |
| $f_3$ | Utilisation balance | $\text{stdev}(u_1, \ldots, u_n)$ | Minimise |

Where $T_r = D_r / 55 + (n_r \times 35 + 30) / 60$ and
$u_r = W_r / C_r$.

### 9.2 Pareto Dominance

Solution $\mathbf{p}$ dominates $\mathbf{q}$ ($\mathbf{p} \prec \mathbf{q}$) if:

$$\forall k \in \{1,2,3\}: f_k(\mathbf{p}) \leq f_k(\mathbf{q}) \quad \text{AND} \quad \exists k: f_k(\mathbf{p}) < f_k(\mathbf{q})$$

The Pareto front is the set of all non-dominated solutions. Each point
represents a fundamentally different trade-off.

### 9.3 Algorithm

1. Run ALNS with default parameters (cost-focused).
2. Run greedy solver (fast baseline).
3. Run ALNS 6 more times with randomised hyperparameters ($T_0 \sim U[200, 1000]$, $\alpha \sim U[0.999, 0.9999]$, 2,000 iterations each).
4. Non-domination filter: keep only non-dominated solutions.
5. Return the cheapest Pareto-optimal solution as primary, store the full front in metadata.

---

## 10 -- Key Assumptions and Simplifications

| # | Assumption | Rationale |
|:---|:---|:---|
| 1 | Single depot (Cincinnati, OH) | Matches the problem statement; hub-and-spoke is typical for food-commodity logistics |
| 2 | Road distance = Haversine x 1.30 | Avoids dependency on external routing APIs; 1.30 is within the empirical US intercity range |
| 3 | Constant speed = 55 mph | Reasonable loaded-truck highway average |
| 4 | Fuel economy depends only on truck class | Simplification -- load weight, terrain, and driving style also affect MPG |
| 5 | Simplified HOS: 14h work / 10h rest only | Captures the dominant constraint; full FMCSA rules omitted for scope |
| 6 | Uniform stop overhead: 35 min | Real unloading time depends on cargo volume and dock availability |
| 7 | No time windows | Customers accept delivery at any time; adding windows makes this a harder VRPTW |
| 8 | Deterministic demand | All 300 orders known upfront; no stochastic cancellations |
| 9 | No split deliveries | Each order goes on exactly one truck |
| 10 | Distance as solver objective | Solvers minimise distance (correlated with cost); dollar costs computed post-optimisation |

---

## 11 -- Carbon Emissions Tracking

### Emission Model

$$\text{CO}_2^{(r)} = \frac{d_r}{\eta_r} \times 10.18 \;\text{kg/gal}$$

where $d_r$ is route distance, $\eta_r$ is fuel economy, and 10.18 kg
CO2 per US gallon is the EPA reference factor for diesel (EPA, 2023).

### Fleet-Level Metrics

| Metric | Definition |
|:---|:---|
| Total CO2 | $\sum_r \text{CO}_2^{(r)}$ |
| CO2 per route | Total CO2 / $n_{\text{routes}}$ |
| CO2 per mile | Total CO2 / $\sum_r d_r$ |

### Sample Output (ALNS)

| Metric | Value |
|:---|:---|
| Total CO2 | 161,753 kg (162 t) |
| CO2 / route | 4,902 kg |
| CO2 / mile | 1.851 kg |

---

## 12 -- Monte Carlo Risk Simulation

Deterministic solutions assume perfect knowledge. In practice, travel times
fluctuate, trucks break down, and fuel prices change. The Monte Carlo
simulator quantifies this uncertainty.

### Why Not Weather Data or Traffic APIs?

Monte Carlo simulation does not require live data feeds. We use parametric
stochastic modelling -- each source of uncertainty is characterised by a
probability distribution calibrated to industry-accepted ranges:

| Factor | Calibration Source | Parameter |
|:---|:---|:---|
| Travel-time CV = 20% | Empirical US inter-city trucking (Figliozzi, 2010) | Log-normal sigma |
| Breakdown rate 5% | FMCSA roadside inspection out-of-service rate | Bernoulli p |
| Fuel-price CV = 15% | Annualised diesel volatility (EIA Weekly Petroleum Status Report) | Normal sigma |

We model the effect of disruptions on operational variables (travel time,
delay, cost) rather than the underlying physical causes. Whether a
30-minute delay comes from a traffic jam or a rainstorm, the cost impact
is the same. By the Central Limit Theorem, 1,000 independent trials give
a statistically robust picture of risk.

### Perturbation Model

Each trial applies four independent stochastic shocks:

| Factor | Distribution | Default sigma |
|:---|:---|:---|
| Travel time | Log-normal $(\mu=0, \sigma=0.20)$ | +/-20% |
| Truck breakdown | Bernoulli $(p=0.05)$ + Exponential $(\lambda=4\text{h})$ | 5% per route |
| Fuel price | Truncated Normal $(\mu=1, \sigma=0.15)$ | +/-15% |
| Demand surge | +15% labour cost | One-at-a-time |

### Cost Under Perturbation

For each trial $t$ and route $r$:

$$\tilde{C}_r^{(t)} = \underbrace{\frac{d_r \cdot \alpha_r^{(t)}}{\eta_r} \pi^{(t)}}_{\text{fuel}} + \underbrace{(\tilde{w}_r^{(t)} + \delta_r^{(t)}) \rho_r}_{\text{labour}} + \underbrace{d_r \mu}_{\text{maint}}$$

where $\alpha_r$ is the travel-time multiplier, $\pi$ is the perturbed
diesel price, $\delta_r$ is breakdown delay, and $\rho_r$ is the driver
hourly rate.

### Output Metrics

| Metric | Definition |
|:---|:---|
| P5 / P50 / P95 cost | Percentiles of the total cost distribution |
| Late-delivery risk | % of trials where any route exceeds the 14h HOS limit |
| Robustness score | $1 - \text{CV}$ (higher = more stable) |

### Sample Output (ALNS, 1,000 trials)

| Metric | Value |
|:---|:---|
| Deterministic cost | $126,854 |
| P50 (median) | $129,240 |
| P95 (worst case) | $137,880 |
| Robustness score | 96% |

---

## 13 -- How to Run

### Prerequisites

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Solve and Export

```bash
cd optimization
python main.py --solver greedy      # fast, ~0.03s
python main.py --solver alns        # ALNS, ~10 min
python main.py --solver colgen      # Column Generation, ~42s
python main.py --solver pareto      # Multi-objective, variable
```

### Solve and Launch Dashboard

```bash
python main.py --solver greedy --dashboard
# Open http://localhost:8050
```

### Compare All Solvers

```bash
python main.py --solver all --dashboard
# Runs greedy + colgen + alns, opens multi-solver dashboard
# Total runtime: ~11 min (ALNS dominates)
```

### Monte Carlo

```bash
python main.py --solver greedy --monte-carlo --dashboard
python main.py --solver all --monte-carlo --dashboard
```

### Instant Dashboard Reload

```bash
python main.py --dashboard-only
# Reloads from results/solver_cache.pkl (< 2s)
```

### Lint

```bash
python -m ruff check .
python -m ruff format .
```

---

## 14 -- Interactive Dashboard and Visualisations

The Dash dashboard provides 21 interactive charts plus cross-solver
comparison panels, styled with corporate palette.

### Per-Solver Analytics

| Chart | Type | What It Shows |
|:---|:---|:---|
| Route Network Map | Scattergeo | Geographic route lines with depot marker |
| Cost Waterfall | Waterfall | Fuel to labour to maintenance to total |
| Cost by Route | Stacked bar | Cost composition per route |
| Fleet Utilisation | Horizontal bar | Capacity % vs 90% target |
| Distance vs Weight | Bubble scatter | Bubble size = cost |
| Driver Workload Radar | Radar | Routes / distance / weight / cost |
| Cost Treemap | Treemap | Fleet to category to route hierarchy |
| Dispatch Timeline | Bar (Gantt) | HOS-compliant schedule per driver |
| Driver-Distance Heatmap | Heatmap | Miles by driver x week |
| Cost Sankey | Sankey | Money flow from routes to cost components |
| Efficiency Frontier | Scatter + hull | Cost/kg vs dist/stop |
| Delivery Cadence | Step chart | Cumulative deliveries over time |
| CO2 by Route | Horizontal bar | Per-route emissions vs average |
| Emissions Waterfall | Waterfall | CO2 by truck class |
| MC Cost Distribution | Histogram | 1,000-trial histogram with P5/P95 |
| Sensitivity Tornado | Tornado bar | Factor impact on total cost |
| Robustness Gauge | Gauge | 0-100% robustness score |

### Cross-Solver Comparison

| Chart | Type | What It Shows |
|:---|:---|:---|
| Cost Breakdown | Grouped bar | Fuel / labour / maintenance per strategy |
| Efficiency Radar | Radar | 5-axis normalised comparison |
| Routes and Distance | Dual-axis | Route count (bar) + total miles (line) |
| Risk Distribution | Violin + box | MC cost distributions per solver |
| Cost per Mile | Box + strip | Route-level $/mi spread |
| Risk Metrics Table | Table | P50, P95, spread, robustness |

### Jupyter Notebook

`fleet_analytics.ipynb` contains 20 publication-quality charts with full
analytical narrative, mathematical formulations, and strategic commentary.

---

## 15 -- Project Structure

```
optimization/
|-- main.py                         # CLI entry point and pipeline
|-- README.md                       # This file
|
+-- src/
|   |-- config.py                   # Constants, city coords, hyperparameters
|   |-- models.py                   # Truck, Order, Stop, Route, Solution
|   |-- data_loader.py              # CSV to domain objects
|   |-- cost_engine.py              # Fuel + labour + maintenance + CO2
|   |-- distance.py                 # Haversine, distance matrix
|   |-- scheduler.py                # HOS-compliant timeline builder
|   |-- reporting.py                # CSV + console output
|   |-- monte_carlo.py              # 1,000-trial risk simulation
|   +-- solvers/
|       |-- base.py                 # Abstract solver interface
|       |-- greedy.py               # Four-phase constructive heuristic
|       |-- alns.py                 # Adaptive Large Neighbourhood Search
|       |-- column_gen.py           # Column Generation (LP + pricing)
|       +-- pareto.py               # Multi-objective Pareto optimiser
|
+-- dashboard/
|   |-- app.py                      # Dash application factory
|   |-- layouts.py                  # Component tree
|   |-- figures.py                  # 21 Plotly figure factories
|   |-- callbacks.py                # Interactive callbacks
|   +-- assets/
|       +-- style.css               # Corporate theme
|
+-- fleet_analytics.ipynb           # 20-chart analytical notebook
|
+-- results/                        # Auto-generated outputs
    |-- route_details.csv
    |-- order_assignments.csv
    |-- fleet_summary.csv
    |-- dispatch_timeline.csv
    |-- solution_metadata.json
    +-- solver_cache.pkl            # Cached results for --dashboard-only
```

---

## 16 -- References

| Reference | Context |
|:---|:---|
| Clarke, G. & Wright, J.W. (1964). *Scheduling of vehicles from a central depot.* Operations Research, 12(4), 568-581. | Savings merge heuristic |
| Croes, G.A. (1958). *A method for solving traveling-salesman problems.* Operations Research, 6(6), 791-812. | 2-Opt local search |
| Dantzig, G.B. & Wolfe, P. (1960). *Decomposition principle for linear programs.* Operations Research, 8(1), 101-111. | Column generation |
| Lubbecke, M.E. & Desrosiers, J. (2005). *Selected topics in column generation.* Operations Research, 53(6), 1007-1023. | CG-Lagrangian duality |
| Desaulniers, G. et al. (2005). *Column Generation.* Springer. | Comprehensive CG reference |
| Chvatal, V. (1979). *A greedy heuristic for the set-covering problem.* Mathematics of OR, 4(3), 233-235. | Integer rounding |
| Johnson, D.S. (1974). *Fast algorithms for bin packing.* J. Computer and System Sciences, 8(3), 272-314. | First-Fit Decreasing |
| Potvin, J.Y. & Rousseau, J.M. (1993). *A parallel route building algorithm for VRPTW.* European J. of OR, 66(3), 331-340. | Regret-2 insertion |
| Ropke, S. & Pisinger, D. (2006). *An ALNS heuristic for pickup and delivery with time windows.* Transportation Science, 40(4), 455-472. | ALNS framework |
| Shaw, P. (1997). *A new local search algorithm for vehicle routing.* University of Strathclyde. | Shaw removal operator |
| Toth, P. & Vigo, D. (2014). *Vehicle Routing: Problems, Methods, and Applications* (2nd ed.). MOS-SIAM. | General VRP background |
| Ballou, R.H. et al. (2002). *Estimating intercity freight transportation costs.* Transportation Research Part E, 38(6). | Circuity factor |
| Boscoe, F.P. et al. (2012). *A nationwide comparison of driving distance versus straight-line distance.* Professional Geographer, 64(2). | Road correction factor |
| EPA (2023). *Emission Factors for Greenhouse Gas Inventories.* US EPA. | Diesel CO2: 10.18 kg/gal |
| Lenstra, J.K. & Rinnooy Kan, A.H.G. (1981). *Complexity of vehicle routing and scheduling problems.* Networks, 11(2). | NP-hardness proof |
| Figliozzi, M.A. (2010). *Impacts of congestion on commercial vehicle tour costs.* Transportation Research Part E, 46(4), 496-506. | Travel-time variability |
| Metropolis, N. & Ulam, S. (1949). *The Monte Carlo method.* J. American Statistical Association, 44(247), 335-341. | Monte Carlo framework |

---

<p align="center">
  AI / Cloud Architecture -- Technical Case<br>
  <strong>Pedro Paulo da Cruz Mendes</strong> -- Sao Paulo, SP, Brazil | Decatur, Illinois, US<br>
  February 2026
</p>

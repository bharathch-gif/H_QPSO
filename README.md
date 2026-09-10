# QPSO Traffic Router — SIH Starter Project

Quantum-inspired (QPSO) metaheuristic for solving Vehicle Routing Problems on
real city road networks, benchmarked against classical PSO and a Genetic
Algorithm.

## Setup

```bash
pip install -r requirements.txt
```

`osmnx` downloads real OpenStreetMap road networks the first time you run it
for a given city, so the first run needs an internet connection and will be
slow (it's cached afterwards).

## Project layout

| File | Concept it implements |
|---|---|
| `graph_builder.py` | Weighted graph model of the road network (nodes, edges, travel-time weights) + simulated congestion |
| `vrp_problem.py` | Mathematical formulation of the VRP: objective function (`route_cost`) and constraint handling (capacity penalty) |
| `encoding.py` | Continuous → discrete conversion (random-key encoding) so swarm algorithms can represent routes |
| `qpso.py` | The Quantum Particle Swarm Optimization algorithm |
| `baselines.py` | Classical PSO and Genetic Algorithm, for comparison |
| `benchmark.py` | Runs every algorithm N times, collects convergence + fitness stats |
| `utils.py` | Plotting helpers (convergence curves, route maps) |
| `app.py` | Streamlit dashboard tying everything together |

## Running

Quick command-line test (no map/UI):

```bash
python benchmark.py
```

Full interactive dashboard:

```bash
streamlit run app.py
```

## Where to go from here

- `vrp_problem.py`'s `route_cost` currently handles vehicle capacity via a
  simple greedy split + penalty. Swap in a smarter split algorithm if you
  need it.
- `graph_builder.py` simulates congestion randomly since live traffic APIs
  are hard to source in a hackathon timeline — say this explicitly in your
  presentation rather than implying it's live data.
- Tune `n_particles` / `n_iterations` in `qpso.py` and `baselines.py` to
  trade off solution quality vs. runtime for your demo.

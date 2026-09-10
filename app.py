"""
app.py

Streamlit dashboard: pick a city, generate a depot + customer locations,
run QPSO / PSO / GA, and visualize the resulting route + convergence
comparison + benchmark table.

Run with:  streamlit run app.py
"""

import os
import csv
import streamlit as st
import numpy as np

from graph_builder import (
    build_city_graph, apply_simulated_congestion, pick_random_nodes, nodes_from_latlon
)
from vrp_problem import VRPProblem
from live_traffic import apply_live_traffic_to_problem
from tomtom_traffic import apply_tomtom_traffic_to_problem
from benchmark import compare_algorithms
from utils import plot_convergence, plot_route_on_graph, build_interactive_traffic_map
from qpso import run_qpso

try:
    import config
except ImportError:
    config = None

st.set_page_config(page_title="H-QPSO Traffic Router", layout="wide")
st.title("H-QPSO: Hybrid Quantum-Inspired Traffic Routing")

with st.expander("About this project (SIH objectives mapping)", expanded=False):
    st.markdown("""
    This platform implements a **Quantum-Inspired Metaheuristic Optimization
    Framework** for intelligent traffic routing, addressing each SIH objective:

    1. **Framework for large-scale VRP & shortest-path problems** — the road
       network is modeled as a weighted graph (`graph_builder.py`), and H-QPSO
       (`qpso.py` + `local_search.py`) searches over route orderings via a
       random-key encoding (`encoding.py`), with a 2-opt local-search polish
       step on top of the swarm search.
    2. **Minimize travel time, distance, and congestion** — the objective
       function (`vrp_problem.py`) scores every candidate route using
       simulated congestion-weighted travel times.
    3. **Reduce computational complexity, improve convergence** — benchmarked
       against classical PSO and a Genetic Algorithm (`baselines.py`), with
       convergence curves and runtime tracked for all three.
    4. **Scalability for smart-city logistics** — the number of delivery
       stops is adjustable in the sidebar to demonstrate behavior as problem
       size grows.
    """)

# --- Sidebar controls ---------------------------------------------------
st.sidebar.header("Setup")
city = st.sidebar.text_input("City / Place", "Vijayawada, India")
search_radius_km = st.sidebar.slider(
    "Search radius (km)", 2, 15, 5,
    help="Downloads roads within this radius of the city center, rather "
         "than its full administrative boundary (which can be much larger "
         "than expected and cause slow/failed downloads)."
)

stop_mode = st.sidebar.radio(
    "Delivery stops", ["Random (for testing)", "Enter my own coordinates"]
)

if stop_mode == "Random (for testing)":
    n_customers = st.sidebar.slider("Number of delivery stops", 3, 30, 10)
else:
    st.sidebar.caption(
        "One 'lat, lon' pair per line. Get coordinates by right-clicking a "
        "spot on Google Maps and copying the numbers shown."
    )
    depot_coord_text = st.sidebar.text_input("Depot coordinate (lat, lon)", "16.5062, 80.6480")
    customer_coords_text = st.sidebar.text_area(
        "Customer coordinates (one per line)",
        "16.5100, 80.6300\n16.4950, 80.6550\n16.5200, 80.6400",
        height=120,
    )

n_particles = st.sidebar.slider("Swarm size", 10, 100, 30)
n_iterations = st.sidebar.slider("Iterations", 20, 300, 100)
n_runs = st.sidebar.slider("Benchmark runs per algorithm", 3, 30, 10)
seed = st.sidebar.number_input("Random seed", value=42)

st.sidebar.header("Objective weights")
st.sidebar.caption(
    "Trade off what the optimizer prioritizes. Equal weights = balanced; "
    "push one up to favor it over the others."
)
w_time = st.sidebar.slider("Weight: travel time", 0.0, 1.0, 0.5, 0.05)
w_distance = st.sidebar.slider("Weight: distance", 0.0, 1.0, 0.3, 0.05)
w_congestion = st.sidebar.slider("Weight: congestion avoidance", 0.0, 1.0, 0.2, 0.05)

st.sidebar.header("Real-time traffic (optional)")

# API keys live in config.py, never in the UI or on screen -- see config.example.py
_configured_provider = getattr(config, "LIVE_TRAFFIC_PROVIDER", None) if config else None
_configured_key = None
if config:
    if _configured_provider == "tomtom":
        _configured_key = getattr(config, "TOMTOM_API_KEY", "") or None
    elif _configured_provider == "google":
        _configured_key = getattr(config, "GOOGLE_API_KEY", "") or None

if _configured_key:
    use_live_traffic = st.sidebar.checkbox(
        f"Use live traffic data ({_configured_provider.title()})", value=False
    )
    st.sidebar.caption(
        "If this fails, the app automatically falls back to the simulated "
        "congestion model — it will never crash the demo."
    )
else:
    use_live_traffic = False
    st.sidebar.caption(
        "Live traffic not configured — add your API key to config.py "
        "(see config.example.py) to enable this."
    )

if "graph" not in st.session_state:
    st.session_state.graph = None

if st.sidebar.button("Load city graph"):
    with st.spinner(f"Downloading road network for {city}..."):
        G = build_city_graph(city, dist_meters=search_radius_km * 1000)
        G = apply_simulated_congestion(G, seed=seed)
        st.session_state.graph = G
    st.success(f"Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def parse_latlon_lines(text):
    """Parses 'lat, lon' lines into a list of (lat, lon) float tuples, skipping blanks."""
    coords = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        lat_str, lon_str = line.split(",")
        coords.append((float(lat_str.strip()), float(lon_str.strip())))
    return coords


# --- Main panel -----------------------------------------------------------
if st.session_state.graph is None:
    st.info("Load a city graph from the sidebar to get started.")
else:
    G = st.session_state.graph

    if stop_mode == "Random (for testing)":
        depot = pick_random_nodes(G, 1, seed=int(seed))[0]
        customers = pick_random_nodes(G, n_customers, seed=int(seed) + 1, exclude={depot})
    else:
        try:
            depot_latlon = parse_latlon_lines(depot_coord_text)[0]
            customer_latlons = parse_latlon_lines(customer_coords_text)
            depot = nodes_from_latlon(G, [depot_latlon])[0]
            customers = nodes_from_latlon(G, customer_latlons)
            st.caption(f"Depot and {len(customers)} customer coordinates snapped to nearest roads.")
        except (ValueError, IndexError):
            st.error(
                "Couldn't parse coordinates — make sure each line is 'lat, lon' "
                "(e.g. 16.5062, 80.6480) and the depot field has exactly one pair."
            )
            st.stop()

    problem = VRPProblem(
        G, depot, customers,
        objective_weights={"time": w_time, "distance": w_distance, "congestion": w_congestion},
    )

    if use_live_traffic:
        with st.spinner(f"Fetching live traffic data from {_configured_provider.title()}..."):
            if _configured_provider == "tomtom":
                success, traffic_errors = apply_tomtom_traffic_to_problem(problem, _configured_key)
            else:
                success, traffic_errors = apply_live_traffic_to_problem(problem, _configured_key)
        if success:
            st.success(f"Live {_configured_provider.title()} traffic data applied for this route.")
            if traffic_errors:
                with st.expander("Live traffic warnings (partial data)"):
                    for err in traffic_errors:
                        st.caption(f"• {err}")
        else:
            st.warning(
                "Couldn't fetch live traffic data — using the simulated congestion "
                "model instead. Details:"
            )
            for err in traffic_errors:
                st.caption(f"• {err}")

    tab1, tab2, tab3 = st.tabs([
        "Single H-QPSO run", "Live benchmark: H-QPSO vs PSO vs GA", "Saved results (backup)"
    ])

    with tab1:
        if st.button("Run H-QPSO"):
            with st.spinner("Optimizing route..."):
                best_route, best_fitness, conv_log = run_qpso(
                    problem, n_particles=n_particles, n_iterations=n_iterations, seed=int(seed)
                )
            st.metric("Best route score (weighted, lower is better)", f"{best_fitness:.3f}")
            st.caption(
                "This is a blended score across your time/distance/congestion "
                "weights, not raw seconds — see the Route Metrics section for "
                "the actual distance and time breakdown."
            )

            st.subheader("Convergence")
            fig = plot_convergence({"H-QPSO": [{"convergence": conv_log}]})
            st.pyplot(fig)

            st.subheader("Route Map (colored by congestion)")
            full_path = [depot] + best_route + [depot]
            traffic_map, map_has_live_data = build_interactive_traffic_map(
                G, full_path, weight_attr=problem.weight_attr
            )
            if map_has_live_data:
                st.caption("🟢 Includes real live traffic data on at least one segment.")
            else:
                st.caption("Showing the simulated congestion model (no live traffic data applied).")
            st.components.v1.html(traffic_map._repr_html_(), height=500)

            st.subheader("Route Metrics")
            metrics = problem.route_metrics(best_route)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Distance", f"{metrics['total_distance_m']/1000:.2f} km")
            m2.metric("Free-flow time", f"{metrics['total_free_flow_time_s']/60:.1f} min")
            m3.metric("Congested time", f"{metrics['total_congested_time_s']/60:.1f} min")
            m4.metric("Congestion delay", f"{metrics['congestion_delay_s']/60:.1f} min")

    with tab2:
        if st.button("Run full benchmark"):
            with st.spinner(f"Running {n_runs} runs each for H-QPSO, PSO, GA..."):
                summary, all_results = compare_algorithms(
                    problem, n_runs=n_runs, n_iterations=n_iterations, n_particles=n_particles
                )

            st.subheader("Summary statistics")
            st.dataframe(summary)

            st.subheader("Convergence comparison")
            fig = plot_convergence(all_results)
            st.pyplot(fig)

    with tab3:
        st.caption(
            "Pre-computed results from benchmark.py — shown here as a backup "
            "in case a live run is slow or fails during judging."
        )
        if os.path.exists("benchmark_results.csv"):
            with open("benchmark_results.csv") as f:
                rows = list(csv.reader(f))
            st.table(rows[1:] if len(rows) > 1 else rows)
        else:
            st.warning("benchmark_results.csv not found — run `python3 benchmark.py` first.")

        if os.path.exists("convergence_plot.png"):
            st.image("convergence_plot.png", caption="H-QPSO vs PSO vs GA convergence")
        else:
            st.warning("convergence_plot.png not found — run `python3 benchmark.py` first.")

"""
graph_builder.py

Builds the weighted graph model of the road network: G = (V, E, W)
  V = intersections (nodes)
  E = road segments (edges)
  W = travel-time weight per edge (with simulated congestion applied)

Uses osmnx to pull a real road network from OpenStreetMap, and networkx
to represent/manipulate it as a graph.
"""

import time
import random
import re
import os
import numpy as np
import networkx as nx
import osmnx as ox
import requests

# --- Pre-baked local graph cache ------------------------------------------
# A fully-built city graph can be saved to disk as GraphML and committed to
# the repo. If a matching file exists here, build_city_graph() loads it
# directly -- zero network calls, zero dependency on Overpass being up or
# fast at that exact moment. This is what protects a live demo from the
# exact Overpass timeout failure this project has already hit once.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_CACHE_DIR = os.path.join(_MODULE_DIR, "cached_graphs")


def _slugify(place_name):
    """'Vijayawada, India' -> 'vijayawada_india' -- used as the cache filename."""
    return re.sub(r"[^a-z0-9]+", "_", place_name.strip().lower()).strip("_")

# --- Resilient Overpass access -------------------------------------------
# overpass-api.de is a free, shared, rate-limited public server. Large or
# geometrically complex places can take a long time to process, and under
# load it can simply fail to respond inside osmnx's default timeout. None
# of this is fixable client-side, so instead we: (1) give it more time,
# (2) retry with backoff, and (3) fall back to a mirror instance if the
# primary is down/slow. This trades a bit of latency on a cache-miss for
# not crashing the whole demo when OSM's public infra has a bad moment.

ox.settings.timeout = 60           # per-attempt timeout; kept short since we retry/fall back instead
ox.settings.use_cache = True       # explicit: cache to disk so a city is only ever downloaded once
ox.settings.log_console = False

# Known public Overpass mirrors, tried in order if the primary one fails.
# (Source: https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances)
_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# Hard ceiling on total time spent retrying, so a bad-server day fails fast
# (a couple of minutes) instead of silently grinding for 20-30+ minutes
# mid-demo.
_MAX_TOTAL_WAIT_SECONDS = 150


def _download_with_fallback(place_name, network_type, max_retries_per_mirror=2):
    """
    Tries graph_from_place against each known Overpass mirror in turn,
    retrying each one with a short backoff before moving to the next.
    Bails out entirely once _MAX_TOTAL_WAIT_SECONDS elapses, so a genuinely
    down/overloaded Overpass network fails within a couple of minutes
    rather than hanging indefinitely during a live demo. Raises the last
    error if every mirror/attempt fails (or the time budget runs out).
    """
    last_error = None
    start = time.monotonic()

    for mirror_url in _OVERPASS_MIRRORS:
        ox.settings.overpass_url = mirror_url

        for attempt in range(1, max_retries_per_mirror + 1):
            if time.monotonic() - start > _MAX_TOTAL_WAIT_SECONDS:
                raise RuntimeError(
                    f"Gave up downloading '{place_name}' after "
                    f"{_MAX_TOTAL_WAIT_SECONDS}s of retries across all "
                    f"Overpass mirrors — OSM's public servers seem to be "
                    f"down/overloaded right now. Try again shortly, or use "
                    f"a smaller/more specific place name. "
                    f"Last error: {last_error}"
                )
            try:
                return ox.graph_from_place(place_name, network_type=network_type)
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                wait = 3 * attempt
                print(
                    f"[graph_builder] {mirror_url} timed out/failed "
                    f"(attempt {attempt}/{max_retries_per_mirror}) — "
                    f"retrying in {wait}s..."
                )
                time.sleep(wait)
            except Exception:
                # Non-network errors (e.g. place name not found by Nominatim)
                # won't be fixed by retrying or switching mirrors.
                raise

    raise RuntimeError(
        f"Could not download road network for '{place_name}' — every Overpass "
        f"mirror timed out or failed. This is almost always OSM server load, "
        f"not your code. Try again in a minute, try a smaller/more specific "
        f"place name, or pre-cache this city ahead of time. "
        f"Last error: {last_error}"
    )


def build_city_graph(place_name: str = "Vijayawada, India", use_local_cache: bool = True):
    """
    Downloads a real drivable road network for the given place, adds
    speed + travel-time attributes, and restricts the graph to its LARGEST
    STRONGLY CONNECTED COMPONENT.

    Why this matters: osmnx road networks are directed (they respect one-way
    streets) and real downloads often include small disconnected fragments
    (service roads, gated areas, isolated segments) that aren't reachable
    from the main network in one or both directions. If a stop lands on one
    of those, there is genuinely no path to/from it. Restricting to the
    largest strongly connected component guarantees every remaining node is
    reachable from every other node, so shortest-path calls never fail.

    use_local_cache: if True (default) and a pre-baked GraphML file for this
    exact place_name exists under cached_graphs/ (see save_graph_cache /
    precache_graphs.py), it's loaded directly from disk -- no network call
    at all, so this can't fail even if Overpass is completely down. This is
    the recommended path for any city you plan to demo live. Set this False
    to force a fresh download even if a cached file exists (e.g. to refresh
    a stale cache).

    If no local cache hit, falls back to a live Overpass download (with
    retries + mirror fallback, see _download_with_fallback), which osmnx
    also caches to disk on its own (settings.use_cache=True above) -- so
    even an uncached city only ever costs one live download per machine.
    """
    if use_local_cache:
        cache_path = os.path.join(GRAPH_CACHE_DIR, f"{_slugify(place_name)}.graphml")
        if os.path.exists(cache_path):
            return ox.load_graphml(cache_path)

    G = _download_with_fallback(place_name, network_type="drive")
    G = ox.add_edge_speeds(G)        # estimates speed_kph per edge from road type
    G = ox.add_edge_travel_times(G)  # adds travel_time (seconds) per edge

    largest_scc_nodes = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_scc_nodes).copy()

    return G


def save_graph_cache(G, place_name):
    """
    Saves a fully-built graph (as returned by build_city_graph) to
    cached_graphs/<slugified place_name>.graphml. Commit this file to your
    repo and build_city_graph(place_name) will load it instantly from disk
    on every future run -- including during live judging -- with zero
    dependency on Overpass being reachable at that moment.

    Note: this saves the graph BEFORE apply_simulated_congestion is applied
    -- congestion is randomized per-run via a seed and applied fresh every
    time in app.py, so it deliberately isn't baked into the cached file.
    """
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
    path = os.path.join(GRAPH_CACHE_DIR, f"{_slugify(place_name)}.graphml")
    ox.save_graphml(G, path)
    return path


def apply_simulated_congestion(G, congestion_range=(1.0, 2.5), seed=None):
    """
    Since live traffic data isn't available for this project, congestion is
    simulated by scaling each edge's free-flow travel_time by a random
    factor. This produces the 'congested_time' attribute used as the actual
    optimization weight.

    congestion_range: (min_factor, max_factor). 1.0 = no congestion,
    2.5 = 2.5x slower than free flow.
    """
    if seed is not None:
        random.seed(seed)

    for _, _, data in G.edges(data=True):
        factor = random.uniform(*congestion_range)
        base_time = data.get("travel_time", 1.0)
        data["congested_time"] = base_time * factor

    return G


def pick_random_nodes(G, n, seed=None, exclude=None):
    """
    Utility for picking n random nodes from the graph (e.g. to stand in for
    a depot + customer locations when you don't have real delivery data yet).
    """
    if seed is not None:
        random.seed(seed)

    exclude = exclude or set()
    candidates = [node for node in G.nodes if node not in exclude]
    return random.sample(candidates, n)


def nodes_from_latlon(G, latlon_list):
    """
    Maps real-world coordinates to the nearest graph node — this is how you
    plug in YOUR OWN chosen stops (e.g. picked off Google Maps) instead of
    random ones.

    latlon_list: list of (latitude, longitude) tuples, e.g.
                 [(16.5062, 80.6480), (16.5100, 80.6300)]

    Returns a list of node IDs (plain Python ints), one per coordinate, in
    the same order.
    """
    lats = [pair[0] for pair in latlon_list]
    lons = [pair[1] for pair in latlon_list]
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)

    # osmnx returns a bare scalar for a single point but a numpy array for
    # multiple points -- NOT a list/tuple in either case. Node IDs also come
    # back as numpy int64, which (unlike a numpy array) IS technically
    # hashable, but behaves inconsistently with the plain python ints used
    # everywhere else (dict/set lookups, equality with pick_random_nodes'
    # output, etc.), so we normalize everything to plain int here.
    if isinstance(nodes, np.ndarray):
        return [int(n) for n in nodes]
    if isinstance(nodes, (list, tuple)):
        return [int(n) for n in nodes]
    return [int(nodes)]  # single bare scalar


if __name__ == "__main__":
    # Quick smoke test
    G = build_city_graph("Vijayawada, India")
    G = apply_simulated_congestion(G, seed=42)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    depot = pick_random_nodes(G, 1, seed=1)[0]
    customers = pick_random_nodes(G, 8, seed=2, exclude={depot})
    print(f"Depot: {depot}")
    print(f"Customers: {customers}")

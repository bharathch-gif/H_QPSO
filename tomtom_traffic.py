"""
tomtom_traffic.py

Fetches REAL, live traffic flow data from TomTom's Traffic Flow Segment
Data API and applies it to the actual road-graph edges most relevant to a
specific VRPProblem instance (the edges along free-flow shortest paths
between the depot and customers) -- so the optimizer's decisions reflect
genuine current road conditions, not the random simulated congestion model.

Why edges, not OD pairs (unlike live_traffic.py's Google integration):
TomTom's API reports live speed for a road segment nearest a given point,
which maps naturally onto our graph's EDGES (each edge already has a
free-flow 'travel_time', and we can scale it by the observed live
current-speed/free-flow-speed ratio) -- this is the same pattern
apply_simulated_congestion() already uses, just with a real ratio instead
of a random one, and it generalizes to ANY route that reuses these edges,
not just the specific depot-customer pairs queried.

Fetches run CONCURRENTLY (via a thread pool) rather than one at a time --
sequential fetching of even 40-150 edges at ~1-2s per HTTP round trip is
what caused multi-minute runtimes; in parallel, the same batch finishes in
roughly (n / max_workers) round trips instead of n.

HONEST LIMITATION: this was NOT tested against the real TomTom API from
this environment (no network access to external APIs from the sandbox this
was built in) -- only request-building, chunking/scoping, and error-
handling were verified against mocked responses. Test with your real API
key before relying on it for a live demo, and keep the simulated congestion
model as the default fallback.
"""

import requests
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"


def fetch_tomtom_flow(lat, lon, api_key, timeout=10):
    """
    Queries TomTom for the live flow data of the road segment nearest
    (lat, lon). Returns a dict with currentSpeed/freeFlowSpeed/
    currentTravelTime/freeFlowTravelTime on success, or None on any failure
    (bad key, network error, no segment found, etc.) -- never raises.
    """
    params = {"point": f"{lat},{lon}", "key": api_key}
    try:
        resp = requests.get(TOMTOM_FLOW_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None, "Request failed (network error, timeout, or bad response)"

    segment = data.get("flowSegmentData")
    if not segment:
        return None, f"No flow segment data in response: {data}"

    required = ("currentSpeed", "freeFlowSpeed")
    if any(k not in segment for k in required):
        return None, f"Response missing expected fields: {segment}"

    return segment, None


def _relevant_edges_for_problem(problem, max_edges=40):
    """
    Scopes live traffic fetching to edges actually likely to matter for this
    problem instance: the free-flow shortest paths between the depot and
    every customer, and between customer pairs -- rather than querying every
    edge in a potentially huge city graph, which would blow through rate
    limits and be extremely slow.
    """
    nodes_of_interest = [problem.depot] + list(problem.customers)
    edges = set()

    for i, a in enumerate(nodes_of_interest):
        for b in nodes_of_interest[i + 1:]:
            try:
                path = nx.shortest_path(problem.graph, a, b, weight="travel_time")
            except nx.NetworkXNoPath:
                continue
            for u, v in zip(path[:-1], path[1:]):
                edges.add((u, v))
                if len(edges) >= max_edges:
                    return list(edges)

    return list(edges)


def apply_tomtom_traffic_to_problem(problem, api_key, max_edges=40, max_workers=10):
    """
    Fetches live traffic flow for the edges relevant to `problem` (see
    _relevant_edges_for_problem) and updates their 'congested_time'
    attribute directly on the graph, scaled by the REAL observed
    current-speed/free-flow-speed ratio at that segment. Also clears the
    problem's internal shortest-path cost cache afterward, since it may
    already hold stale values computed before this update.

    This mutates the shared graph object for the specific edges queried --
    which is appropriate here, since live traffic is a genuine property of
    the road at this moment, not something that should be isolated per
    VRPProblem the way objective weighting is.

    Fetches run concurrently via a thread pool (max_workers) instead of one
    at a time -- this is the fix for multi-minute runtimes seen when
    fetching sequentially.

    Returns (success, errors) -- success is True if at least one edge was
    updated with real data; errors is a list of human-readable messages for
    anything that failed, so the caller can warn the user without crashing.
    """
    edges = _relevant_edges_for_problem(problem, max_edges=max_edges)
    if not edges:
        return False, ["No relevant edges found between depot and customers to query."]

    errors = []
    updated_count = 0

    def fetch_one(edge):
        u, v = edge
        try:
            lat = (problem.graph.nodes[u]["y"] + problem.graph.nodes[v]["y"]) / 2
            lon = (problem.graph.nodes[u]["x"] + problem.graph.nodes[v]["x"]) / 2
        except KeyError:
            return edge, None, "missing lat/lon node attributes — skipped."
        segment, error = fetch_tomtom_flow(lat, lon, api_key)
        return edge, segment, error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_one, edge) for edge in edges]
        for future in as_completed(futures):
            (u, v), segment, error = future.result()

            if segment is None:
                errors.append(f"Edge ({u},{v}): {error}")
                continue

            current_speed = segment["currentSpeed"]
            free_flow_speed = segment["freeFlowSpeed"]
            if current_speed <= 0 or free_flow_speed <= 0:
                errors.append(f"Edge ({u},{v}): invalid speed values, skipped.")
                continue

            congestion_ratio = free_flow_speed / current_speed  # >= 1, how much slower than free-flow

            # Apply to ALL parallel edges between u and v (osmnx MultiDiGraph
            # may have more than one), using the same observed ratio for each.
            # Also tag 'live_congestion_ratio' so the map/UI can show which
            # segments used REAL data vs. the simulated model.
            edge_data_dict = problem.graph.get_edge_data(u, v)
            if problem.graph.is_multigraph():
                for attrs in edge_data_dict.values():
                    free_flow_time = attrs.get("travel_time", 0.0)
                    attrs["congested_time"] = free_flow_time * congestion_ratio
                    attrs["live_congestion_ratio"] = congestion_ratio
            else:
                free_flow_time = edge_data_dict.get("travel_time", 0.0)
                edge_data_dict["congested_time"] = free_flow_time * congestion_ratio
                edge_data_dict["live_congestion_ratio"] = congestion_ratio

            updated_count += 1

    if updated_count == 0:
        errors.append("No edges could be updated with live data — leaving simulated model in place.")
        return False, errors

    # Any shortest-path costs already cached might be stale now.
    problem._cost_cache.clear()

    if updated_count < len(edges):
        errors.append(f"Partial live data: {updated_count}/{len(edges)} edges updated.")

    return True, errors

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
import networkx as nx
import osmnx as ox
import requests


def build_city_graph(place_name: str = "Vijayawada, India",
                      dist_meters: int = 5000,
                      max_retries: int = 3,
                      retry_delay_sec: int = 5):
    """
    Downloads a real drivable road network within `dist_meters` of the
    geocoded center of `place_name`, adds speed + travel-time attributes,
    and restricts the graph to its LARGEST STRONGLY CONNECTED COMPONENT.

    Why a fixed-radius query instead of the full place boundary: OSM's place
    lookup (Nominatim) often resolves a city name to a much larger
    administrative area than expected (e.g. an entire district/mandal
    instead of just the city) -- this was an OBSERVED real failure: a query
    "651 times" the configured max area, causing multi-minute downloads,
    truncated responses (ChunkedEncodingError), or outright crashes on
    resource-limited hosts like Streamlit Community Cloud. Querying a fixed
    radius around a geocoded point avoids this entirely and gives
    predictable download size regardless of how OSM defines that place's
    boundary.

    Retries with a short delay on transient network failures (connection
    drops, chunked-encoding errors, timeouts) since OSM's public Overpass
    servers occasionally fail requests under load -- this is usually
    resolved by simply trying again.

    Why the largest strongly connected component matters: osmnx road
    networks are directed (they respect one-way streets) and real downloads
    often include small disconnected fragments (service roads, gated areas,
    isolated segments) that aren't reachable from the main network in one or
    both directions. If a stop lands on one of those, there is genuinely no
    path to/from it. Restricting to the largest strongly connected component
    guarantees every remaining node is reachable from every other node, so
    shortest-path calls never fail.
    """
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            G = ox.graph_from_address(
                place_name, dist=dist_meters, network_type="drive"
            )
            break
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(retry_delay_sec * attempt)
                continue
            raise RuntimeError(
                f"Failed to download the road network for '{place_name}' "
                f"after {max_retries} attempts due to a network/OSM server "
                f"issue: {e}. This is usually transient -- try again in a "
                f"moment, check your internet connection, or try a "
                f"different place name."
            ) from e

    G = ox.add_edge_speeds(G)        # estimates speed_kph per edge from road type
    G = ox.add_edge_travel_times(G)  # adds travel_time (seconds) per edge

    largest_scc_nodes = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_scc_nodes).copy()

    return G


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

    Returns a list of node IDs, one per coordinate, in the same order.
    """
    lats = [pair[0] for pair in latlon_list]
    lons = [pair[1] for pair in latlon_list]
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)
    if not isinstance(nodes, (list, tuple)):
        nodes = [nodes]  # osmnx returns a bare value instead of a list for a single point
    return list(nodes)


if __name__ == "__main__":
    # Quick smoke test
    G = build_city_graph("Vijayawada, India")
    G = apply_simulated_congestion(G, seed=42)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    depot = pick_random_nodes(G, 1, seed=1)[0]
    customers = pick_random_nodes(G, 8, seed=2, exclude={depot})
    print(f"Depot: {depot}")
    print(f"Customers: {customers}")

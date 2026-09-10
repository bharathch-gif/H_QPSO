"""
utils.py

Plotting helpers: convergence curves (for comparing algorithms) and route
maps (for visualizing the actual chosen route on the city graph).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_convergence(all_results: dict, title="Convergence Comparison"):
    """
    all_results: {"QPSO": [ {convergence: [...]}, ... ], "PSO": [...], "GA": [...]}
    Plots the mean convergence curve (averaged across runs) for each algorithm.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, results in all_results.items():
        curves = np.array([r["convergence"] for r in results])
        mean_curve = curves.mean(axis=0)
        ax.plot(mean_curve, label=name)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness found so far (lower is better)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_route_on_graph(G, route_with_depot, weight_attr="congested_time", ax=None):
    """
    route_with_depot: full STOP sequence including depot at start and end,
    e.g. [depot, c1, c3, c2, depot]

    Draws the graph faintly and highlights the chosen route on top. Crucially,
    this stitches together the actual shortest ROAD path between each pair of
    consecutive stops (using the same weight the optimizer used), rather than
    drawing a straight line between stops — a straight line would cut through
    buildings/lakes and wouldn't represent the real driving route at all.

    NOTE: this static matplotlib version is kept for backward compatibility
    (e.g. quick scripts) but the Streamlit app uses build_interactive_traffic_map
    instead, since it can actually show congestion color-coding.
    """
    import osmnx as ox

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ox.plot_graph(G, ax=ax, show=False, close=False,
                  node_size=0, edge_color="#cccccc", edge_linewidth=0.5)

    # Stitch together the real road-following path across all stops
    full_path_nodes = [route_with_depot[0]]
    for a, b in zip(route_with_depot[:-1], route_with_depot[1:]):
        try:
            segment = nx.shortest_path(G, a, b, weight=weight_attr)
        except nx.NetworkXNoPath:
            # Shouldn't happen if the graph was built via build_city_graph()
            # (which restricts to the largest strongly connected component),
            # but fall back to a straight line rather than crashing the app.
            segment = [a, b]
        full_path_nodes.extend(segment[1:])  # skip first node, already in the list

    path_xs = [G.nodes[n]["x"] for n in full_path_nodes]
    path_ys = [G.nodes[n]["y"] for n in full_path_nodes]
    ax.plot(path_xs, path_ys, color="red", linewidth=2, zorder=5)

    # Mark the actual delivery stops on top (not every intermediate road node)
    stop_xs = [G.nodes[n]["x"] for n in route_with_depot]
    stop_ys = [G.nodes[n]["y"] for n in route_with_depot]
    ax.plot(stop_xs, stop_ys, "o", color="red", markersize=6, zorder=6)

    ax.plot(stop_xs[0], stop_ys[0], color="green", marker="*", markersize=18, zorder=7, label="Depot")
    ax.legend()

    return ax


def _min_edge_attrs(G, u, v, weight_attr):
    """
    Picks the representative attribute dict for edge (u,v), handling both
    simple graphs and osmnx's MultiDiGraph (parallel edges) — mirrors the
    same logic VRPProblem._min_edge_data uses, so the map shows the SAME
    edge the optimizer actually costed, not an arbitrary parallel edge.
    """
    edge_data = G.get_edge_data(u, v)
    if not edge_data:
        return {}
    if not G.is_multigraph():
        return edge_data

    if callable(weight_attr):
        return min(edge_data.values(), key=lambda d: weight_attr(u, v, {0: d}))
    return min(edge_data.values(), key=lambda d: d.get(weight_attr, float("inf")))


def _congestion_color(ratio):
    """
    Maps a congestion ratio (1.0 = free-flowing, higher = more congested) to
    a green -> yellow -> red color, clamped to a 1.0-3.0x range so a single
    very extreme segment doesn't wash out the rest of the map's contrast.
    """
    ratio = min(max(ratio, 1.0), 3.0)
    frac = (ratio - 1.0) / 2.0  # 0.0 (free-flowing) -> 1.0 (heavily congested)

    if frac < 0.5:
        # green -> yellow
        t = frac / 0.5
        r, g, b = int(40 + t * 215), 200, 60
    else:
        # yellow -> red
        t = (frac - 0.5) / 0.5
        r, g, b = 255, int(200 - t * 200), 60 - int(t * 60)

    return f"#{max(0,r):02x}{max(0,g):02x}{max(0,b):02x}"


def build_interactive_traffic_map(G, route_with_depot, weight_attr="congested_time"):
    """
    Builds an interactive Folium map of the chosen route, with each road
    segment colored by its congestion level (green = free-flowing, red =
    heavily congested) -- so live traffic data (when applied via TomTom/
    Google) becomes visually obvious instead of being an invisible number
    that only affects which route gets picked.

    Segments updated with REAL live data (tagged 'live_congestion_ratio' by
    tomtom_traffic.py) are labeled as such in their tooltip; segments still
    on the simulated model are labeled accordingly, so it's honest about
    which color is backed by real data vs. simulation.

    Returns (folium.Map, has_live_data) -- has_live_data is True if at least
    one drawn segment used real fetched traffic data, so the caller can show
    an accurate "this map includes live data" indicator.
    """
    import folium

    full_path_nodes = [route_with_depot[0]]
    for a, b in zip(route_with_depot[:-1], route_with_depot[1:]):
        try:
            segment = nx.shortest_path(G, a, b, weight=weight_attr)
        except nx.NetworkXNoPath:
            segment = [a, b]
        full_path_nodes.extend(segment[1:])

    depot_node = route_with_depot[0]
    depot_lat, depot_lon = G.nodes[depot_node]["y"], G.nodes[depot_node]["x"]
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=13, tiles="OpenStreetMap")

    has_live_data = False

    for u, v in zip(full_path_nodes[:-1], full_path_nodes[1:]):
        attrs = _min_edge_attrs(G, u, v, weight_attr)

        live_ratio = attrs.get("live_congestion_ratio")
        if live_ratio is not None:
            has_live_data = True
            ratio = live_ratio
            tooltip = f"LIVE traffic data — {ratio:.2f}x slower than free-flow"
        else:
            free_flow = attrs.get("travel_time", 1.0) or 1.0
            congested = attrs.get("congested_time", free_flow)
            ratio = (congested / free_flow) if free_flow else 1.0
            tooltip = f"Simulated congestion — {ratio:.2f}x slower than free-flow"

        color = _congestion_color(ratio)
        u_coord = (G.nodes[u]["y"], G.nodes[u]["x"])
        v_coord = (G.nodes[v]["y"], G.nodes[v]["x"])
        folium.PolyLine(
            [u_coord, v_coord], color=color, weight=5, opacity=0.85, tooltip=tooltip
        ).add_to(m)

    for i, node in enumerate(route_with_depot[:-1]):  # skip duplicate final depot
        lat, lon = G.nodes[node]["y"], G.nodes[node]["x"]
        if i == 0:
            folium.Marker(
                [lat, lon], icon=folium.Icon(color="green", icon="home"), tooltip="Depot"
            ).add_to(m)
        else:
            folium.Marker(
                [lat, lon], icon=folium.Icon(color="red", icon="flag"), tooltip=f"Stop {i}"
            ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 13px;">
      <b>Congestion level</b><br>
      <span style="color:#28c83c;">■</span> Free-flowing<br>
      <span style="color:#ffc83c;">■</span> Moderate<br>
      <span style="color:#ff3c3c;">■</span> Heavy
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m, has_live_data

"""
precache_graphs.py

Run this ONCE, ahead of time (with a working internet connection), for
every city you plan to demo with. It downloads and processes the full road
network and saves it to cached_graphs/<slug>.graphml.

Commit the resulting cached_graphs/ folder to your repo. From then on,
build_city_graph() loads these cities straight from disk -- no network
call, no dependency on Overpass being reachable or fast -- which is what
you want for a live judging demo. Any city NOT pre-cached still works
exactly as before (falls back to a live Overpass download with retries).

Usage:
    python3 precache_graphs.py "Vijayawada, India" "Hyderabad, India"

With no arguments, caches just the app's default demo city.
"""

import sys
from graph_builder import build_city_graph, save_graph_cache


def main():
    cities = sys.argv[1:] or ["Vijayawada, India"]

    for city in cities:
        print(f"Downloading and processing: {city} ...")
        try:
            G = build_city_graph(city, use_local_cache=False)  # force a fresh download
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        path = save_graph_cache(G, city)
        print(f"  -> cached to {path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    print("\nDone. Commit the cached_graphs/ folder to your repo so this "
          "works during live judging without needing Overpass at all.")


if __name__ == "__main__":
    main()

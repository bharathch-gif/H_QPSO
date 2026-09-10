"""
live_traffic.py

Fetches REAL, traffic-aware travel times from Google's Distance Matrix API
and uses them to override the optimizer's cost function for the depot and
customer stops in a specific VRPProblem instance.

Design choice: Google's Distance Matrix API returns a travel time directly
between coordinate pairs (using its own real-time routing), rather than a
per-road-segment traffic flow like some other providers. This fits our use
case well -- one API call (or a few, chunked) fetches the entire depot <->
customers travel-time matrix directly, which is far fewer requests than
querying every road-graph edge individually.

HONEST LIMITATION: Google's Distance Matrix API gives a travel TIME, not the
actual road-by-road path taken. This means route_metrics() (the distance /
free-flow-time / congestion-delay breakdown) still reflects the internal
SIMULATED road graph, not Google's live number -- only the optimizer's cost
function itself (route_cost, and therefore which route wins) is driven by
live data when this is enabled. A fully consistent live breakdown would
require Google's Directions/Routes API per candidate route, which is a
further extension, not implemented here.

This was NOT tested against the real Google API from this environment (no
network access to external APIs from the sandbox this was built in) -- only
the request-building, chunking, and error-handling logic were verified
against mocked responses. Test this against your real API key before
relying on it for a live demo, and keep the simulated congestion model as
the default fallback.
"""

import requests

GOOGLE_DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"


def _chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def fetch_google_traffic_matrix(coords, api_key, departure_time="now",
                                 chunk_size=10, timeout=10):
    """
    coords: list of (lat, lon) tuples, in the order you want the matrix indexed.
    departure_time: "now" for live traffic, or a Unix timestamp for a future time.

    Returns (matrix, errors):
      matrix[i][j] = travel time in seconds from coords[i] to coords[j] WITH
                     live traffic, or None if that specific pair failed (the
                     caller should fall back to the simulated/road-graph
                     estimate for that pair rather than treat this as fatal).
      errors:        list of human-readable messages for anything that went
                     wrong, so the caller can warn the user without crashing.

    Chunked into blocks of chunk_size x chunk_size to stay within Google's
    element-count limits per request (100 elements per request on the
    standard tier: origins_in_chunk * destinations_in_chunk <= 100 when
    chunk_size <= 10).
    """
    n = len(coords)
    matrix = [[None] * n for _ in range(n)]
    errors = []

    index_chunks = list(_chunked(list(range(n)), chunk_size))

    for origin_idx_chunk in index_chunks:
        origins_param = "|".join(f"{coords[i][0]},{coords[i][1]}" for i in origin_idx_chunk)

        for dest_idx_chunk in index_chunks:
            destinations_param = "|".join(f"{coords[j][0]},{coords[j][1]}" for j in dest_idx_chunk)

            params = {
                "origins": origins_param,
                "destinations": destinations_param,
                "key": api_key,
                "departure_time": departure_time,
                "traffic_model": "best_guess",
                "mode": "driving",
            }

            try:
                resp = requests.get(GOOGLE_DISTANCE_MATRIX_URL, params=params, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                errors.append(f"Request failed for a chunk: {e}")
                continue

            if data.get("status") != "OK":
                errors.append(
                    f"Google API returned status={data.get('status')}: "
                    f"{data.get('error_message', '(no further detail)')}"
                )
                continue

            rows = data.get("rows", [])
            for oi, row in zip(origin_idx_chunk, rows):
                elements = row.get("elements", [])
                for dj, element in zip(dest_idx_chunk, elements):
                    if element.get("status") != "OK":
                        errors.append(f"No route data for pair ({oi},{dj}): {element.get('status')}")
                        continue

                    duration_info = element.get("duration_in_traffic", element.get("duration"))
                    if duration_info is None:
                        errors.append(f"Missing duration for pair ({oi},{dj})")
                        continue

                    matrix[oi][dj] = duration_info["value"]  # seconds

    return matrix, errors


def apply_live_traffic_to_problem(problem, api_key, departure_time="now"):
    """
    Fetches real-time, traffic-aware travel times between the depot and all
    customers in `problem`, and overrides the optimizer's internal cost
    cache for those pairs -- so QPSO/PSO/GA/OR-Tools make decisions based on
    ACTUAL current traffic conditions rather than the simulated congestion
    model, for this specific problem instance.

    Returns (success, errors):
      success: True if at least SOME live data was fetched and applied.
      errors:  list of anything that went wrong (including partial
               failures for individual pairs), so the caller can decide
               whether/how to warn the user -- this never raises, even on
               total failure, so a bad API key or network issue degrades
               gracefully back to the simulated model rather than crashing
               the app.
    """
    node_list = [problem.depot] + list(problem.customers)

    try:
        coords = [(problem.graph.nodes[n]["y"], problem.graph.nodes[n]["x"]) for n in node_list]
    except KeyError:
        return False, ["Graph nodes are missing lat/lon attributes — is this a real osmnx graph?"]

    matrix, errors = fetch_google_traffic_matrix(coords, api_key, departure_time=departure_time)

    applied_count = 0
    n = len(node_list)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = matrix[i][j]
            if value is not None:
                problem._cost_cache[(node_list[i], node_list[j])] = float(value)
                applied_count += 1

    if applied_count == 0:
        errors.append("No live traffic data could be applied — falling back entirely to the simulated model.")
        return False, errors

    total_possible = n * (n - 1)
    if applied_count < total_possible:
        errors.append(
            f"Partial live data: {applied_count}/{total_possible} pairs updated, "
            f"the rest fall back to the simulated model."
        )

    return True, errors

"""
vrp_problem.py

Mathematical formulation of the Vehicle Routing Problem.

  Objective:  minimize total travel time across all vehicle sub-routes
  Constraint: no single vehicle trip may exceed its capacity

route_cost() is the objective function every optimizer (QPSO, PSO, GA) calls
to score a candidate solution. Constraint handling uses the penalty-function
method: infeasible solutions aren't rejected outright, they're just scored
worse, so the search naturally moves away from them over time.
"""

import networkx as nx

INFEASIBLE_PENALTY = 1e7  # large constant added when a customer can never fit


class VRPProblem:
    def __init__(self, graph, depot_node, customer_nodes,
                 demands: dict = None, vehicle_capacity: float = None,
                 weight_attr: str = "congested_time",
                 objective_weights: dict = None):
        """
        graph:            networkx graph with weighted edges
        depot_node:       node ID vehicles start/end at
        customer_nodes:   list of node IDs that must be visited
        demands:          optional {node_id: demand} for capacity constraint
        vehicle_capacity: optional max demand a single vehicle trip can carry
        weight_attr:      which edge attribute to use as the cost, when
                          objective_weights is NOT given (default behavior:
                          minimize congestion-weighted travel time only).
        objective_weights: optional dict like {"time": 1.0, "distance": 0.3,
                          "congestion": 0.5} to genuinely trade off between
                          minimizing distance, free-flow time, and congestion
                          delay as three SEPARATE, differently-weighted
                          quantities, rather than one number that always
                          blends time+congestion together with a fixed,
                          implicit weighting. Each component is scaled by
                          its own mean value across the graph first, so the
                          weights are meaningfully comparable despite very
                          different natural units (meters vs seconds).
                          When given, this OVERRIDES weight_attr.
        """
        self.graph = graph
        self.depot = depot_node
        self.customers = customer_nodes
        self.demands = demands or {node: 1 for node in customer_nodes}
        self.capacity = vehicle_capacity
        self.objective_weights = objective_weights

        if objective_weights is not None:
            self.weight_attr = self._build_weighted_cost_function(objective_weights)
        else:
            self.weight_attr = weight_attr

        # Cache pairwise shortest-path costs so we don't recompute Dijkstra
        # thousands of times during optimization.
        self._cost_cache = {}

    def _build_weighted_cost_function(self, objective_weights):
        """
        Builds a CALLABLE edge-weight function (networkx supports passing a
        function instead of an attribute name string to shortest_path /
        shortest_path_length). This avoids mutating the shared graph object
        directly, which matters because the same graph is reused across
        multiple VRPProblem instances (e.g. in the Streamlit app) — writing
        a 'blended_cost' attribute onto the graph itself would let different
        problems with different weights silently clobber each other.
        """
        w_time = objective_weights.get("time", 0.0)
        w_distance = objective_weights.get("distance", 0.0)
        w_congestion = objective_weights.get("congestion", 0.0)

        # Scale each raw component by its own mean across the graph, so the
        # weights are comparable despite very different natural units.
        distances, times, delays = [], [], []
        for _, _, data in self.graph.edges(data=True):
            distances.append(data.get("length", 0.0))
            free_flow = data.get("travel_time", 0.0)
            times.append(free_flow)
            congested = data.get("congested_time", free_flow)
            delays.append(max(congested - free_flow, 0.0))

        mean_distance = (sum(distances) / len(distances)) if distances else 0.0
        mean_time = (sum(times) / len(times)) if times else 0.0
        mean_delay = (sum(delays) / len(delays)) if delays else 0.0
        # Guard against division by zero (e.g. synthetic test graphs with no
        # 'length'/'travel_time' attributes set).
        mean_distance = mean_distance or 1.0
        mean_time = mean_time or 1.0
        mean_delay = mean_delay or 1.0

        def blended_cost(data):
            distance_norm = data.get("length", 0.0) / mean_distance
            free_flow = data.get("travel_time", 0.0)
            time_norm = free_flow / mean_time
            congested = data.get("congested_time", free_flow)
            delay_norm = max(congested - free_flow, 0.0) / mean_delay
            return (w_time * time_norm) + (w_distance * distance_norm) + (w_congestion * delay_norm)

        is_multigraph = self.graph.is_multigraph()

        def weight_function(u, v, d):
            if is_multigraph:
                # d is {edge_key: attr_dict} for parallel edges — take the
                # cheapest, consistent with how networkx treats string weights.
                return min(blended_cost(attrs) for attrs in d.values())
            return blended_cost(d)

        return weight_function

    def _edge_cost(self, a, b):
        """Shortest-path cost between two nodes, using the cache."""
        key = (a, b)
        if key not in self._cost_cache:
            try:
                self._cost_cache[key] = nx.shortest_path_length(
                    self.graph, a, b, weight=self.weight_attr
                )
            except nx.NetworkXNoPath:
                self._cost_cache[key] = INFEASIBLE_PENALTY
        return self._cost_cache[key]

    def _split_into_trips(self, route_order):
        """
        Greedily splits a full customer visiting order into capacity-feasible
        sub-trips (each sub-trip starts and ends at the depot). This is the
        'split algorithm' approach to capacitated VRP: the optimizer searches
        over orderings, and trip boundaries are derived deterministically.
        """
        if self.capacity is None:
            return [route_order]  # single vehicle, no capacity limit

        trips, current_trip, current_load = [], [], 0
        for node in route_order:
            demand = self.demands.get(node, 1)
            if demand > self.capacity:
                # This customer can never be served by any vehicle — infeasible.
                return None
            if current_load + demand > self.capacity:
                trips.append(current_trip)
                current_trip, current_load = [], 0
            current_trip.append(node)
            current_load += demand

        if current_trip:
            trips.append(current_trip)
        return trips

    def _min_edge_data(self, u, v):
        """Handles both simple graphs and osmnx's MultiDiGraph (parallel edges)."""
        if self.graph.is_multigraph():
            edges = self.graph.get_edge_data(u, v)
            if not edges:
                return {}
            if callable(self.weight_attr):
                # weight_attr is our custom multi-objective function, which
                # expects (u, v, d) with d as the dict-of-parallel-edges —
                # but here we just need each candidate's OWN scalar cost to
                # rank them, so call the function with a single-edge dict.
                return min(edges.values(), key=lambda d: self.weight_attr(u, v, {0: d}))
            return min(edges.values(), key=lambda d: d.get(self.weight_attr, float("inf")))
        return self.graph.get_edge_data(u, v) or {}

    def nearest_neighbor_route(self):
        """
        Builds a route using the classic greedy nearest-neighbor heuristic:
        starting at the depot, repeatedly visit whichever unvisited customer
        is cheapest to reach next. This is a fast, decent (not optimal)
        starting point — used to SEED part of the swarm's initial population
        instead of starting every particle from a fully random route, which
        speeds up early convergence since the swarm starts closer to a
        reasonable solution.
        """
        unvisited = set(self.customers)
        current = self.depot
        route = []

        while unvisited:
            nearest = min(unvisited, key=lambda c: self._edge_cost(current, c))
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return route

    def build_cost_matrix(self):
        """
        Returns (matrix, node_list) where node_list[0] is the depot and the
        rest are self.customers in order. matrix[i][j] is the shortest-path
        cost from node_list[i] to node_list[j]. Needed by solvers like
        OR-Tools that expect an explicit distance/cost matrix rather than
        calling route_cost() on a full permutation directly.
        """
        node_list = [self.depot] + list(self.customers)
        n = len(node_list)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self._edge_cost(node_list[i], node_list[j])
        return matrix, node_list

    def route_cost(self, route_order):
        """
        route_order: list of customer node IDs in visiting order (a permutation
        of self.customers). Returns total cost, with penalties added for any
        capacity violation.
        """
        trips = self._split_into_trips(route_order)
        if trips is None:
            return INFEASIBLE_PENALTY

        total_cost = 0.0
        for trip in trips:
            if not trip:
                continue
            full_path = [self.depot] + trip + [self.depot]
            for a, b in zip(full_path[:-1], full_path[1:]):
                total_cost += self._edge_cost(a, b)

        return total_cost

    def route_metrics(self, route_order):
        """
        Reports distance, free-flow travel time, and congestion delay as
        SEPARATE figures for a chosen route — used for reporting/slides only
        (not called during optimization, since it's more expensive than
        route_cost). This satisfies the requirement to show total travel
        time, distance, and congestion as distinct minimized quantities,
        even though the optimizer itself searches using one blended weight.
        """
        trips = self._split_into_trips(route_order)
        if trips is None:
            return None

        total_distance_m = 0.0
        total_free_flow_s = 0.0
        total_congested_s = 0.0

        for trip in trips:
            if not trip:
                continue
            full_path = [self.depot] + trip + [self.depot]
            for a, b in zip(full_path[:-1], full_path[1:]):
                path_nodes = nx.shortest_path(self.graph, a, b, weight=self.weight_attr)
                for x, y in zip(path_nodes[:-1], path_nodes[1:]):
                    edata = self._min_edge_data(x, y)
                    total_distance_m += edata.get("length", 0.0)
                    total_free_flow_s += edata.get("travel_time", 0.0)
                    total_congested_s += edata.get("congested_time", edata.get("travel_time", 0.0))

        return {
            "total_distance_m": total_distance_m,
            "total_free_flow_time_s": total_free_flow_s,
            "total_congested_time_s": total_congested_s,
            "congestion_delay_s": total_congested_s - total_free_flow_s,
        }

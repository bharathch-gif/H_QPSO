"""
baselines.py

Classical algorithms benchmarked against QPSO. Both use the exact same
problem interface (VRPProblem.route_cost) and the exact same random-key
encoding, so the only variable being compared is the search strategy itself.

  - run_pso:   standard velocity-based Particle Swarm Optimization
  - run_ga:    Genetic Algorithm with tournament selection, crossover, mutation
  - run_exact: brute-force optimal solution, for validating the heuristics
               against the true optimum on SMALL instances only
"""

import itertools
import math
import numpy as np
from encoding import decode_position, random_position


# ---------------------------------------------------------------------------
# Classical PSO
# ---------------------------------------------------------------------------

class PSOParticle:
    def __init__(self, dim):
        self.position = random_position(dim)
        self.velocity = np.random.uniform(-0.1, 0.1, size=dim)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float("inf")


def run_pso(problem, n_particles=30, n_iterations=100,
            inertia=0.7, c1=1.5, c2=1.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dim = len(problem.customers)
    particles = [PSOParticle(dim) for _ in range(n_particles)]

    gbest_position, gbest_fitness = None, float("inf")
    convergence_log = []

    for _ in range(n_iterations):
        for p in particles:
            route = decode_position(p.position, problem.customers)
            fitness = problem.route_cost(route)

            if fitness < p.pbest_fitness:
                p.pbest_fitness, p.pbest_position = fitness, p.position.copy()
            if fitness < gbest_fitness:
                gbest_fitness, gbest_position = fitness, p.position.copy()

        for p in particles:
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            p.velocity = (
                inertia * p.velocity
                + c1 * r1 * (p.pbest_position - p.position)
                + c2 * r2 * (gbest_position - p.position)
            )
            p.position = p.position + p.velocity

        convergence_log.append(gbest_fitness)

    best_route = decode_position(gbest_position, problem.customers)
    return best_route, gbest_fitness, convergence_log


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

def run_ga(problem, population_size=30, n_iterations=100,
           mutation_rate=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dim = len(problem.customers)
    population = [random_position(dim) for _ in range(population_size)]

    gbest_position, gbest_fitness = None, float("inf")
    convergence_log = []

    def fitness_of(position):
        route = decode_position(position, problem.customers)
        return problem.route_cost(route)

    def tournament_select(pop, fitnesses, k=3):
        idxs = np.random.choice(len(pop), k, replace=False)
        best_idx = min(idxs, key=lambda i: fitnesses[i])
        return pop[best_idx]

    for _ in range(n_iterations):
        fitnesses = [fitness_of(ind) for ind in population]

        gen_best_idx = int(np.argmin(fitnesses))
        if fitnesses[gen_best_idx] < gbest_fitness:
            gbest_fitness = fitnesses[gen_best_idx]
            gbest_position = population[gen_best_idx].copy()

        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_select(population, fitnesses)
            parent2 = tournament_select(population, fitnesses)

            # Single-point crossover on the priority vector
            point = np.random.randint(1, dim)
            child = np.concatenate([parent1[:point], parent2[point:]])

            # Mutation: randomly reset some priority values
            mutation_mask = np.random.rand(dim) < mutation_rate
            child[mutation_mask] = np.random.rand(mutation_mask.sum())

            new_population.append(child)

        population = new_population
        convergence_log.append(gbest_fitness)

    best_route = decode_position(gbest_position, problem.customers)
    return best_route, gbest_fitness, convergence_log


# ---------------------------------------------------------------------------
# Exact method (brute force) — for validation against the true optimum
# ---------------------------------------------------------------------------

def run_exact(problem, max_customers=8, seed=None):
    """
    Checks EVERY possible visiting order and returns the true optimal route.
    Only feasible for small instances (permutations grow as n!), so this is
    meant for validating QPSO/PSO/GA quality on a small subset of customers,
    NOT for production-scale routing. This satisfies the problem statement's
    requirement to benchmark against exact methods, not just other
    metaheuristics.

    Returns (best_route, best_fitness, convergence_log) — convergence_log is
    a single-value list since there's no iterative improvement to plot.
    """
    n = len(problem.customers)
    if n > max_customers:
        raise ValueError(
            f"run_exact is only feasible for <= {max_customers} customers "
            f"(got {n}). Build a smaller VRPProblem with a subset of "
            f"customers to compare against the true optimum."
        )

    best_route, best_fitness = None, float("inf")
    for perm in itertools.permutations(problem.customers):
        cost = problem.route_cost(list(perm))
        if cost < best_fitness:
            best_fitness, best_route = cost, list(perm)

    return best_route, best_fitness, [best_fitness]


# ---------------------------------------------------------------------------
# OR-Tools — Google's production-grade VRP solver, used as a strong reference
# ---------------------------------------------------------------------------

def run_ortools(problem, time_limit_seconds=5, seed=None, **kwargs):
    """
    Solves the same VRP instance using Google OR-Tools' routing library —
    a well-established, production-grade solver, not a brute-force exact
    method. Unlike run_exact (limited to ~8 customers), OR-Tools can handle
    the FULL problem size, so this is a much stronger reference point for
    "how close to the best achievable solution are QPSO/PSO/GA getting?"
    across realistic instance sizes.

    `seed` is accepted but unused (kept for interface compatibility with the
    other baselines, which are called uniformly elsewhere).

    Returns (best_route, best_fitness, convergence_log) — convergence_log is
    a single-value list, since OR-Tools doesn't expose iteration-by-iteration
    progress the way the swarm/evolutionary methods do.
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    matrix, node_list = problem.build_cost_matrix()
    n = len(node_list)
    depot_index = 0

    if problem.capacity is not None:
        total_demand = sum(problem.demands.get(c, 1) for c in problem.customers)
        num_vehicles = max(1, math.ceil(total_demand / problem.capacity) + 1)
    else:
        num_vehicles = 1

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # OR-Tools requires integer costs, so scale up and round
    SCALE = 1000

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(matrix[from_node][to_node] * SCALE)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    if problem.capacity is not None:
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            if from_node == depot_index:
                return 0
            return int(problem.demands.get(node_list[from_node], 1))

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0,
            [int(problem.capacity)] * num_vehicles, True, "Capacity",
        )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        return None, float("inf"), [float("inf")]

    route_order = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != depot_index:
                route_order.append(node_list[node])
            index = solution.Value(routing.NextVar(index))

    total_cost = solution.ObjectiveValue() / SCALE
    return route_order, total_cost, [total_cost]

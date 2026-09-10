"""
local_search.py

2-opt local search — the "H" (Hybrid) in H-QPSO.

QPSO's swarm search is good at globally exploring the solution space but,
like most metaheuristics, can leave small easily-fixable inefficiencies in
its best-found route (e.g. two edges that visibly cross each other on a
map). 2-opt is a classic local-search polish step: it looks at every pair
of edges in a route and checks whether reversing the segment between them
shortens the total route. This can't discover a completely different route
structure (that's QPSO's job), but it reliably squeezes out these small,
"obviously fixable" inefficiencies that a purely stochastic swarm search
can be slow to stumble onto by chance.
"""


def two_opt(route, problem, max_passes=1):
    """
    route:      list of customer node IDs (a candidate route)
    problem:    a VRPProblem instance (used to score candidate routes)
    max_passes: how many full sweeps over the route to attempt; each pass
                only keeps going while it's still finding improvements

    Returns (refined_route, refined_cost). If no improving move exists,
    returns the original route unchanged.
    """
    best_route = list(route)
    best_cost = problem.route_cost(best_route)

    n = len(best_route)
    for _ in range(max_passes):
        improved_this_pass = False

        for i in range(n - 1):
            for j in range(i + 1, n):
                # 2-opt move: reverse the segment between i and j
                candidate = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                candidate_cost = problem.route_cost(candidate)

                if candidate_cost < best_cost:
                    best_route, best_cost = candidate, candidate_cost
                    improved_this_pass = True

        if not improved_this_pass:
            break  # converged — no more improving moves found

    return best_route, best_cost

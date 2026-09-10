"""
qpso.py

Quantum Particle Swarm Optimization (Sun, Feng & Xu, 2004), hybridized with
2-opt local search (H-QPSO).

Unlike classical PSO, particles have no velocity. Instead each particle's
next position is sampled from a probability distribution ("quantum well")
centered at an attractor point pulled between its personal best and the
swarm's global best, with the spread of that distribution controlled by
the swarm's mean best position (mbest).

The hybrid part: after each swarm update, the current best route is polished
with 2-opt local search (local_search.py). QPSO handles global exploration
(discovering good overall route structures); 2-opt handles local refinement
(fixing small inefficiencies within that structure). The refined route is
re-encoded and fed back in as the swarm's gbest, so it genuinely influences
the rest of the search — not just a cosmetic final polish step.
"""

import numpy as np
from encoding import decode_position, random_position, encode_route
from local_search import two_opt


class Particle:
    def __init__(self, dim):
        self.position = random_position(dim)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float("inf")


def run_qpso(problem, n_particles=30, n_iterations=100,
             beta_start=0.8, beta_end=0.3,
             apply_local_search=True, local_search_every=5,
             seed_with_nearest_neighbor=True,
             adaptive_beta=False, stagnation_patience=10, beta_boost_factor=1.6,
             seed=None):
    """
    problem:      a VRPProblem instance (must expose .customers and .route_cost)
    n_particles:  swarm size
    n_iterations: number of update rounds
    beta_start/beta_end: contraction-expansion coefficient. The BASE schedule
                  linearly anneals from exploration-heavy (high beta) to
                  exploitation-heavy (low beta) over the run. Defaults
                  (0.8 -> 0.3) were tuned empirically against classical PSO.
    adaptive_beta: if True, beta reacts to search STAGNATION on top of the
                  base schedule — if gbest hasn't improved for
                  `stagnation_patience` iterations, beta is temporarily
                  boosted (up to beta_start) to push the swarm back into
                  exploration mode. NOTE: empirically tested and found to
                  give NO measurable improvement on our benchmark problems
                  (with or without 2-opt hybridization) — kept available for
                  experimentation, but defaults to False since the simpler
                  fixed schedule performs at least as well. Don't claim this
                  as a proven improvement without re-testing on your actual
                  problem instances.
    stagnation_patience: how many iterations without improvement before
                  triggering a beta boost.
    beta_boost_factor: multiplier applied to the base beta when stagnation
                  triggers a boost (capped at beta_start).
    apply_local_search: whether to hybridize with 2-opt (H-QPSO). Set False
                  to run plain QPSO (e.g. for a fair ablation comparison).
    local_search_every: how often (in iterations) to run the 2-opt polish
                  step on the current best route. Running it every single
                  iteration is the "purest" reading of the hybrid pipeline,
                  but 2-opt is O(n^2) route evaluations per pass — for
                  larger problems that adds up fast across 100+ iterations,
                  so this defaults to every 5th iteration as a runtime/
                  quality tradeoff. Set to 1 for maximum refinement if your
                  problem size allows it.
    seed_with_nearest_neighbor: if True, one particle starts from a greedy
                  nearest-neighbor route (with small random jitter for
                  diversity) instead of a fully random position — gives the
                  swarm a decent head start instead of searching blind from
                  iteration 0.

    Returns: (best_route, best_fitness, convergence_log)
    """
    if seed is not None:
        np.random.seed(seed)

    dim = len(problem.customers)
    particles = [Particle(dim) for _ in range(n_particles)]

    if seed_with_nearest_neighbor and n_particles > 0:
        nn_route = problem.nearest_neighbor_route()
        nn_position = encode_route(nn_route, problem.customers)
        # Small jitter keeps this particle from being a rigid duplicate while
        # still starting much closer to a decent solution than pure random.
        jitter = np.random.normal(0, 0.02, size=dim)
        particles[0].position = np.clip(nn_position + jitter, 0.0, 1.0)
        particles[0].pbest_position = particles[0].position.copy()

    gbest_position = None
    gbest_fitness = float("inf")
    convergence_log = []
    stagnation_counter = 0

    for iteration in range(n_iterations):
        # 1. Evaluate every particle's fitness and update personal/global bests
        improved_this_iteration = False
        for p in particles:
            route = decode_position(p.position, problem.customers)
            fitness = problem.route_cost(route)

            if fitness < p.pbest_fitness:
                p.pbest_fitness = fitness
                p.pbest_position = p.position.copy()

            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest_position = p.position.copy()
                improved_this_iteration = True

        # 2. Mean best position across the swarm
        mbest = np.mean([p.pbest_position for p in particles], axis=0)

        # 3. Base beta: linear anneal from exploration-heavy to exploitation-heavy
        beta = beta_start - (beta_start - beta_end) * (iteration / max(n_iterations - 1, 1))

        # 3b. Adaptive component: if the swarm has stagnated (no improvement
        #     for stagnation_patience iterations), boost beta back up to push
        #     exploration, then reset the counter to give it a fresh window.
        if adaptive_beta:
            stagnation_counter = 0 if improved_this_iteration else stagnation_counter + 1
            if stagnation_counter >= stagnation_patience:
                beta = min(beta_start, beta * beta_boost_factor)
                stagnation_counter = 0

        # 4. QPSO update: every particle's position
        for p in particles:
            phi = np.random.rand(dim)
            attractor = phi * p.pbest_position + (1 - phi) * gbest_position

            u = np.random.uniform(1e-6, 1.0, size=dim)  # avoid log(0)
            direction = np.where(np.random.rand(dim) > 0.5, 1.0, -1.0)

            p.position = attractor + direction * beta * np.abs(mbest - p.position) * np.log(1.0 / u)

        # 5. Hybrid refinement: polish the current best route with 2-opt,
        #    then feed the improved route back in as the swarm's gbest so
        #    it actually shapes the rest of the search (not just a final
        #    cosmetic pass).
        if apply_local_search and (iteration % local_search_every == 0):
            current_best_route = decode_position(gbest_position, problem.customers)
            refined_route, refined_cost = two_opt(current_best_route, problem, max_passes=1)

            if refined_cost < gbest_fitness:
                gbest_fitness = refined_cost
                gbest_position = encode_route(refined_route, problem.customers)

        convergence_log.append(gbest_fitness)

    best_route = decode_position(gbest_position, problem.customers)
    return best_route, gbest_fitness, convergence_log

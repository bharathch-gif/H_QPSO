"""
benchmark.py

Runs each algorithm multiple times (since they're stochastic — a single run
proves nothing) and reports mean/std fitness plus convergence curves, so you
can fairly compare H-QPSO against classical PSO and a Genetic Algorithm.
"""

import time
import numpy as np

from graph_builder import build_city_graph, apply_simulated_congestion, pick_random_nodes
from vrp_problem import VRPProblem
from qpso import run_qpso
from baselines import run_pso, run_ga, run_exact, run_ortools


def run_hqpso(problem, **kwargs):
    """H-QPSO = QPSO + 2-opt local search hybridization (the default)."""
    kwargs.setdefault("apply_local_search", True)
    return run_qpso(problem, **kwargs)


def run_plain_qpso(problem, **kwargs):
    """Plain QPSO with no 2-opt — kept only as an ablation baseline, to show
    what the hybridization itself contributes."""
    kwargs["apply_local_search"] = False
    return run_qpso(problem, **kwargs)


ALGORITHMS = {
    "H-QPSO": run_hqpso,
    "PSO": run_pso,
    "GA": run_ga,
}

ABLATION_ALGORITHMS = {
    "H-QPSO (with 2-opt)": run_hqpso,
    "QPSO (no 2-opt)": run_plain_qpso,
}


def run_benchmark(problem, algorithm_fn, n_runs=10, **kwargs):
    """
    Runs a single algorithm n_runs times and returns raw results:
    a list of {fitness, convergence, runtime_sec} dicts.
    """
    results = []
    for run_idx in range(n_runs):
        start = time.time()
        _, fitness, conv_log = algorithm_fn(problem, seed=run_idx, **kwargs)
        elapsed = time.time() - start
        results.append({"fitness": fitness, "convergence": conv_log, "runtime_sec": elapsed})
    return results


def summarize(results):
    """Turns a list of run results into mean/std summary stats."""
    fitnesses = [r["fitness"] for r in results]
    runtimes = [r["runtime_sec"] for r in results]
    return {
        "mean_fitness": float(np.mean(fitnesses)),
        "std_fitness": float(np.std(fitnesses)),
        "best_fitness": float(np.min(fitnesses)),
        "mean_runtime_sec": float(np.mean(runtimes)),
    }


def compare_algorithms(problem, n_runs=10, n_iterations=100, n_particles=30):
    """
    Runs QPSO, PSO, and GA on the same problem and returns a comparison
    table + all raw results (raw results are useful for plotting
    convergence curves afterward).
    """
    all_results = {}
    summary_table = {}

    for name, fn in ALGORITHMS.items():
        kwargs = {"n_iterations": n_iterations}
        kwargs["n_particles" if name != "GA" else "population_size"] = n_particles

        results = run_benchmark(problem, fn, n_runs=n_runs, **kwargs)
        all_results[name] = results
        summary_table[name] = summarize(results)

    return summary_table, all_results


def validate_against_exact(problem, small_n=7, n_iterations=80, n_particles=25, seed=0):
    """
    Satisfies the problem statement's requirement to benchmark against
    'exact methods', not just other metaheuristics. Since brute force is
    only feasible for small instances, this builds a smaller VRPProblem
    using the first `small_n` customers from the given problem, finds the
    TRUE optimal route via run_exact, and reports how close QPSO/PSO/GA get
    to it in percentage terms.
    """
    small_customers = problem.customers[:small_n]
    small_problem = VRPProblem(
        problem.graph, problem.depot, small_customers,
        demands={c: problem.demands.get(c, 1) for c in small_customers},
        vehicle_capacity=problem.capacity,
        weight_attr=problem.weight_attr,
    )

    _, exact_fitness, _ = run_exact(small_problem, max_customers=small_n)

    comparison = {"Exact (optimal)": exact_fitness}
    for name, fn in [("H-QPSO", run_hqpso), ("PSO", run_pso), ("GA", run_ga)]:
        kwargs = {"n_iterations": n_iterations, "seed": seed}
        kwargs["n_particles" if name != "GA" else "population_size"] = n_particles
        _, fitness, _ = fn(small_problem, **kwargs)
        gap_pct = 100 * (fitness - exact_fitness) / exact_fitness if exact_fitness > 0 else 0
        comparison[name] = {"fitness": fitness, "gap_from_optimal_pct": gap_pct}

    return comparison


def validate_against_ortools(problem, time_limit_seconds=5, n_iterations=100, n_particles=30, seed=0):
    """
    Unlike validate_against_exact (limited to ~8 customers via brute force),
    OR-Tools can solve the FULL problem instance directly, so this compares
    H-QPSO/PSO/GA against a strong, production-grade reference at realistic
    problem sizes, not just a tiny subset.
    """
    ortools_route, ortools_fitness, _ = run_ortools(problem, time_limit_seconds=time_limit_seconds)

    comparison = {"OR-Tools (reference)": ortools_fitness}
    for name, fn in [("H-QPSO", run_hqpso), ("PSO", run_pso), ("GA", run_ga)]:
        kwargs = {"n_iterations": n_iterations, "seed": seed}
        kwargs["n_particles" if name != "GA" else "population_size"] = n_particles
        _, fitness, _ = fn(problem, **kwargs)
        gap_pct = 100 * (fitness - ortools_fitness) / ortools_fitness if ortools_fitness > 0 else 0
        comparison[name] = {"fitness": fitness, "gap_from_ortools_pct": gap_pct}

    return comparison


def run_ablation(problem, n_runs=10, n_iterations=100, n_particles=30):
    """
    Isolates the contribution of the 2-opt hybridization by comparing
    H-QPSO (with 2-opt) against plain QPSO (without) on identical settings.
    This is the evidence for the specific claim "hybridizing with 2-opt
    improves on plain QPSO" — a comparison against PSO/GA alone can't show
    that the hybrid part itself is what's helping.
    """
    summary = {}
    for name, fn in ABLATION_ALGORITHMS.items():
        results = run_benchmark(problem, fn, n_runs=n_runs,
                                 n_iterations=n_iterations, n_particles=n_particles)
        summary[name] = summarize(results)
    return summary


def run_scalability_test(G, depot_pool_seed=1, customer_counts=(10, 20, 30, 50),
                          n_iterations=60, n_particles=25, ortools_time_limit=5):
    """
    Tests how H-QPSO's solution quality and runtime behave as the VRP
    problem size (number of customers) grows — this is the actual evidence
    for a scalability claim, as opposed to just citing the size of the
    underlying road graph (which is a separate, already-large number from
    downloading real OSM data — see build_city_graph's node/edge count).

    Returns a list of dicts, one per customer count tested.
    """
    depot = pick_random_nodes(G, 1, seed=depot_pool_seed)[0]
    results = []

    for n_customers in customer_counts:
        customers = pick_random_nodes(G, n_customers, seed=depot_pool_seed + 1, exclude={depot})
        problem = VRPProblem(G, depot, customers)

        start = time.time()
        _, hqpso_fitness, _ = run_hqpso(
            problem, n_particles=n_particles, n_iterations=n_iterations, seed=0
        )
        hqpso_runtime = time.time() - start

        try:
            _, ortools_fitness, _ = run_ortools(problem, time_limit_seconds=ortools_time_limit)
            gap_pct = (
                100 * (hqpso_fitness - ortools_fitness) / ortools_fitness
                if ortools_fitness > 0 else float("nan")
            )
        except Exception as e:
            ortools_fitness, gap_pct = float("nan"), float("nan")

        results.append({
            "n_customers": n_customers,
            "hqpso_fitness": hqpso_fitness,
            "hqpso_runtime_sec": hqpso_runtime,
            "ortools_fitness": ortools_fitness,
            "gap_from_ortools_pct": gap_pct,
        })

    return results


if __name__ == "__main__":
    print("Building graph...")
    G = build_city_graph("Vijayawada, India")
    G = apply_simulated_congestion(G, seed=42)
    print(f"Road network graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(this is the underlying network size — separate from VRP customer count below)")

    depot = pick_random_nodes(G, 1, seed=1)[0]
    customers = pick_random_nodes(G, 10, seed=2, exclude={depot})

    problem = VRPProblem(G, depot, customers)

    print(f"Running benchmark on {len(customers)} customers...")
    summary, all_results = compare_algorithms(problem, n_runs=10, n_iterations=80, n_particles=25)

    print("\n--- Benchmark Results (10 runs each) ---")
    print(f"{'Algorithm':<10} {'Mean Fitness':>14} {'Std Dev':>10} {'Best':>10} {'Avg Time(s)':>12}")
    for name, stats in summary.items():
        print(f"{name:<10} {stats['mean_fitness']:>14.2f} {stats['std_fitness']:>10.2f} "
              f"{stats['best_fitness']:>10.2f} {stats['mean_runtime_sec']:>12.3f}")

    # --- Save results for slides: CSV table + convergence plot PNG ---
    import csv
    from utils import plot_convergence

    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Mean Fitness", "Std Dev", "Best Fitness", "Avg Runtime (s)"])
        for name, stats in summary.items():
            writer.writerow([name, stats["mean_fitness"], stats["std_fitness"],
                              stats["best_fitness"], stats["mean_runtime_sec"]])
    print("\nSaved benchmark_results.csv")

    fig = plot_convergence(all_results, title="H-QPSO vs PSO vs GA — Convergence")
    fig.savefig("convergence_plot.png", dpi=200, bbox_inches="tight")
    print("Saved convergence_plot.png")

    # --- Validate against the true optimum on a small subset (brute force) ---
    print("\n--- Validation vs. Exact Method (small subset, brute force) ---")
    exact_comparison = validate_against_exact(problem, small_n=min(7, len(customers)))
    print(f"True optimal cost: {exact_comparison['Exact (optimal)']:.2f}")
    for name in ("H-QPSO", "PSO", "GA"):
        stats = exact_comparison[name]
        print(f"{name:<8} cost: {stats['fitness']:>10.2f}   gap from optimal: {stats['gap_from_optimal_pct']:.2f}%")

    # --- Validate against OR-Tools on the FULL problem size ---
    print("\n--- Validation vs. OR-Tools (full instance, production-grade reference) ---")
    ortools_comparison = validate_against_ortools(problem)
    print(f"OR-Tools reference cost: {ortools_comparison['OR-Tools (reference)']:.2f}")
    for name in ("H-QPSO", "PSO", "GA"):
        stats = ortools_comparison[name]
        print(f"{name:<8} cost: {stats['fitness']:>10.2f}   gap from OR-Tools: {stats['gap_from_ortools_pct']:.2f}%")

    # --- Ablation: does the 2-opt hybridization itself actually help? ---
    print("\n--- Ablation: H-QPSO (with 2-opt) vs plain QPSO (no 2-opt) ---")
    ablation = run_ablation(problem, n_runs=10, n_iterations=80, n_particles=25)
    for name, stats in ablation.items():
        print(f"{name:<22} mean={stats['mean_fitness']:.2f}  std={stats['std_fitness']:.2f}")

    # --- Report distance / time / congestion as separate figures ---
    print("\n--- Route Metrics Breakdown (H-QPSO's best route on full instance) ---")
    best_route, best_fitness, _ = run_hqpso(problem, n_particles=25, n_iterations=80, seed=0)
    metrics = problem.route_metrics(best_route)
    print(f"Total distance:          {metrics['total_distance_m']:.1f} m")
    print(f"Total free-flow time:    {metrics['total_free_flow_time_s']:.1f} s")
    print(f"Total congested time:    {metrics['total_congested_time_s']:.1f} s")
    print(f"Congestion delay added:  {metrics['congestion_delay_s']:.1f} s")

    # --- Scalability: does solution quality/runtime hold up as problem size grows? ---
    print("\n--- Scalability Test (H-QPSO across increasing customer counts) ---")
    scalability_results = run_scalability_test(
        G, customer_counts=(10, 20, 30, 50), n_iterations=60, n_particles=25
    )
    print(f"{'#Customers':>10} {'H-QPSO Cost':>12} {'Runtime(s)':>11} {'OR-Tools':>10} {'Gap %':>8}")
    for r in scalability_results:
        print(f"{r['n_customers']:>10} {r['hqpso_fitness']:>12.1f} {r['hqpso_runtime_sec']:>11.2f} "
              f"{r['ortools_fitness']:>10.1f} {r['gap_from_ortools_pct']:>7.2f}%")

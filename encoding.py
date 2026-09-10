"""
encoding.py

Bridges the gap between swarm algorithms (which only understand continuous
numbers) and the VRP (which needs a discrete ordering of customers).

Technique: random-key encoding. Each particle's "position" is a vector of
continuous priority values, one per customer. Sorting customers by their
priority value produces a valid visiting order (a permutation) — so every
possible continuous vector always decodes to a valid route, with no extra
repair step needed.
"""

import numpy as np


def decode_position(position_vector, customer_nodes):
    """
    position_vector: array of floats, e.g. [0.7, 0.2, 0.9, 0.1]
    customer_nodes:  list of node IDs, e.g. [A, B, C, D]

    Returns the customer nodes sorted by ascending priority value, i.e.
    the decoded route order. Example: [0.7, 0.2, 0.9, 0.1] for [A, B, C, D]
    decodes to [D, B, A, C].
    """
    paired = list(zip(position_vector, customer_nodes))
    paired.sort(key=lambda pair: pair[0])
    return [node for _, node in paired]


def random_position(dim):
    """A random starting position vector — one priority value per customer."""
    return np.random.rand(dim)


def encode_route(route_order, customer_nodes):
    """
    Inverse of decode_position: given a discrete route (a permutation of
    customer_nodes), builds a continuous position vector that decode_position
    will reproduce exactly.

    Needed for hybridizing QPSO with local search (e.g. 2-opt): local search
    operates on discrete routes, but the swarm's gbest is stored as a
    continuous position, so an improved route found by local search needs to
    be re-encoded before it can become the swarm's new gbest.

    route_order:    e.g. [D, B, A, C] — the desired decoded order
    customer_nodes: e.g. [A, B, C, D] — same node set, in the swarm's
                    original indexing order (problem.customers)
    """
    n = len(customer_nodes)
    rank_in_route = {node: i for i, node in enumerate(route_order)}
    denom = max(n - 1, 1)
    return np.array([rank_in_route[node] / denom for node in customer_nodes])

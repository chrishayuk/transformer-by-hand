"""The other half — assemble the many-body force.

Each routing head (routing.py) brings one neighbour's (x, y, q) into a slot; the
per-pair FFN (frozen_ffn.py) computes that pair's Coulomb force from the slot; we
sum over slots. Attention routes, the FFN computes — the whole two-video claim, as
one function.

`mean_aggregate_force` is the foil: a single head averaging all neighbours, which
cannot recover the sum because the per-pair force is nonlinear. That's the `mean`
demo's cliff.
"""

from __future__ import annotations

import numpy as np

from frozen_ffn import l1_forward, l2_forward, l3_forward
from routing import (
    SQ, SX, SY, TARGET_HEAD1, TARGET_HEAD2, RoutingHead, build_residual,
    neighbor_targets, softmax,
)


def _per_pair_force(dx, dy, q0, q1, ffn):
    """The constructed per-pair FFN: (dx, dy, q0, q1) → (fx, fy)."""
    dist_sq = l1_forward(dx, dy, ffn.l1)
    inv_r3 = l2_forward(dist_sq, ffn.l2)
    out = l3_forward(dx, dy, q0, q1, inv_r3, ffn.l3)
    return out["fx"], out["fy"]


def constructed_block_force(positions, charges, beta, ffn, use_true_operands=False):
    """Route each neighbour, run the FFN per slot, sum → force (N,2).

    use_true_operands=False (honest): the FFN reads the post-softmax routed
    operands. =True (control): feed exact neighbour state, to isolate the routing
    seam from the FFN itself. Returns (forces, {n1, n2}) for inspection."""
    X = build_residual(positions, charges)
    h1, h2 = RoutingHead(TARGET_HEAD1, beta), RoutingHead(TARGET_HEAD2, beta)
    if use_true_operands:
        n1, n2 = X[:, 0:3][TARGET_HEAD1], X[:, 0:3][TARGET_HEAD2]
    else:
        n1, n2 = h1.route(X), h2.route(X)

    F = np.zeros((len(positions), 2))
    for nb in (n1, n2):
        dx = X[:, SX] - nb[:, 0]
        dy = X[:, SY] - nb[:, 1]
        fx, fy = _per_pair_force(dx, dy, X[:, SQ], nb[:, 2], ffn)
        F[:, 0] += fx
        F[:, 1] += fy
    return F, {"n1": n1, "n2": n2}


def _route_general(positions, charges, target, beta, true_operand=False):
    """Softmax-route neighbour target[i] for each token i (self-masked), at any N."""
    state = np.column_stack([positions[:, 0], positions[:, 1], charges])
    if true_operand:
        return state[target]
    N = len(positions)
    eyeN = np.eye(N)
    scores = eyeN[target] @ eyeN.T
    attn = softmax(beta * scores + np.where(eyeN.astype(bool), -1e9, 0.0), axis=1)
    return attn @ state


def block_force_general(positions, charges, beta, ffn, true_operands=False):
    """The same block at arbitrary N: N−1 width-one routing heads + FFN sum.
    Equals `constructed_block_force` at N=3."""
    F = np.zeros((len(positions), 2))
    for tk in neighbor_targets(len(positions)):
        nb = _route_general(positions, charges, tk, beta, true_operands)
        dx = positions[:, 0] - nb[:, 0]
        dy = positions[:, 1] - nb[:, 1]
        fx, fy = _per_pair_force(dx, dy, charges, nb[:, 2], ffn)
        F[:, 0] += fx
        F[:, 1] += fy
    return F


def mean_aggregate_force(positions, charges, ffn):
    """The obstacle, concrete. One head attending uniformly to all neighbours gives
    their MEAN, and f(i, mean) ≠ mean f(i,j) for nonlinear f — so averaging (even
    rescaled by the neighbour count) cannot recover the sum. This is what `mean`
    collapses on."""
    X = build_residual(positions, charges)
    N = len(positions)
    attn = softmax(np.where(np.eye(N, dtype=bool), -1e9, 0.0), axis=1)  # uniform over neighbours
    mean_nb = attn @ X[:, 0:3]
    dx = X[:, SX] - mean_nb[:, 0]
    dy = X[:, SY] - mean_nb[:, 1]
    fx, fy = _per_pair_force(dx, dy, X[:, SQ], mean_nb[:, 2], ffn)
    return np.stack([fx * (N - 1), fy * (N - 1)], axis=1)

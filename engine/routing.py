"""The attention half — routing, the way the first video builds it by hand.

A head decides *who reads whom* (one target per token), turns that into a one-hot
weight matrix via QK on positional addresses, masks the diagonal, sharpens with a
softmax, and routes each neighbour's raw (x, y, charge) into a slot — unchanged.

This is `by_hand/build_attention_by_hand.py` as a reusable module: the same five
steps — state → P → target → scores → softmax → A @ state — just wrapped so the
block and the CLI can call it at any N. `RoutingHead.attention()` below is exactly
the matrix you watch get built by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# residual layout (d_model = 12, N = 3):
#   [ x y q | positional one-hot (3) | neighbour slot 1 (3) | neighbour slot 2 (3) ]
D_MODEL = 12
SX, SY, SQ = 0, 1, 2          # self state: x, y, charge
POS = slice(3, 6)             # positional one-hot (token id, for QK routing)
SLOT1 = slice(6, 9)           # routed neighbour 1
SLOT2 = slice(9, 12)          # routed neighbour 2

# who each token reads (N=3, self-masked): head 1 the first neighbour, head 2 the second
TARGET_HEAD1 = np.array([1, 0, 0])
TARGET_HEAD2 = np.array([2, 2, 1])


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def build_residual(positions, charges):
    """(N,2) positions + (N,) charges → residual stream X (N, 12):
    self (x,y,q) | positional one-hot | two empty neighbour slots."""
    N = len(positions)
    X = np.zeros((N, D_MODEL))
    X[:, SX] = positions[:, 0]
    X[:, SY] = positions[:, 1]
    X[:, SQ] = charges
    X[:, POS] = np.eye(N)              # positional address per token
    return X


@dataclass(frozen=True)
class RoutingHead:
    """One frozen attention head. `target[i]` is the token that token i reads.

    `attention()` is the hand-built matrix, line for line:
        Q = onehot(target);  scores = Q @ P.T;  A = softmax(beta*scores + no_self)
    `route()` then does A @ state — carrying the neighbour's (x, y, q) across,
    unchanged (a blend for finite beta — that's the routing 'sharpness')."""

    target: np.ndarray   # (N,) neighbour index per token
    beta: float          # softmax sharpness

    def attention(self, X):
        N = len(X)
        Q = np.eye(N)[self.target]                       # row i = the address it reads
        scores = Q @ X[:, POS].T                          # a 1 where j == target(i)
        no_self = np.where(np.eye(N, dtype=bool), -1e9, 0.0)
        return softmax(self.beta * scores + no_self, axis=1)

    def route(self, X):
        """The routed neighbour state (N,3) = (x, y, q)."""
        return self.attention(X) @ X[:, 0:3]


def neighbor_targets(N):
    """N−1 target arrays; head k routes the k-th neighbour (all j ≠ i) of each token."""
    return [np.array([[j for j in range(N) if j != i][k] for i in range(N)])
            for k in range(N - 1)]


def beta_for_target_weight(N, w):
    """beta giving softmax weight `w` on the routed neighbour over N−1 self-masked
    positions: w = e^beta / (e^beta + (N−2)). Holding w fixed across N keeps every
    head equally sharp — the control that separates real error growth from the
    softmax thinning out as more tokens are added."""
    return float(np.log(max(N - 2, 1) * w / (1.0 - w)))

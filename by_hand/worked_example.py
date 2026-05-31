#!/usr/bin/env python3
"""Proof that the bare grid-of-1s in build_attention_by_hand.py IS real attention.

The on-camera build draws A by hand as a grid of 1s. The real thing computes A
with a softmax over scores and learns where to point. Here we build BOTH on the
same targets and crank the softmax confidence up — and the routed states agree
well past eight decimal places. So the simple version isn't a cartoon of
attention; it's the sharp limit of it.

numpy only.   Run:  python3 by_hand/worked_example.py
"""
import numpy as np
from numpy import array, eye
np.set_printoptions(precision=3, suppress=True)

state = array([[6.13, 8.57,  1.],
               [7.48, 2.53, -1.],
               [3.20, 8.36,  1.]])
target = [1, 0, 0]                       # token 0->1, tokens 1,2->0

# --- the by-hand version: a grid of 1s, drawn directly ---
A_hand = eye(3)[target]                  # the same grid you type as a literal matrix
routed_hand = A_hand @ state

# --- the real version: softmax over scores, self-masked, confidence turned up ---
def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

beta = 30.0                              # crank the confidence up -> the sharp limit
scores = eye(3)[target] @ eye(3).T       # 1 where each token points
no_self = np.where(eye(3, dtype=bool), -1e9, 0.0)
A_soft = softmax(beta * scores + no_self)
routed_soft = A_soft @ state

print("A_hand (drawn by hand):")
print(A_hand)
print(f"\nA_soft (real softmax form, beta={beta:.0f}):")
print(A_soft)

diff = np.max(np.abs(routed_hand - routed_soft))
print(f"\nmax difference in routed state:  {diff:.2e}")
assert diff < 5e-9, "the two forms disagree past eight decimal places!"
print("=> identical past eight decimal places. the hand-drawn grid is exactly")
print("   what the softmax collapses to once its confidence is high.")

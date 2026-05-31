#!/usr/bin/env python3
"""Build a transformer's attention head BY HAND — no training, nothing hidden.
Write it, run it, watch a value get routed through completely untouched.

Faithful to the engine's routing.RoutingHead (one-hot positional QK, self-mask,
softmax at beta, A @ state). numpy only. Run it:

    python3 by_hand/build_attention_by_hand.py
"""
import numpy as np
from numpy import array, eye
np.set_printoptions(precision=3, suppress=True)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


no_self = np.where(eye(3, dtype=bool), -1e9, 0.0)   # a token may not read itself

# --- the three particles. this is ALL attention gets to move. ---
state = array([[6.13, 8.57,  1.],
               [7.48, 2.53, -1.],
               [3.20, 8.36,  1.]])
print("state  [ x     y    charge ]  — one row per particle:")
print(state)

# --- each particle's address (who's who) ---
P = eye(3)
print("\nP  — token i lives at row i (just the identity):")
print(P)

# --- the ONLY decision in the whole head: who reads whom. I type this. ---
target = [1, 0, 0]          # token 0 -> token 1 ;  tokens 1,2 -> token 0
print(f"\ntarget = {target}   <- chosen by hand. no optimiser. no data.")

# --- turn that decision into the weight matrix ---
Q = eye(3)[target]
scores = Q @ P.T            # a 1 exactly where I pointed
print("\nscores = Q @ P.T   — the weight: a 1 where I pointed, 0 elsewhere:")
print(scores)

# --- forbid self-reading, then sharpen into one clean pick ---
A = softmax(20 * scores + no_self)
print("\nA = softmax(20*scores + no_self)   — the attention matrix:")
print(A)

# --- multiply: the value moves. watch it stay UNCHANGED. ---
routed = A @ state
print("\nrouted = A @ state   — each token now holds its neighbour's row:")
print(routed)
print(f"  token 0 carries q = {routed[0,2]:+.0f}   (token 1's real q = {state[1,2]:+.0f})  -> identical")

# --- poke it: change ONE target, the wiring follows. no magic. ---
target = [2, 0, 0]          # token 0: read token 2 instead of token 1
routed2 = softmax(20 * (eye(3)[target] @ P.T) + no_self) @ state
print(f"\npoke: token 0 target 1 -> 2.  token 0 now carries q = {routed2[0,2]:+.0f}   (token 2's real q = {state[2,2]:+.0f})")

# --- routing CARRIED a value. computing MAKES a new one. ---
a, b = 1.0, -1.0
prod = ((a + b)**2 - (a - b)**2) / 4        # multiply, built from squares
print(f"\nroute vs compute:")
print(f"  routing carried a charge untouched:        -1 -> -1")
print(f"  computing q0*q1 = ((a+b)^2-(a-b)^2)/4 = {prod:+.0f}   <- a NEW number, in neither input")
"""Attention, by hand — no training, nothing hidden — then the calculator, imported.

Part 3 (top): attention is a grid of 1s, times the data — pure numpy.
Part 5 (bottom): import the hand-built calculator (engine/) and run the whole layer.

Run:   python3 by_hand/build_attention_by_hand.py
(worked_example.py proves the bare grid equals the real softmax form.)
"""
import numpy as np
from numpy import array
np.set_printoptions(precision=3, suppress=True)

# the three particles: position + charge, one row each. ALL attention can touch.
state = array([[6.13, 8.57,  1.],
               [7.48, 2.53, -1.],
               [3.20, 8.36,  1.]])
print("state  [ x     y    charge ]:")
print(state)

# attention is a grid of 1s. each row has ONE 1 — the token that row reads.
# I draw these arrows by hand. no optimiser, no data.
A = array([[0., 1., 0.],      # token 0 needs token 1
           [1., 0., 0.],      # token 1 needs token 0
           [1., 0., 0.]])     # token 2 needs token 0
print("\nA — the attention grid (a 1 where each token points):")
print(A)

# route: each token copies in the row it pointed at. that's all '@' does here.
routed = A @ state
print("\nrouted = A @ state — each token now holds its neighbour's row:")
print(routed)
print(f"  token 0 carries q = {routed[0,2]:+.0f}   (token 1's real q = {state[1,2]:+.0f})  -> identical")

# poke it: move ONE 1, and the wiring follows. no magic to break.
A[0] = [0., 0., 1.]           # token 0 now needs token 2 instead
print(f"\npoke: token 0's 1 moved to column 2.  token 0 carries q = {(A @ state)[0,2]:+.0f}   (token 2's real q = {state[2,2]:+.0f})")

# routing CARRIED a value untouched. computing MAKES a new one.
a, b = 1.0, -1.0
prod = ((a + b)**2 - (a - b)**2) / 4
print("\nroute vs compute:")
print("  routing carried a charge untouched:        -1 -> -1")
print(f"  computing q0*q1 = ((a+b)^2-(a-b)^2)/4 = {prod:+.0f}   <- a NEW number, in neither input")


# ===========================================================================
#  PART 5 — the COMPUTING half (typed on camera AFTER the attention part runs)
# ===========================================================================
#  Attention only ever COPIED a value across, untouched. The other half of the
#  layer MAKES new numbers — it computes the force. That calculator is built by
#  hand too, but out of ReLU "hinges", and *that* construction is its own whole
#  video. So here I don't build it — two lines so Python can find the engine/
#  folder next door, then I import the finished calculator and use it.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # so `import engine` finds the folder next door
from engine.frozen_ffn import ConstructFFN              # the hand-built calculator
from engine.block import constructed_block_force        # route each neighbour -> FFN -> sum
from engine.multi_force import MultiForceSimulator, MultiForceParams  # physics ground truth


def r2(pred, true):
    p, t = pred.reshape(-1), true.reshape(-1)
    return 1.0 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)


pos, q = state[:, :2], state[:, 2]
ffn = ConstructFFN.build()                               # the calculator (its hinges = next video)

# the whole layer, end to end: the SAME routing you watched me type, feeding the
# calculator, summed over neighbours. nothing trained.
pred, _ = constructed_block_force(pos, q, 20.0, ffn)
true    = MultiForceSimulator(MultiForceParams(n_particles=3)).compute_coulomb(pos, q)

print("\nthe whole layer — force on each token   (hand-built  vs  simulator):")
for i in range(3):
    print(f"  token {i}:  ({pred[i,0]:+.3f}, {pred[i,1]:+.3f})   vs   ({true[i,0]:+.3f}, {true[i,1]:+.3f})")
print(f"\n  R^2 = {r2(pred, true):.4f}    (untrained, hand-built)")

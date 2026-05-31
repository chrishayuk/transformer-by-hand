import numpy as np
from numpy import array
np.set_printoptions(precision=3, suppress=True)

state = array([[6.13, 8.57,  1.],
               [7.48, 2.53, -1.],
               [3.20, 8.36,  1.]])
print("state  [ x     y    charge ]:")
print(state)

A = array([[0., 1., 0.],
           [1., 0., 0.],
           [1., 0., 0.]])
print("\nA — the attention grid (a 1 where each token points):")
print(A)

routed = A @ state
print("\nrouted = A @ state — each token now holds its neighbour's row:")
print(routed)
print(f"  token 0 carries q = {routed[0, 2]:+.0f}   (token 1's real q = {state[1, 2]:+.0f})  -> identical")

A[0] = [0., 0., 1.]
routed2 = A @ state
print(f"\npoke: token 0's 1 moved to column 2.  token 0 carries q = {routed2[0, 2]:+.0f}   (token 2's real q = {state[2, 2]:+.0f})")

a, b = 1.0, -1.0
prod = ((a + b)**2 - (a - b)**2) / 4
print("\nroute vs compute:")
print( "  routing carried a charge untouched:        -1 -> -1")
print(f"  computing q0*q1 = ((a+b)^2-(a-b)^2)/4 = {prod:+.0f}   <- a NEW number, in neither input")

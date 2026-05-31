#!/usr/bin/env python3
"""Interactive CLI for the constructed transformer block.

Build a scenario yourself — set charges, positions, particle count, routing
sharpness — then run the hand-built model and compare its prediction to the
simulator's reality. All live in the terminal, real numbers, nothing saved.

    python3 cli/attention_cli.py
    (then type `help`)

Pipe a script of commands too:
    printf 'charge 0 1\ncharge 1 1\nrun\nattn\nquit\n' | python3 cli/attention_cli.py
"""
# vendored engine modules live one level up, in engine/
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "engine"))

import numpy as np

from routing import (
    TARGET_HEAD1, TARGET_HEAD2, beta_for_target_weight, neighbor_targets, softmax,
)
from block import block_force_general, constructed_block_force, mean_aggregate_force
from frozen_ffn import (
    SOFT2, ConstructFFN, build_square, l1_forward, l2_forward, l3_forward,
    per_term_verify, relative_error_report,
)
from multi_force import MultiForceParams, MultiForceSimulator

np.set_printoptions(precision=3, suppress=True)

HELP = """  commands:
    show              the current particle config
    run               run the model -> prediction vs reality + R²
    attn              attention working: routing matrices + carried charge
    ffn               BUILD the per-pair FFN layer by layer, verified vs truth
    build             the construction (hand-set weights, residual layout)
    mean              the broken 'average the neighbours' variant (over a batch)
    rand              fresh random config (keeps current N)
    n <k>             set number of particles (rebuilds a random config)
    charge <i> <v>    set particle i's charge (e.g. `charge 0 1`)
    pos <i> <x> <y>   set particle i's position
    beta <b>          set routing sharpness (try `beta 2` to watch it degrade)
    help              this list
    quit"""


def r2(pred, true):
    p, t = pred.reshape(-1), true.reshape(-1)
    return 1.0 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)


class Model:
    def __init__(self):
        self.ffn = ConstructFFN.build(l2_n_knots=64, l2_knot_spacing="log")
        self.rng = np.random.default_rng(7)
        self.N, self.beta = 3, 20.0
        self.rand()

    def _sim(self):
        return MultiForceSimulator(MultiForceParams(n_particles=self.N))

    def rand(self):
        sim = self._sim()
        while True:
            pos, _ = sim.sample_state(self.rng)
            chg = sim.sample_particle_properties(self.rng)["charges"]
            if np.abs(chg).min() > 0:
                break
        self.pos, self.chg = pos, np.round(chg).astype(float)
        self.beta = 20.0 if self.N == 3 else beta_for_target_weight(self.N, 0.9997)

    def predict(self):
        if self.N == 3:
            return constructed_block_force(self.pos, self.chg, self.beta, self.ffn)[0]
        return block_force_general(self.pos, self.chg, self.beta, self.ffn)

    def truth(self):
        return self._sim().compute_coulomb(self.pos, self.chg)


def cmd_show(m):
    print(f"\n  config: {m.N} particles · routing β={m.beta:.1f}")
    for i in range(m.N):
        print(f"     token {i}:  pos=({m.pos[i,0]:+.2f}, {m.pos[i,1]:+.2f})   q={m.chg[i]:+.0f}")


def cmd_run(m):
    pred, true = m.predict(), m.truth()
    print(f"\n  PREDICTION vs REALITY   (hand-built block  vs  simulator)   β={m.beta:.1f}")
    print("     particle   predicted (Fx,Fy)       true (Fx,Fy)          |err|")
    for i in range(m.N):
        e = np.linalg.norm(pred[i] - true[i])
        print(f"     {i:<2}         ({pred[i,0]:+.3f}, {pred[i,1]:+.3f})      "
              f"({true[i,0]:+.3f}, {true[i,1]:+.3f})      {e:.4f}")
    print(f"\n     R² = {r2(pred, true):.4f}    (untrained, hand-built)")


def cmd_attn(m):
    state = np.column_stack([m.pos[:, 0], m.pos[:, 1], m.chg])
    eye = np.eye(m.N)
    print(f"\n  ATTENTION WORKING   ({m.N - 1} routing heads, β={m.beta:.1f})")
    for k, tk in enumerate(neighbor_targets(m.N)):
        A = softmax(m.beta * (eye[tk] @ eye.T) + np.where(eye.astype(bool), -1e9, 0.0), axis=1)
        routed = A @ state
        print(f"  head {k + 1}:")
        if m.N <= 4:
            for i in range(m.N):
                print(f"     token {i}: {A[i]} -> carries token {int(tk[i])}, q={routed[i,2]:+.0f} (unchanged)")
        else:
            for i in range(m.N):
                print(f"     token {i} -> token {int(tk[i])} (w={A[i].max():.2f}), q={routed[i,2]:+.0f}")


def cmd_ffn(m, n=2000):
    """Build the per-pair FFN layer by layer; verify each layer against the true
    quantity (the construction the whole video is about). Runs on n random pairs."""
    rng = np.random.default_rng(0)
    dx = rng.uniform(-9, 9, n); dy = rng.uniform(-9, 9, n)
    q0 = rng.choice([-1.0, 1.0], n); q1 = rng.choice([-1.0, 1.0], n)
    ds_true = dx ** 2 + dy ** 2 + SOFT2
    ir_true = ds_true ** -1.5
    fx_true, fy_true = q0 * q1 * dx * ir_true, q0 * q1 * dy * ir_true

    print(f"\n  BUILDING THE FFN — per-pair force, layer by layer (verified on {n} pairs)")
    print("  " + "-" * 56)
    ds = l1_forward(dx, dy, m.ffn.l1)
    v1 = per_term_verify("dist2", ds, ds_true)
    print(f"  L1  dist² = dx²+dy²+soft²    cosine={v1.cosine:.6f}  max|err|={v1.max_abs_error:.3f}")
    ir_iso = l2_forward(ds_true, m.ffn.l2)                 # fed the true dist²
    ir = l2_forward(ds, m.ffn.l2)                          # composed with the real (curvature-aware) L1
    sq_uni = build_square(9.0, 64, "uniform")              # the NAIVE knot placement
    ir_uni = l2_forward(sq_uni(dx) + sq_uni(dy) + SOFT2, m.ffn.l2)
    print("  L2  1/r³ = dist²^-1.5   (log-spaced knots — the hard layer)")
    print(f"        fed TRUE dist²               :  rel_p99={relative_error_report(ir_iso, ir_true)['rel_p99']:.3f}")
    print(f"        fed L1, curvature-aware knots:  rel_p99={relative_error_report(ir, ir_true)['rel_p99']:.3f}   <- composed chain holds")
    print(f"        fed L1, naive uniform knots  :  rel_p99={relative_error_report(ir_uni, ir_true)['rel_p99']:.3f}   <- 1/r³ amplifies the error")
    out = l3_forward(dx, dy, q0, q1, ir, m.ffn.l3)
    vqq = per_term_verify("qq", out["charge_product"], q0 * q1)
    fr2 = r2(np.stack([out["fx"], out["fy"]], 1), np.stack([fx_true, fy_true], 1))
    print(f"  L3  q0·q1 (ReLU multiply)    cosine={vqq.cosine:.6f}")
    print(f"  L3  force = q0q1·(dx,dy)/r³  R²={fr2:.4f}  vs the true Coulomb force")
    print("  → every layer hand-set, verified against the real quantity. No training.")


def cmd_build(m):
    print("\n  THE CONSTRUCTION  (hand-set, frozen, never trained)")
    print("     residual (d=12, N=3): [ x y q | pos(3) | slot1(3) | slot2(3) ]")
    print(f"     routing head 1 target : {TARGET_HEAD1.tolist()}")
    print(f"     routing head 2 target : {TARGET_HEAD2.tolist()}")
    print(f"     at N={m.N}: {m.N - 1} one-hot routing heads + the reused N=2 Coulomb FFN.")


def cmd_mean(m, n=200):
    rng = np.random.default_rng(123)
    sim = MultiForceSimulator(MultiForceParams(n_particles=3))
    pr = np.zeros((n, 3, 2)); pm = np.zeros((n, 3, 2)); tr = np.zeros((n, 3, 2))
    for k in range(n):
        while True:
            p, _ = sim.sample_state(rng); c = sim.sample_particle_properties(rng)["charges"]
            if np.abs(c).min() > 0:
                break
        pr[k], _ = constructed_block_force(p, c, 20.0, m.ffn)
        pm[k] = mean_aggregate_force(p, c, m.ffn)
        tr[k] = sim.compute_coulomb(p, c)
    print(f"\n  THE CLIFF — routing vs averaging  (N=3, over {n} configs)")
    print(f"     route the RIGHT neighbour:  R² = {r2(pr, tr):+.4f}")
    print(f"     average the neighbours:     R² = {r2(pm, tr):+.3f}   <- collapses (< 0)")


def main():
    print("\n  constructed transformer block — interactive CLI   (type `help`)")
    m = Model()
    cmd_show(m)
    while True:
        try:
            parts = input("\n  > ").strip().split()
        except EOFError:
            break
        if not parts:
            continue
        c, args = parts[0].lower(), parts[1:]
        try:
            if c in ("quit", "q", "exit"):
                break
            elif c == "help":
                print(HELP)
            elif c == "show":
                cmd_show(m)
            elif c == "run":
                cmd_run(m)
            elif c == "attn":
                cmd_attn(m)
            elif c == "ffn":
                cmd_ffn(m)
            elif c == "build":
                cmd_build(m)
            elif c == "mean":
                cmd_mean(m)
            elif c == "rand":
                m.rand(); cmd_show(m)
            elif c == "n":
                m.N = max(2, int(args[0])); m.rand(); cmd_show(m)
            elif c == "charge":
                m.chg[int(args[0])] = float(args[1]); cmd_show(m)
            elif c == "pos":
                m.pos[int(args[0])] = [float(args[1]), float(args[2])]; cmd_show(m)
            elif c == "beta":
                m.beta = float(args[0]); print(f"     β = {m.beta}")
            else:
                print("     ? type `help`")
        except (IndexError, ValueError):
            print(f"     bad args for `{c}` — see `help`")
    print("\n  done.\n")


if __name__ == "__main__":
    main()

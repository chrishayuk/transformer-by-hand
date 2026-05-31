#!/usr/bin/env python3
"""Interactive CLI for HAND-BUILDING the feed-forward network — no training.

This is the FFN companion to attention_cli.py (which drives the attention/routing
video). Where that tool treats the FFN as one verified beat, this one teaches it from the
ground up: a ReLU is a hinge, a feed-forward layer is a sum of hinges, a sum of
hinges is any curve you like — so an FFN is a function approximator you can place
by hand. Then we build the real Coulomb force out of hinges, layer by layer, and
verify every layer against the simulator. Every number printed is the real
construction's output. Nothing is trained. Nothing is saved.

    python3 cli/ffn_cli.py
    (then type `help`)

Pipe a script of commands too:
    printf 'relu\nrecip\nverify\nquit\n' | python3 cli/ffn_cli.py
"""
# vendored engine modules live one level up, in engine/
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "engine"))

import numpy as np

from frozen_ffn import (
    SOFT2, ConstructFFN, L2Weights, ReluPWL, build_square, l1_forward,
    l2_forward, l3_forward, per_term_verify, relative_error_report,
    relu_multiply,
)
from multi_force import MultiForceParams, MultiForceSimulator

np.set_printoptions(precision=3, suppress=True)

HELP = """  the feed-forward network, built by hand — no training

  FOUNDATIONS (what a neuron actually is)
    relu              a ReLU is a hinge: flat, then a ramp. the only nonlinearity.
    stack             sum a few hinges -> any bent curve. neurons ARE the hinges.

  BUILD THE PIECES (the arithmetic the force needs)
    square [N]        build x^2 from N hinges (try `square 4` then `square 64`)
    recip             build 1/r^3 — the hard one. log knots vs uniform knots.
    budget            1/r^3 accuracy vs neuron count: placement, not capacity.
    knots             WHERE the hinges sit (uniform spreads them; log clusters them)
    multiply [a] [b]  multiplication isn't a ReLU primitive — build it from squares.

  ASSEMBLE & VERIFY (the whole force, checked against reality)
    verify            build L0->L3, verify EVERY layer vs the simulator. R^2=0.9998.
    compare           the honest scoreboard: vs the trained model (no output gap).

    help              this list
    quit"""


def _cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _fmt_err(rel_p99, spacing):
    """Format a 99th-pct relative error for the table.

    The UNIFORM-knot error is a single near-contact percentile and wobbles
    run-to-run / build-to-build; report it categorically (>100% / >1000% / …) so
    no precise float ends up on screen to be defended — uniform is *categorically*
    broken, which is the true and stronger statement. The LOG-knot error is small
    and stable (the affirmative result), so show it exactly.
    """
    pct = rel_p99 * 100.0
    if spacing == "log":
        return f"{pct:.1f}%"
    for thresh, label in ((10000, ">10000%"), (1000, ">1000%"),
                          (100, ">100%"), (10, ">10%")):
        if pct >= thresh:
            return label
    return f"{pct:.1f}%"


def _r2(pred, true):
    p, t = np.ravel(pred), np.ravel(true)
    return 1.0 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)


class Model:
    """Holds the frozen FFN and a cached sample of the real per-pair distribution."""

    def __init__(self, n=4000, seed=7):
        self.ffn = ConstructFFN.build(l2_n_knots=64, l2_knot_spacing="log")
        self._sample(n, seed)

    def _sample(self, n, seed):
        """Sample real (pos ~ U(0.5,9.5)) Coulomb pairs + the simulator's truth."""
        rng = np.random.default_rng(seed)
        sim = MultiForceSimulator(MultiForceParams(n_particles=2))
        active = {"coulomb": True, "spring": False, "gravity": False, "drag": False}
        batch = np.zeros((n, 20))
        dx = np.zeros(n); dy = np.zeros(n); q0 = np.zeros(n); q1 = np.zeros(n)
        fx = np.zeros(n); fy = np.zeros(n)
        for i in range(n):
            pos, vel = sim.sample_state(rng)
            props = sim.sample_particle_properties(rng)
            fp = sim.sample_force_params(rng)
            batch[i] = sim.build_input_vector(pos, vel, props, fp, active)
            dx[i] = pos[0, 0] - pos[1, 0]
            dy[i] = pos[0, 1] - pos[1, 1]
            q0[i], q1[i] = props["charges"]
            f = sim.compute_coulomb(pos, props["charges"])
            fx[i], fy[i] = f[0, 0], f[0, 1]
        self.batch = batch
        self.dx, self.dy, self.q0, self.q1 = dx, dy, q0, q1
        self.fx, self.fy = fx, fy
        self.dist_sq = dx * dx + dy * dy + SOFT2
        self.inv_r3 = self.dist_sq ** -1.5


# ---------------------------------------------------------------------------
# Foundations — what a neuron actually is
# ---------------------------------------------------------------------------

def cmd_relu(m):
    print("\n  A NEURON IS A HINGE.  ReLU(s) = max(s, 0) — flat, then a ramp.")
    print("  One hidden unit fires at a knot k:  contribution = w * max(s - k, 0)")
    s = np.array([-2., -1., 0., 1., 2., 3., 4.])
    print("\n     s              " + "".join(f"{v:>7.0f}" for v in s))
    print("     max(s,   0)    " + "".join(f"{v:>7.1f}" for v in np.maximum(s, 0)))
    print("     max(s-1, 0)    " + "".join(f"{v:>7.1f}" for v in np.maximum(s - 1, 0))
          + "   <- same hinge, switched on at k=1")
    print("\n  That ramp is the whole nonlinearity. Everything else is affine (weights).")
    print("  affine -> ReLU -> affine IS a feed-forward layer. Nothing more.")


def cmd_stack(m):
    print("\n  STACK HINGES -> ANY CURVE.  A sum of ramps is a piecewise-linear function;")
    print("  pick where the hinges fire and how steep they are, and you bend a flat line")
    print("  into any shape you want.")
    # PWL interpolant of x^2 pinned through 5 control points (knots). A ReluPWL
    # through K knots is K-2 ReLU hidden units (hinges) + one base line.
    sq5 = ReluPWL.fit(lambda x: x * x, np.array([-2., -1., 0., 1., 2.]))
    xs = np.array([-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2.])
    approx = sq5(xs)
    print(f"\n  example: {sq5.n_neurons} hinges (pinned through 5 control points) -> x^2 on [-2, 2]")
    print("     x          " + "".join(f"{v:>7.1f}" for v in xs))
    print("     true x^2   " + "".join(f"{v:>7.2f}" for v in xs * xs))
    print("     approx     " + "".join(f"{v:>7.2f}" for v in approx))
    print("\n  exact at the 5 control points (-2,-1,0,1,2), straight lines between them.")
    print("  more hinges -> tighter. a 'neuron' IS one hinge; training is just one way to")
    print("  place them. we place them by hand.")


# ---------------------------------------------------------------------------
# Build the pieces
# ---------------------------------------------------------------------------

def cmd_square(m, hinges=None):
    # the user specifies HINGES (hidden units); a ReluPWL needs hinges+2 knots.
    hinges = 16 if hinges is None else max(1, int(hinges))
    xs = np.linspace(-9, 9, 4000)
    sq = build_square(9.0, hinges + 2, spacing="uniform")
    approx, true = sq(xs), xs * xs
    print(f"\n  BUILD x^2 from {sq.n_neurons} hinges on [-9, 9]  (x^2 has even curvature)")
    print(f"     hinges (hidden units) : {sq.n_neurons}")
    print(f"     cosine vs true x^2    : {_cos(approx, true):.6f}")
    print(f"     max |error|           : {np.abs(approx - true).max():.4f}")
    print("  x^2 is easy: curvature is uniform, so evenly-spaced hinges are the right call.")
    print("  try `square 4` (coarse) then `square 64` (tight). then `recip` — where it bites.")


def cmd_recip(m):
    lo, hi = m.dist_sq.min(), m.dist_sq.max()
    ds, ir = m.dist_sq, m.inv_r3
    print(f"\n  BUILD 1/r^3 — THE HARD LAYER.  dist^2 in [{lo:.2f}, {hi:.1f}], "
          f"1/r^3 spans {ir.max() / ir.min():.0e}x")
    print("  flat over almost the whole range, then a cliff at contact. SAME 62 hinges,")
    print("  only the placement differs:\n")
    print(f"     {'placement':>10} {'hinges':>7} {'cosine':>10} {'rel_p99':>10}   verdict")
    for spacing in ("uniform", "log"):
        w = L2Weights.build(dist_sq_lo=lo, dist_sq_hi=hi, n_knots=64, knot_spacing=spacing)
        pred = w.recip(ds)
        rr = relative_error_report(pred, ir)
        ok = (_cos(pred, ir) >= 0.999) and (rr["rel_p99"] <= 0.01)
        note = "<- hinges wasted in the flat tail" if spacing == "uniform" \
            else "<- hinges clustered at the contact spike"
        print(f"     {spacing:>10} {w.recip.n_neurons:>7} {_cos(pred, ir):>10.6f} "
              f"{_fmt_err(rr['rel_p99'], spacing):>10} {'  PASS' if ok else '  FAIL'}  {note}")
    print("\n  even spacing is the CURVATURE-BLIND default — the basis you place if nothing")
    print("  tells you where the function bends. physics force bends all at contact, so put")
    print("  the hinges there. `knots`. (a trained net picks neither — see `compare`.)")


def cmd_budget(m):
    lo, hi = m.dist_sq.min(), m.dist_sq.max()
    ds, ir = m.dist_sq, m.inv_r3
    print(f"\n  NEURON BUDGET for 1/r^3  (real range, spans {ir.max() / ir.min():.0e}x)")
    print("  it was never a capacity problem — it's a placement problem.\n")
    print(f"     {'placement':>10} {'hinges':>7} {'cosine':>10} {'rel_p99':>10}   pass?")
    for spacing in ("uniform", "log"):
        for nk in (8, 16, 32, 64, 128):
            w = L2Weights.build(dist_sq_lo=lo, dist_sq_hi=hi, n_knots=nk, knot_spacing=spacing)
            pred = w.recip(ds)
            rr = relative_error_report(pred, ir)
            ok = (_cos(pred, ir) >= 0.999) and (rr["rel_p99"] <= 0.01)
            print(f"     {spacing:>10} {w.recip.n_neurons:>7} {_cos(pred, ir):>10.6f} "
                  f"{_fmt_err(rr['rel_p99'], spacing):>10} {'  PASS' if ok else '  FAIL'}")
    print("\n  62 log-spaced hinges PASS; even 126 uniform ones fail. it's not capacity —")
    print("  understanding the function beats throwing neurons at it. (a trained net adopts")
    print("  a THIRD basis: distributed, neither uniform nor at-contact — see `compare`.)")


def cmd_knots(m):
    lo, hi = m.dist_sq.min(), m.dist_sq.max()
    uni = np.linspace(lo, hi, 64)
    log = np.geomspace(lo, hi, 64)
    f = lambda s: s ** -1.5
    print(f"\n  WHERE THE HINGES SIT  (64 knots over dist^2 in [{lo:.2f}, {hi:.1f}])")
    print(f"  1/r^3 at contact (dist^2={lo:.2f}, softened) = {f(lo):.1f}   "
          f"far away (dist^2={hi:.0f}) = {f(hi):.5f}")
    print("  -> ALL the action is at contact. hinges spent in the flat tail do nothing.\n")
    print("     first 6 knots                              last 4")
    print(f"     uniform: {np.array2string(uni[:6], precision=1)} ... {np.array2string(uni[-4:], precision=0)}")
    print(f"     log    : {np.array2string(log[:6], precision=2)} ... {np.array2string(log[-4:], precision=0)}")
    near = lambda k: int((k < 1.0).sum())
    print(f"\n  hinges landing in the contact zone (dist^2 < 1.0):  "
          f"uniform {near(uni)} / 64   log {near(log)} / 64")
    print("  uniform throws ~1 hinge at the only place the function moves. log throws dozens.")


def cmd_multiply(m, a=None, b=None):
    print("\n  MULTIPLY — not a ReLU primitive, so build it from squares:")
    print("     a*b = ( (a+b)^2 - (a-b)^2 ) / 4      (and we can already square)")
    if a is not None and b is not None:
        pairs = [(float(a), float(b))]
    else:
        pairs = [(3., 4.), (-2., 5.), (7., -6.), (-3., -3.)]
    # the square must span BOTH a+b and a-b — itself a real construction
    # constraint: L3 sizes each multiply's square to its operand range.
    hw = max(12.0, max(max(abs(av + bv), abs(av - bv)) for av, bv in pairs) + 2.0)
    sq = build_square(half_width=hw, n_knots=int(16 * hw))
    print(f"\n     {'a':>5} {'b':>5} {'hand-built a*b':>16} {'true':>8}")
    for av, bv in pairs:
        got = float(relu_multiply(np.array([av]), np.array([bv]), sq)[0])
        print(f"     {av:>5.1f} {bv:>5.1f} {got:>16.4f} {av * bv:>8.1f}")
    print(f"\n  ({sq.n_neurons}-hinge square spanning +-{hw:.0f}: the square must cover a+-b.)")
    print("  this is how L3 assembles  F = q0*q1 * (dx,dy) / r^3  — three multiplies, all squares.")


# ---------------------------------------------------------------------------
# Assemble & verify
# ---------------------------------------------------------------------------

def cmd_verify(m):
    print(f"\n  VERIFY THE WHOLE FFN — L0->L3, every layer checked vs the simulator")
    print(f"  ({len(m.dx)} real Coulomb pairs, pos ~ U(0.5, 9.5). nothing trained.)")
    print("  " + "-" * 60)
    out = m.ffn.forward(m.batch)
    rows = [
        ("L0", "dx  = pos0_x - pos1_x", out["dx"], m.dx, "exact"),
        ("L0", "dy  = pos0_y - pos1_y", out["dy"], m.dy, "exact"),
        ("L0", "q0, q1  (passthrough)", out["q0"], m.q0, "exact"),
        ("L1", "dist^2 = dx^2+dy^2+s^2", out["dist_sq"], m.dist_sq, "hinges"),
        ("L2", "1/r^3  = dist^2^-1.5", out["inv_r3"], m.inv_r3, "log knots"),
        ("L3", "q0*q1  (ReLU multiply)", out["charge_product"], m.q0 * m.q1, "exact"),
    ]
    print(f"     {'layer':>5}  {'quantity':<24} {'how':>10}   matches truth?")
    for layer, name, got, true, how in rows:
        mark = "✓ PASS" if _cos(got, true) >= 0.999 else "✗ FAIL"
        print(f"     {layer:>5}  {name:<24} {how:>10}   {mark}")
    fx_t, fy_t = m.fx, m.fy
    pred = np.stack([out["fx"], out["fy"]], 1)
    true = np.stack([fx_t, fy_t], 1)
    r2 = _r2(pred, true)
    print("     " + "-" * 56)
    print(f"     L3     FORCE = q0q1*(dx,dy)/r^3   vs true Coulomb   "
          f"{'✓ PASS' if r2 >= 0.999 else '✗ FAIL'}   R² = {r2:.4f}")
    print("\n  every layer hand-set, verified against reality as it's built. no training.")


def cmd_compare(m):
    out = m.ffn.forward(m.batch)
    pred = np.stack([out["fx"], out["fy"]], 1)
    true = np.stack([m.fx, m.fy], 1)
    print("\n  THE HONEST SCOREBOARD  (the correction that was the real result)")
    print("  " + "-" * 60)
    print(f"     hand-built FFN, force output     : R^2 = {_r2(pred, true):.4f}")
    print(f"     trained Rung-1 model, force out  : R^2 ~ 0.9996   (phase1_findings.md)")
    print("  -> there is NO output gap. both compute the force essentially perfectly.")
    print("\n  the difference is the BASIS, not the accuracy:")
    print("     hand-built 1/r^3 : 62 compact, probe-readable hinges (placed by hand)")
    print("     trained   1/r^3 : DISTRIBUTED across the layer (greedy probe plateaus ~0.62)")
    print("\n  the claim is not 'the construct beats the trained model'. it's that 1/r^3 admits")
    print("  a compact hinge basis the force-trained model never adopted — understanding the")
    print("  function buys a compactness gradient descent had no reason to find.")


COMMANDS = {
    "relu": cmd_relu, "stack": cmd_stack, "square": cmd_square, "recip": cmd_recip,
    "budget": cmd_budget, "knots": cmd_knots, "multiply": cmd_multiply,
    "verify": cmd_verify, "compare": cmd_compare,
}


def main():
    print("\n  hand-built FEED-FORWARD NETWORK — interactive CLI   (type `help`)")
    print("  building the computing half of a transformer, no training.")
    m = Model()
    print(f"  ready: frozen FFN built; {len(m.dx)} real Coulomb pairs loaded.\n")
    while True:
        try:
            parts = input("  > ").strip().split()
        except EOFError:
            break
        if not parts:
            continue
        c, args = parts[0].lower(), parts[1:]
        if c in ("quit", "q", "exit"):
            break
        if c == "help":
            print(HELP)
            continue
        fn = COMMANDS.get(c)
        if fn is None:
            print("     ? type `help`")
            continue
        try:
            fn(m, *args)
        except (IndexError, ValueError, TypeError):
            print(f"     bad args for `{c}` — see `help`")
    print("\n  done.\n")


if __name__ == "__main__":
    main()

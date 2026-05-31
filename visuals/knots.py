#!/usr/bin/env python3
"""FFN DEMO — the money shot: WHERE you place the hinges is the whole game.

Same 62 ReLU hinges, same target (1/r^3 over the real range). Left: spaced
EVENLY (the curvature-blind default) — almost every hinge lands in the flat tail
where the function barely moves, and the approximation flies off the contact spike
(99th-pct relative error in the hundreds of percent). Right: spaced by LOG
(clustered where the curve actually bends) — the same hinges hug the spike
(~0.6%). Curvature-blind vs curvature-aware, made visible.

Every curve is the REAL frozen ReluPWL block from engine/frozen_ffn.py.

Run:  python3 visuals/knots.py
      MPLBACKEND=Agg python3 visuals/knots.py   # headless smoke
"""
# repo root on sys.path, so `import engine` resolves
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from _kit import ACCENT, BAD, BG, BLUE, FG, GOOD, MUTED, show

from engine.frozen_ffn import SOFT2, L2Weights, relative_error_report
from engine.multi_force import MultiForceParams, MultiForceSimulator


def _real_range(n=4000, seed=7):
    """The real dist^2 range under the simulator's pos ~ U(0.5, 9.5)."""
    rng = np.random.default_rng(seed)
    sim = MultiForceSimulator(MultiForceParams(n_particles=2))
    ds = np.zeros(n)
    for i in range(n):
        pos, _ = sim.sample_state(rng)
        dx = pos[0, 0] - pos[1, 0]; dy = pos[0, 1] - pos[1, 1]
        ds[i] = dx * dx + dy * dy + SOFT2
    return ds.min(), ds.max(), ds


def _panel(ax, spacing, lo, hi, ds, color, title_tag):
    w = L2Weights.build(dist_sq_lo=lo, dist_sq_hi=hi, n_knots=64, knot_spacing=spacing)
    knots = w.recip.inner_knots
    f = lambda s: s ** -1.5

    # zoom on the contact region — where 1/r^3 is large and all the action is
    xz = np.linspace(lo, 6.0, 2000)
    ax.plot(xz, f(xz), color=FG, lw=3.5, label="true 1/r³", zorder=3)
    ax.plot(xz, w.recip(xz), color=color, lw=2.5, ls="--",
            label="hand-built hinges", zorder=4)
    kz = knots[(knots >= lo) & (knots <= 6.0)]
    ax.plot(kz, f(kz), "o", color=color, ms=7, zorder=5,
            label=f"hinges here: {len(kz)} of {w.recip.n_neurons}")
    ax.scatter(knots, np.full_like(knots, -2.0), marker="|", s=200, color=color,
               clip_on=True, zorder=5)  # rug: every hinge along the axis

    rr = relative_error_report(w.recip(ds), f(ds))
    # uniform error wobbles run-to-run — show it categorically; log error is stable.
    pct = rr["rel_p99"] * 100.0
    err = (f"{pct:.1f}%" if spacing == "log"
           else ">10000%" if pct >= 10000 else ">1000%" if pct >= 1000
           else ">100%" if pct >= 100 else f"{pct:.0f}%")
    ax.set_title(f"{title_tag}\n{w.recip.n_neurons} hinges · 99th-pct error {err}",
                 color=color, fontsize=20, fontweight="bold", pad=14)
    ax.set_xlabel("separation²  (contact ← → apart)", color=FG, fontsize=15)
    ax.set_ylim(-4, 42); ax.set_xlim(lo, 6.0)
    ax.legend(loc="upper right", fontsize=13, facecolor=BG, edgecolor=MUTED, labelcolor=FG)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color(MUTED)


def main():
    lo, hi, ds = _real_range()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7.6))
    fig.patch.set_facecolor(BG)
    for ax in (axL, axR):
        ax.set_facecolor(BG)

    _panel(axL, "uniform", lo, hi, ds, BAD, "EVENLY spaced  (curvature-blind)")
    _panel(axR, "log", lo, hi, ds, GOOD, "LOG spaced  (curvature-aware)")
    axL.set_ylabel("force factor  1/r³", color=FG, fontsize=15)

    fig.suptitle("Same 62 hinges. Only the placement changed.",
                 color=FG, fontsize=26, fontweight="bold", y=0.99)
    fig.text(0.5, 0.015,
             "even spacing is the curvature-blind default — but physics-force "
             "curvature is all at contact.  place the hinges where it bends.",
             ha="center", color=ACCENT, fontsize=15)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    # real numbers to the terminal too
    print(f"\n  real dist² range: [{lo:.2f}, {hi:.1f}]  (1/r³ spans {(lo**-1.5)/(hi**-1.5):.0e}×)")
    for sp in ("uniform", "log"):
        w = L2Weights.build(dist_sq_lo=lo, dist_sq_hi=hi, n_knots=64, knot_spacing=sp)
        rr = relative_error_report(w.recip(ds), ds ** -1.5)
        pct = rr["rel_p99"] * 100.0
        lab = f"{pct:.1f}%" if sp == "log" else (">1000%" if pct >= 1000 else ">100%")
        print(f"  {sp:>8} 62 hinges: 99th-pct relative error = {lab:>8}")
    print()
    show()


if __name__ == "__main__":
    main()

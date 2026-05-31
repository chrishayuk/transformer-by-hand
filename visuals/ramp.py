#!/usr/bin/env python3
"""FFN DEMO — a neuron is a hinge; stack hinges and a curve forms (ANIMATED).

The motion IS the lesson. One ReLU hidden unit is a hinge — flat, then a ramp.
Drop hinges in one at a time and the piecewise-linear approximation tightens onto
the target curve (here x²). Every frame is the real frozen ReluPWL block from
engine/frozen_ffn.py — nothing is faked, nothing is trained.

Interactive backend → it animates (hinges drop in, the curve forms, the error
counter falls). Headless (MPLBACKEND=Agg) → it draws the final frame and exits,
so the same file smoke-tests and renders a still.

Run:  python3 visuals/ramp.py
      MPLBACKEND=Agg python3 visuals/ramp.py   # headless / still
"""
# repo root on sys.path, so `import engine` resolves
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from _kit import ACCENT, BG, BLUE, FG, GOOD, MUTED

from engine.frozen_ffn import ReluPWL

TARGET = lambda x: x * x
LO, HI = -3.0, 3.0
HINGE_COUNTS = [1, 1, 2, 3, 4, 5, 6, 8, 11, 11, 11]   # held start/end frames
XS = np.linspace(LO, HI, 800)
_ANIM = None   # keep the animation alive


def _draw(ax, n_hinges):
    """Redraw the panel for `n_hinges` (a ReluPWL through n_hinges+2 knots)."""
    ax.clear()
    ax.set_facecolor(BG)
    knots = np.linspace(LO, HI, n_hinges + 2)
    block = ReluPWL.fit(TARGET, knots)
    approx = block(XS)
    max_err = float(np.abs(approx - TARGET(XS)).max())

    ax.plot(XS, TARGET(XS), color=MUTED, lw=3, ls=":", label="target  x²", zorder=2)
    ax.plot(XS, approx, color=GOOD, lw=3.5, zorder=4,
            label=f"{n_hinges} hinge{'s' if n_hinges != 1 else ''}")
    ax.plot(knots, TARGET(knots), "o", color=GOOD, ms=10, zorder=5)
    # the single-hinge case: name the ramp so "a neuron is a hinge" lands
    if n_hinges == 1:
        kx = knots[1]
        ax.annotate("one neuron = one hinge\n(flat, then a ramp)", xy=(kx, TARGET(kx)),
                    xytext=(LO + 0.2, 7.2), color=FG, fontsize=15,
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))

    ax.set_xlim(LO, HI); ax.set_ylim(-0.6, 9.4)
    ax.set_xlabel("input x", color=FG, fontsize=15)
    ax.set_ylabel("output", color=FG, fontsize=15)
    ax.set_title(f"{n_hinges} hinge{'s' if n_hinges != 1 else ''}  ·  "
                 f"max error = {max_err:.2f}",
                 color=GOOD, fontsize=21, fontweight="bold", pad=12)
    ax.legend(loc="upper center", fontsize=14, facecolor=BG, edgecolor=MUTED, labelcolor=FG)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color(MUTED)


def main():
    global _ANIM
    fig, ax = plt.subplots(figsize=(11, 7.0))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Stack hinges and a curve forms — affine → ReLU → affine",
                 color=FG, fontsize=22, fontweight="bold", y=0.98)
    fig.text(0.5, 0.015, "a 'neuron' is a hinge · a layer is a sum of hinges · "
             "training is just one way to place them — here we place them by hand",
             ha="center", color=ACCENT, fontsize=14)

    _draw(ax, HINGE_COUNTS[-1])          # final frame: always drawn (still / headless)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    if plt.get_backend().lower() != "agg":          # interactive → animate
        _ANIM = FuncAnimation(fig, lambda i: _draw(ax, HINGE_COUNTS[i]),
                              frames=len(HINGE_COUNTS), interval=700, repeat=True)
        fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    final = ReluPWL.fit(TARGET, np.linspace(LO, HI, HINGE_COUNTS[-1] + 2))
    print(f"\n  hinge-stacking animation: 1 → {HINGE_COUNTS[-1]} hinges on x², "
          f"[{LO:.0f}, {HI:.0f}]")
    print(f"  final: {final.n_neurons} hinges, max |error| = "
          f"{np.abs(final(XS) - TARGET(XS)).max():.2f}  (exact at the knots, straight between)\n")

    if plt.get_backend().lower() != "agg":
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()

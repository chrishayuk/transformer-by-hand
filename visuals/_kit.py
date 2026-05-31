"""Shared palette + helpers for the two visual demos (ramp.py, knots.py).

Pure matplotlib — no engine, no model. Just the dark theme and a `show()` that
blocks on an interactive backend and no-ops headless (so the demos smoke-test
under MPLBACKEND=Agg):

    python3 visuals/knots.py                  # opens a live window
    MPLBACKEND=Agg python3 visuals/knots.py   # runs, shows nothing
"""

import matplotlib.pyplot as plt           # NO forced backend — interactive by default

# palette
BG = "#0d1117"; FG = "#e6edf3"; MUTED = "#8b949e"
GOOD = "#3fb950"; BAD = "#f85149"; ACCENT = "#d29922"; BLUE = "#58a6ff"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "text.color": FG,
    "axes.edgecolor": MUTED, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "font.size": 22, "font.family": "DejaVu Sans",
})


def show():
    """Display the figure(s) LIVE and block until closed, then clean up. Under a
    non-interactive backend (MPLBACKEND=Agg) this is a no-op, so the same script
    runs headless for smoke-tests."""
    plt.show()
    plt.close("all")

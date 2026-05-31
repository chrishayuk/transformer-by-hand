# Transformer, by hand

Build one transformer layer **by hand, from an empty grid, with no training** — and watch it run a real many-body physics calculation, matching a simulator to four decimal places. Every weight is set on camera, with a reason. Nothing is trained, nothing is downloaded, nothing is faked: every number these demos print is the real construction's own output.

This repo is the scripts + runnable demos for two companion videos:

| video | script | one line |
|---|---|---|
| **Attention Was Never the Smart Part** | [`video_scripts/attention_was_never_the_smart_part.md`](video_scripts/attention_was_never_the_smart_part.md) | build attention from zeros — it only *routes*; the value moves through **unchanged** |
| **The Smart Part Is Just Hinges** | [`video_scripts/the_smart_part_is_just_hinges.md`](video_scripts/the_smart_part_is_just_hinges.md) | open the feed-forward half — it's a stack of ReLU **hinges** you place by hand |

Two halves of one claim: **attention routes, the feed-forward network computes — neither is magic.**

## Quick start

```bash
pip install -r requirements.txt        # numpy + matplotlib, nothing else

# ── the teaching surface: build attention from zeros, the lines you type live
python3 by_hand/build_attention_by_hand.py    # the bare type-along (no banners)
python3 by_hand/worked_example.py             # the same build, fully printed at every step

# ── the receipts: the polished CLIs (type `help`)
python3 cli/attention_cli.py           # attention + the full hand-built layer (R²=0.9998)
python3 cli/ffn_cli.py                 # the feed-forward half, from a ReLU up

# ── the visuals (live matplotlib windows)
python3 visuals/ramp.py                # hinges drop in → a curve forms (animated)
python3 visuals/knots.py               # same 62 hinges, evenly vs log placed — the money shot
```

No GPU, no model download, no torch. Pure CPU, pure terminal (plus two matplotlib windows for the FFN visuals).

## What's here

```
video_scripts/                 # the two shooting scripts (narration + [DEMO]/[OUTPUT])
by_hand/
  build_attention_by_hand.py   # ★ bare numpy: the exact lines you type live, from empty
  worked_example.py            # the same build, fully printed at every step
cli/
  attention_cli.py             # attention "receipts": show / run / attn / build / ffn / mean / n <k>
  ffn_cli.py                   # FFN "receipts": relu / square / multiply / recip / knots / budget / verify / compare
visuals/
  ramp.py                      # a neuron is a hinge; stack hinges → any curve (animated)
  knots.py                     # where you place the hinges is the whole game
  _kit.py                      # shared matplotlib palette/helpers
engine/                        # vendored from sm-play (numpy only) — the wrapped math
  routing.py                   # the attention half — RoutingHead; this IS the by-hand build, reusable
  block.py                     # assemble the many-body force: route each neighbour, FFN, sum
  frozen_ffn.py                # the hand-built FFN (the hinges)
  multi_force.py               # the physics simulator (ground truth)
figures/                       # rendered stills used by the FFN script
```

The four `engine/` modules are self-contained copies vendored from the **sm-play** research repository, where this work originated. `engine/routing.py` is deliberately the same construction you build by hand in the first video. Treat sm-play as the upstream source of truth for the research context, tests, and write-ups.

## Headless / reproducibility

The demos are seed-locked (`seed 7`) and deterministic on a given machine; a different numpy/BLAS build can nudge the last digits. The two visuals run headless for a smoke-test:

```bash
MPLBACKEND=Agg python3 visuals/knots.py
MPLBACKEND=Agg python3 visuals/ramp.py
```

Each script's production notes spell out which numbers are stable laws (the PASS/FAIL verdicts, log `0.6%`, `R²=0.9998`) and which are samples not to freeze on screen (the uniform-knot error, the failing-`mean` R²).

## Requirements

Python 3.9+, `numpy`, `matplotlib`. That's the whole list.

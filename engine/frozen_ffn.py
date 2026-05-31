"""Hand-built frozen FFN that constructs Coulomb force computation layer-by-layer,
verified against the simulator at per-term granularity.

Pre-committed verification protocol (PER-TERM, NOT AGGREGATE):
  - Each layer's intermediate output is verified against the simulator's true
    quantity *individually*, not just the summed final force. Aggregate-looking-
    right is exactly how a partially-broken construction would pass.
  - Pass bar per term: cosine similarity ≥ 1 - 1e-6 (for terms that should
    construct exactly with linear ops), or cosine ≥ 0.999 with explicitly-
    reported reconstruction error magnitude for terms that require nonlinear
    approximation (notably 1/r³ — pre-flagged from Rung 1 as the candidate
    failure mode: 1/r³ was "distributed, plateaus at R²=0.62" in the trained
    Rung 1 model).
  - Failure mode to watch (Rung 1's flag): 1/r³ may not construct cleanly
    under any ReLU architecture without many neurons. If 1/r³ requires
    impractical capacity for high accuracy, THAT IS THE FINDING — exposing
    a limit on what's hand-constructible from the Rung 1 mechanistic spec.

Layer architecture (mapping the Rung 1 mechanistic spec to construction):
  - L0: linear feature extraction. dx, dy from raw positions; q0, q1 passthrough;
    dx², dy²-feeder values for L1's product computation. NO nonlinearity needed.
  - L1: ReLU product computation. Compute dx², dy² approximately via piecewise-
    linear ReLU stacks. Sum to get dist² (deferred to L2).
  - L2: reciprocal-power computation. dist_sq + softening², then 1/dist², then
    1/dist³ (the candidate failure mode).
  - L3: force assembly. k·q₀q₁·dx/r³ and k·q₀q₁·dy/r³.

THIS FILE: L0 only, with per-term verification. L1 (ReLU products) and L2
(1/r³ failure mode) are scoped but not implemented — they require fresh
design attention rather than session-end execution.

Input layout (matches MultiForceSimulator.build_input_vector for N=2):
  [pos0_x, pos0_y, pos1_x, pos1_y,        # 0..3  positions
   vel0_x, vel0_y, vel1_x, vel1_y,        # 4..7  velocities
   q0, q1,                                # 8..9  charges
   m0, m1,                                # 10..11 masses
   spring_k, rest_length, drag_b, gravity_G,  # 12..15 force params
   coulomb_active, spring_active, gravity_active, drag_active]  # 16..19
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Input slot indices (constants, matching MultiForceSimulator.build_input_vector for N=2)
SLOT_POS0_X = 0
SLOT_POS0_Y = 1
SLOT_POS1_X = 2
SLOT_POS1_Y = 3
SLOT_Q0 = 8
SLOT_Q1 = 9
INPUT_DIM = 20


# Quantities that L0 extracts and exposes to downstream layers
L0_OUTPUT_NAMES = [
    "dx",   # pos0_x - pos1_x
    "dy",   # pos0_y - pos1_y
    "q0",   # passthrough
    "q1",   # passthrough
    "q0_q1_indicator_proxy_unused",  # reserved slot for future use (e.g. coulomb_active)
]


@dataclass(frozen=True)
class L0Weights:
    """Frozen weight matrix for layer 0. Linear extraction only — no biases needed."""

    W: np.ndarray  # shape (5, 20): output_dim × input_dim
    b: np.ndarray  # shape (5,) — zeros for L0; included for interface uniformity

    @classmethod
    def build(cls) -> "L0Weights":
        """Construct the L0 weight matrix from first principles.

        dx = pos0_x - pos1_x  → row 0:  +1 at SLOT_POS0_X, -1 at SLOT_POS1_X
        dy = pos0_y - pos1_y  → row 1:  +1 at SLOT_POS0_Y, -1 at SLOT_POS1_Y
        q0 (passthrough)       → row 2:  +1 at SLOT_Q0
        q1 (passthrough)       → row 3:  +1 at SLOT_Q1
        (reserved)             → row 4:  zeros (placeholder)
        """
        W = np.zeros((5, INPUT_DIM), dtype=np.float64)
        W[0, SLOT_POS0_X] = +1.0
        W[0, SLOT_POS1_X] = -1.0
        W[1, SLOT_POS0_Y] = +1.0
        W[1, SLOT_POS1_Y] = -1.0
        W[2, SLOT_Q0] = +1.0
        W[3, SLOT_Q1] = +1.0
        # row 4 stays zero (reserved placeholder)
        b = np.zeros(5, dtype=np.float64)
        return cls(W=W, b=b)


def l0_forward(x: np.ndarray, weights: L0Weights) -> np.ndarray:
    """Apply L0 to a batch of inputs.

    x: (batch, 20) input batch (raw, not normalized — construct works in raw space)
    Returns: (batch, 5) L0 outputs in the order L0_OUTPUT_NAMES.

    L0→L1 interface note (UNVERIFIED design constraint, not established by the
    L0 tests): the L0 tests verify L0 is bit-correct as *extraction* (cosine ≥
    1−1e-6, max_abs_err < 1e-12 against simulator truth). That is not the same
    as verifying the L0→L1 interface. L1 needs dx, dy in a form its ReLU-
    product construction can *consume* — the output range and scale are
    themselves a design constraint, because the neuron count required to
    approximate dx² and dy² via ReLU stacks scales with the input range. L0-
    correct does NOT imply L1-buildable. The fresh-session L1 work must pre-
    commit the input range/scale L0 must deliver (and verify L0 actually
    delivers it) BEFORE choosing L1's architecture. Otherwise L0 green plus L1
    bit-correct on a small range still leaves the construction broken at
    realistic input ranges, and the failure would surface only at L3 when the
    aggregate force is wrong.
    """
    return x @ weights.W.T + weights.b


@dataclass
class PerTermVerificationResult:
    """One per-term check's result. Pre-committed pass criteria below."""

    term_name: str
    construct_output: np.ndarray
    simulator_truth: np.ndarray
    cosine: float
    max_abs_error: float
    rms_error: float

    def passes(self, exact: bool = True) -> bool:
        """For terms expected to construct exactly (linear ops only), require
        cosine ≥ 1 − 1e-6. For terms requiring nonlinear approximation, require
        cosine ≥ 0.999 (and the caller should also report max_abs_error)."""
        if exact:
            return self.cosine >= 1.0 - 1e-6
        return self.cosine >= 0.999


def per_term_verify(
    term_name: str, construct_output: np.ndarray, simulator_truth: np.ndarray
) -> PerTermVerificationResult:
    """Compare construct output to simulator truth on the same input batch.

    Pre-committed: per-term, not aggregate. Both inputs must be flattened to 1D
    for a single per-term scalar; the function flattens batched vectors so the
    same protocol applies whether the term is scalar-per-sample or vector-per-
    sample.
    """
    c = construct_output.astype(np.float64).ravel()
    t = simulator_truth.astype(np.float64).ravel()
    denom = np.linalg.norm(c) * np.linalg.norm(t)
    cosine = float(c @ t / denom) if denom > 1e-15 else 0.0
    max_abs_err = float(np.max(np.abs(c - t)))
    rms_err = float(np.sqrt(np.mean((c - t) ** 2)))
    return PerTermVerificationResult(
        term_name=term_name,
        construct_output=construct_output,
        simulator_truth=simulator_truth,
        cosine=cosine,
        max_abs_error=max_abs_err,
        rms_error=rms_err,
    )


# ---------------------------------------------------------------------------
# Frozen ReLU piecewise-linear workhorse
# ---------------------------------------------------------------------------
#
# Every nonlinear layer (L1 squaring, L2 1/r³, L3 multiply) reduces to the same
# primitive: a frozen 2-layer ReLU block that realises the exact piecewise-
# linear interpolant of a scalar function f at a chosen set of knots,
#
#     g(s) = base_slope * s + base_intercept + Σ_k w2[k] * relu(s − knot_k)
#
# This IS an FFN block — affine → ReLU → affine — with hand-set weights. The
# only design freedom is knot placement (and therefore neuron count), which is
# precisely the lever the construction exists to study: uniform knots are what
# an un-biased optimiser gravitates toward; log-spaced knots are the placement
# that makes a 69,000×-dynamic-range reciprocal cheap. See the L2 finding.

SOFTENING = 0.3            # matches MultiForceParams default
SOFT2 = SOFTENING ** 2     # 0.09 — the softening added inside dist_sq


@dataclass(frozen=True)
class ReluPWL:
    """Frozen ReLU piecewise-linear approximation of a scalar function.

    Realises the exact PWL interpolant of `f` through `knots` (sorted). Between
    knots it is linear; outside [knots[0], knots[-1]] it extends with the end
    segment slopes. As a network: hidden units relu(s − inner_knot_k), one
    affine read-in (the base line) and one affine read-out (w2). The number of
    hidden units is len(knots) − 2.
    """

    inner_knots: np.ndarray   # (H,)   interior knot positions (relu thresholds)
    w2: np.ndarray            # (H,)   read-out weights (slope changes)
    base_slope: float         # slope of the leftmost segment
    base_intercept: float     # intercept anchoring f at knots[0]
    lo: float                 # knots[0], for diagnostics
    hi: float                 # knots[-1], for diagnostics

    @classmethod
    def fit(cls, f, knots: np.ndarray) -> "ReluPWL":
        knots = np.asarray(knots, dtype=np.float64)
        assert np.all(np.diff(knots) > 0), "knots must be strictly increasing"
        fk = f(knots)
        seg_slope = np.diff(fk) / np.diff(knots)         # (K-1,)
        base_slope = float(seg_slope[0])
        base_intercept = float(fk[0] - base_slope * knots[0])
        # slope CHANGE injected at each interior knot
        w2 = np.diff(seg_slope)                            # (K-2,)
        return cls(
            inner_knots=knots[1:-1].copy(),
            w2=w2.copy(),
            base_slope=base_slope,
            base_intercept=base_intercept,
            lo=float(knots[0]),
            hi=float(knots[-1]),
        )

    @property
    def n_neurons(self) -> int:
        return len(self.inner_knots)

    def __call__(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        lin = self.base_slope * s + self.base_intercept
        relu = np.maximum(s[..., None] - self.inner_knots, 0.0)
        return lin + relu @ self.w2


def build_square(half_width: float, n_knots: int, spacing: str = "uniform") -> ReluPWL:
    """ReLU PWL approximation of x² on [−half_width, half_width].

    x² has constant curvature, so for *uniform* accuracy in x², uniform knots
    are optimal (max abs error between knots ≈ h²/4, h = 2·half_width/(n−1)).

    BUT uniform x²-accuracy is the WRONG objective when this square feeds the
    L2 reciprocal: 1/r³ has derivative −1.5·dist_sq^{−2.5} ≈ −617 at contact
    (dist_sq = 0.09), so a uniform absolute dist_sq error is amplified ~600× at
    small separation. `spacing="quadratic"` places knots as sign(u)·u²·half_width
    (u uniform in [−1,1]), concentrating them near x=0 — i.e. near small
    separation, where the force is large and the downstream reciprocal is steep.
    This buys near-contact precision at the cost of far-field precision the
    force does not need. See the L1→L2 amplification finding in the verifier.
    """
    if spacing == "uniform":
        knots = np.linspace(-half_width, half_width, n_knots)
    elif spacing == "quadratic":
        u = np.linspace(-1.0, 1.0, n_knots)
        knots = np.unique(np.sign(u) * (u ** 2) * half_width)
    else:
        raise ValueError(f"unknown spacing {spacing!r}")
    return ReluPWL.fit(lambda x: x * x, knots)


def relu_multiply(a: np.ndarray, b: np.ndarray, sq: ReluPWL) -> np.ndarray:
    """a·b via the squaring identity  a·b = ((a+b)² − (a−b)²) / 4.

    Multiplication is not a ReLU primitive; this is the standard reduction to
    squaring. `sq` must cover the range of (a±b). Because it subtracts two
    PWL-squared terms evaluated at nearby points, the squaring errors largely
    cancel when one operand is small — but the residual is the dynamic-range
    cost L3 verification watches for.
    """
    return (sq(a + b) - sq(a - b)) / 4.0


# ---------------------------------------------------------------------------
# L1 — ReLU squaring → dist_sq = dx² + dy² + softening²
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class L1Weights:
    """Frozen squaring block, applied to dx and dy, summed with the softening."""

    square: ReluPWL

    @classmethod
    def build(cls, half_width: float = 9.0, n_knots: int = 64,
              spacing: str = "quadratic") -> "L1Weights":
        # dx, dy ∈ [−9, 9] under the real distribution (pos ∈ U(0.5, 9.5)).
        # Default "quadratic" (denser near 0) because this square feeds L2's
        # reciprocal, which amplifies near-contact error ~600× — verified.
        return cls(square=build_square(half_width, n_knots, spacing=spacing))


def l1_forward(dx: np.ndarray, dy: np.ndarray, weights: L1Weights) -> np.ndarray:
    """dist_sq = dx² + dy² + softening². Per-term target: simulator dist_sq.

    Caveat made concrete (the docstring's L0→L1 warning, now at L1→L2): verified
    *in isolation* against dist_sq, uniform knots look fine (cosine≈1, abs err
    ~0.04). But the COMPOSED chain L1→L2 degrades, because 1/r³'s ~600× contact
    derivative amplifies that 0.04 into a ~15-unit inv_r3 error. Per-term
    verification must run on the COMPOSED output, not each term fed ground truth.
    """
    return weights.square(dx) + weights.square(dy) + SOFT2


# ---------------------------------------------------------------------------
# L2 — reciprocal power  inv_r3 = dist_sq^{-1.5}   (the pre-flagged failure mode)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class L2Weights:
    """Frozen 1/r³ block. knot_spacing ∈ {"log", "uniform"} — the lever."""

    recip: ReluPWL
    knot_spacing: str

    @classmethod
    def build(
        cls,
        dist_sq_lo: float = 0.09,
        dist_sq_hi: float = 152.0,
        n_knots: int = 64,
        knot_spacing: str = "log",
    ) -> "L2Weights":
        if knot_spacing == "log":
            knots = np.geomspace(dist_sq_lo, dist_sq_hi, n_knots)
        elif knot_spacing == "uniform":
            knots = np.linspace(dist_sq_lo, dist_sq_hi, n_knots)
        else:
            raise ValueError(f"unknown knot_spacing {knot_spacing!r}")
        return cls(recip=ReluPWL.fit(lambda s: s ** -1.5, knots), knot_spacing=knot_spacing)


def l2_forward(dist_sq: np.ndarray, weights: L2Weights) -> np.ndarray:
    """inv_r3 = dist_sq^{-1.5}.

    PASS BAR (pre-committed, cosine alone is INSUFFICIENT — it is scale-
    invariant and dominated by near-contact spikes): cosine ≥ 0.999 AND a
    RELATIVE-error bar across the distribution (rel99 ≤ 0.01). See
    `relative_error_report`. If the bar needs impractical neurons, that is the
    finding; do not lower it or train the layer.
    """
    return weights.recip(dist_sq)


# ---------------------------------------------------------------------------
# L3 — force assembly  F = q₀·q₁·(dx, dy)·inv_r3
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class L3Weights:
    """Three squaring blocks for the three multiplications in the assembly."""

    sq_charge: ReluPWL   # q0·q1            (operands ∈ {−1,0,1})
    sq_disp: ReluPWL     # (q0q1)·dx        (operands ∈ [−1,1]×[−9,9])
    sq_force: ReluPWL    # (q0q1·dx)·inv_r3 (operands ∈ [−9,9]×[0,~37])

    @classmethod
    def build(cls) -> "L3Weights":
        # Charges are discrete in {−1,0,1}; a±b ∈ {−2..2}. Knots on every
        # half-integer hit the integer operand sums exactly → q0·q1 is exact.
        sq_charge = build_square(half_width=2.0, n_knots=9)
        # q0q1 ∈ {−1,0,1}, dx ∈ [−9,9] → a±b ∈ [−10,10].
        sq_disp = build_square(half_width=10.0, n_knots=128)
        # q0q1·dx ∈ [−9,9], inv_r3 ∈ [0,~37] → a±b ∈ [−46,46]; widest range,
        # most knots. This is where multiply cancellation is stressed.
        sq_force = build_square(half_width=46.0, n_knots=256)
        return cls(sq_charge=sq_charge, sq_disp=sq_disp, sq_force=sq_force)


def l3_forward(
    dx: np.ndarray,
    dy: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    inv_r3: np.ndarray,
    weights: L3Weights,
) -> dict:
    """Assemble the Coulomb force on particle 0.

    F = q₀·q₁·(dx, dy)/r³, built as a cascade of ReLU multiplies. Returns each
    intermediate so per-term verification can check charge_product and the
    displacement product, not just the final force.
    """
    qq = relu_multiply(q0, q1, weights.sq_charge)        # q0·q1
    ux = relu_multiply(qq, dx, weights.sq_disp)          # q0q1·dx
    uy = relu_multiply(qq, dy, weights.sq_disp)          # q0q1·dy
    fx = relu_multiply(ux, inv_r3, weights.sq_force)     # × 1/r³
    fy = relu_multiply(uy, inv_r3, weights.sq_force)
    return {"charge_product": qq, "ux": ux, "uy": uy, "fx": fx, "fy": fy}


# ---------------------------------------------------------------------------
# Full chain L0 → L3
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstructFFN:
    """The whole hand-built frozen FFN, weights only — no training, ever."""

    l0: L0Weights
    l1: L1Weights
    l2: L2Weights
    l3: L3Weights

    @classmethod
    def build(cls, l2_n_knots: int = 64, l2_knot_spacing: str = "log") -> "ConstructFFN":
        return cls(
            l0=L0Weights.build(),
            l1=L1Weights.build(),
            l2=L2Weights.build(n_knots=l2_n_knots, knot_spacing=l2_knot_spacing),
            l3=L3Weights.build(),
        )

    def forward(self, x: np.ndarray) -> dict:
        """Raw input batch (batch, 20) → all intermediates + final (fx, fy).

        Every key is a per-term-verifiable quantity against the simulator.
        """
        l0 = l0_forward(x, self.l0)
        dx, dy, q0, q1 = l0[:, 0], l0[:, 1], l0[:, 2], l0[:, 3]
        dist_sq = l1_forward(dx, dy, self.l1)
        inv_r3 = l2_forward(dist_sq, self.l2)
        l3 = l3_forward(dx, dy, q0, q1, inv_r3, self.l3)
        return {
            "dx": dx, "dy": dy, "q0": q0, "q1": q1,
            "dist_sq": dist_sq, "inv_r3": inv_r3,
            **l3,
        }


def relative_error_report(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Relative-error percentiles — the magnitude bar cosine cannot see.

    Used for the reciprocal-power and force terms, whose dynamic range makes a
    scale-invariant cosine misleading. Floors |truth| to avoid div-by-zero on
    exact-zero charge products.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    denom = np.maximum(np.abs(truth), 1e-12)
    rel = np.abs(pred - truth) / denom
    return {
        "rel_median": float(np.median(rel)),
        "rel_p99": float(np.percentile(rel, 99)),
        "rel_max": float(np.max(rel)),
    }

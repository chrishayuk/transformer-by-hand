"""Multi-force simulator for the superposition rung.

Computes forces from multiple physics laws (Coulomb, Spring, Gravity, Drag)
on N-particle systems. Each force law can be independently activated per sample,
enabling controlled co-occurrence studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class MultiForceParams:
    n_particles: int = 2
    box_size: float = 10.0
    softening: float = 0.3
    charge_classes: List[float] = field(default_factory=lambda: [-1.0, 0.0, 1.0])
    mass_range: Tuple[float, float] = (0.5, 2.0)
    spring_k_range: Tuple[float, float] = (0.1, 1.0)
    rest_length_range: Tuple[float, float] = (1.0, 3.0)
    gravity_G: float = 1.0
    coulomb_k: float = 1.0
    drag_b_range: Tuple[float, float] = (0.1, 0.5)


FORCE_NAMES = ["coulomb", "spring", "gravity", "drag"]


class MultiForceSimulator:
    def __init__(self, params: MultiForceParams | None = None):
        self.p = params or MultiForceParams()

    def sample_state(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        N = self.p.n_particles
        margin = 0.5
        positions = rng.uniform(margin, self.p.box_size - margin, size=(N, 2))
        velocities = rng.normal(0, 1.0, size=(N, 2))
        return positions, velocities

    def sample_particle_properties(
        self, rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        N = self.p.n_particles
        charges = rng.choice(self.p.charge_classes, size=N)
        masses = rng.uniform(*self.p.mass_range, size=N)
        return {"charges": charges, "masses": masses}

    def sample_force_params(
        self, rng: np.random.Generator
    ) -> Dict[str, float]:
        return {
            "spring_k": rng.uniform(*self.p.spring_k_range),
            "rest_length": rng.uniform(*self.p.rest_length_range),
            "drag_b": rng.uniform(*self.p.drag_b_range),
        }

    def compute_coulomb(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
    ) -> np.ndarray:
        N = len(positions)
        diff = positions[:, None, :] - positions[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1) + self.p.softening**2
        np.fill_diagonal(dist_sq, 1.0)
        charge_product = charges[:, None] * charges[None, :]
        f_scalar = self.p.coulomb_k * charge_product / (dist_sq**1.5)
        np.fill_diagonal(f_scalar, 0.0)
        return (f_scalar[:, :, None] * diff).sum(axis=1)

    def compute_spring(
        self,
        positions: np.ndarray,
        spring_k: float,
        rest_length: float,
    ) -> np.ndarray:
        N = len(positions)
        diff = positions[:, None, :] - positions[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1) + self.p.softening**2)
        np.fill_diagonal(dist, 1.0)
        displacement = dist - rest_length
        direction = diff / (dist[:, :, None] + 1e-10)
        f_scalar = -spring_k * displacement
        np.fill_diagonal(f_scalar, 0.0)
        return (f_scalar[:, :, None] * direction).sum(axis=1)

    def compute_gravity(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        N = len(positions)
        diff = positions[:, None, :] - positions[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1) + self.p.softening**2
        np.fill_diagonal(dist_sq, 1.0)
        mass_product = masses[:, None] * masses[None, :]
        f_scalar = -self.p.gravity_G * mass_product / (dist_sq**1.5)
        np.fill_diagonal(f_scalar, 0.0)
        return (f_scalar[:, :, None] * diff).sum(axis=1)

    def compute_drag(
        self,
        velocities: np.ndarray,
        drag_b: float,
    ) -> np.ndarray:
        return -drag_b * velocities

    def compute_forces(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        properties: Dict[str, np.ndarray],
        force_params: Dict[str, float],
        active_tasks: Dict[str, bool],
    ) -> np.ndarray:
        forces = np.zeros_like(positions)
        if active_tasks.get("coulomb", False):
            forces += self.compute_coulomb(positions, properties["charges"])
        if active_tasks.get("spring", False):
            forces += self.compute_spring(
                positions, force_params["spring_k"], force_params["rest_length"]
            )
        if active_tasks.get("gravity", False):
            forces += self.compute_gravity(positions, properties["masses"])
        if active_tasks.get("drag", False):
            forces += self.compute_drag(velocities, force_params["drag_b"])
        return forces

    def compute_ground_truth_features(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        properties: Dict[str, np.ndarray],
        force_params: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute all ground-truth intermediate features for probing.

        Returns scalar features for the N=2 case. For N>2 this would need
        to return per-pair features; keep N=2 for now.
        """
        assert len(positions) == 2, "Ground-truth features currently only support N=2"

        dx = positions[1, 0] - positions[0, 0]
        dy = positions[1, 1] - positions[0, 1]
        dist_sq = dx**2 + dy**2 + self.p.softening**2
        dist = np.sqrt(dist_sq)
        inv_r2 = 1.0 / dist_sq
        inv_r3 = inv_r2 / dist

        q0, q1 = properties["charges"]
        m0, m1 = properties["masses"]
        charge_product = q0 * q1
        mass_product = m0 * m1

        spring_k = force_params["spring_k"]
        rest_length = force_params["rest_length"]
        displacement = dist - rest_length
        spring_force_mag = spring_k * displacement

        drag_b = force_params["drag_b"]

        return {
            "dx": dx,
            "dy": dy,
            "dist": dist,
            "dist_sq": dist_sq,
            "inv_r2": inv_r2,
            "inv_r3": inv_r3,
            "charge_product": charge_product,
            "mass_product": mass_product,
            "displacement": displacement,
            "spring_force_mag": spring_force_mag,
            "drag_b_vx0": drag_b * velocities[0, 0],
            "drag_b_vy0": drag_b * velocities[0, 1],
        }

    def build_input_vector(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        properties: Dict[str, np.ndarray],
        force_params: Dict[str, float],
        active_tasks: Dict[str, bool],
    ) -> np.ndarray:
        """Build the flat input vector for the model.

        Layout: [particle_states | charges | masses | spring_k | rest_length |
                 drag_b | gravity_G | task_indicators]
        """
        N = self.p.n_particles
        states = np.concatenate([positions.ravel(), velocities.ravel()])
        charges = properties["charges"]
        masses = properties["masses"]
        params = np.array([
            force_params["spring_k"],
            force_params["rest_length"],
            force_params["drag_b"],
            self.p.gravity_G,
        ])
        task_indicators = np.array([
            float(active_tasks.get(name, False)) for name in FORCE_NAMES
        ])
        return np.concatenate([states, charges, masses, params, task_indicators])

    @staticmethod
    def input_dim(n_particles: int) -> int:
        return n_particles * 4 + n_particles * 2 + 4 + 4

    @staticmethod
    def output_dim(n_particles: int) -> int:
        return n_particles * 2

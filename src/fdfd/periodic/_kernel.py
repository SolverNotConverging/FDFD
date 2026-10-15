"""Shared periodic (Bloch) kernel: a unit cell that repeats along z.

The eigenvalue is ``gamma`` in the Bloch factor ``exp(-gamma*z)`` of a
periodic envelope, so ``neff = -i*gamma/k0`` under ``exp(+i*omega*t)`` and
passive forward modes have ``Im(neff) <= 0``. Unknowns are physical fields
(SI units); the pencil ``A v = gamma B v`` has a z-averaging mass matrix
``B`` that may be singular for an even number of cells.
"""

import numpy as np

from ..core import eigs, yee
from ..core.constants import C0
from ..core.errors import SolverError
from ..core.grid import CellMaterials

FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


class PeriodicKernel(CellMaterials):
    """Materials and PEC/PMC volumes on a z-periodic Yee cell."""

    axes = ()
    periodic = ("z",)

    def __init__(self, frequency, lengths, cells):
        self.frequency = float(frequency)
        self.omega = 2 * np.pi * self.frequency
        self.k0 = self.omega / C0
        self.cells = tuple(int(n) for n in cells)
        self.steps = tuple(length / n for length, n in zip(lengths, self.cells, strict=True))
        self._init_materials(self.cells)
        self.field_shapes = {
            name: yee.staggered_shape(self.cells, name[1].lower(), self.axes, name[0], periodic=self.periodic)
            for name in FIELDS
        }
        self.sizes = {name: int(np.prod(shape)) for name, shape in self.field_shapes.items()}
        self._pec_cells = np.zeros(self.cells, dtype=bool)
        self._pmc_cells = np.zeros(self.cells, dtype=bool)

    def add_pec(self, cell_mask):
        """A PEC volume: tangential E and normal H vanish on and inside it."""
        self._pec_cells |= np.asarray(cell_mask, dtype=bool)

    def add_pmc(self, cell_mask):
        """A PMC volume: the dual constraints, applied strictly inside it."""
        self._pmc_cells |= np.asarray(cell_mask, dtype=bool)

    def update_component_materials(self):
        return self.materials_and_masks()

    def _outer_boundary(self, masks):
        """Hook for constraints on the transverse domain boundary."""

    def materials_and_masks(self):
        """Component materials and the six constraint masks (Ex, Ey, Ez on E; Hx, Hy, Hz on H)."""
        masks = {}
        for name in FIELDS:
            field, component = name[0], name[1].lower()
            masks[name] = yee.footprint(
                self._pec_cells, component, self.axes, field, periodic=self.periodic
            ) | yee.footprint(
                self._pmc_cells, component, self.axes, field, periodic=self.periodic, interior=True
            )
        self._outer_boundary(masks)
        eps, mu = self.cell_tensors()
        materials = {}
        for index, component in enumerate("xyz"):
            for prefix, tensor, field in (("eps_", eps, "E"), ("mu_", mu, "H")):
                values = yee.staggered_average(
                    tensor[index], component, self.axes, field, periodic=self.periodic
                )
                values[masks[field + component]] = 1.0
                materials[prefix + component * 2] = values
        return materials, masks

    def _eigenpairs(
        self, A, B, free, *, num_modes, sigma, tol, device, method, ncv, max_restarts, random_seed, v0=None
    ):
        free_count = int(np.count_nonzero(free))
        if free_count <= num_modes + 1:
            raise ValueError(f"Not enough unconstrained DOFs ({free_count}) to solve {num_modes} modes.")
        A, B = A[free][:, free], B[free][:, free]
        pairs = eigs.solve(
            A,
            B,
            k=num_modes,
            sigma=sigma,
            method=method,
            device=device,
            tol=tol,
            ncv=ncv,
            v0=None if v0 is None else v0(A.shape[0]),
            max_restarts=max_restarts,
            random_seed=random_seed,
        )
        limit = max(1e-8, 100.0 * float(tol))
        if not np.isfinite(pairs.values).all() or np.max(pairs.residuals) > limit:
            raise SolverError(
                f"Periodic eigenpairs failed the backward-error test: {np.max(pairs.residuals):.3e}."
            )
        order = np.argsort(np.abs(pairs.values - sigma), kind="stable")
        vectors = np.zeros((free.size, num_modes), dtype=complex)
        vectors[free] = pairs.vectors[:, order]
        return pairs.values[order], vectors, pairs.residuals[order]

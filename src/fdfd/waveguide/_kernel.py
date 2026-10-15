"""Shared waveguide kernel: Yee cross-section, conductors and SIBC bookkeeping.

The cross-section has one or two transverse axes (x, or x and y); z is the
propagation axis with ``exp(-i*beta*z)``. Along a transverse axis ``a``,
``E_c`` sits on nodes for ``a != c`` and on cell centres for ``a == c``; ``H_c``
is the dual. Conductors are given as cell masks and constrain the field
samples on their footprint. Stored magnetic fields are ``-i*eta0*H``.
"""

from dataclasses import dataclass

import numpy as np

from ..core import yee
from ..core.constants import C0
from ..core.grid import CellMaterials

COMPONENTS = ("xx", "yy", "zz")
FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


@dataclass(frozen=True)
class ModeSolution:
    """Modes of one kernel solve, sorted by descending ``Re(neff**2)``."""

    polarization: str
    neff: np.ndarray
    fields: dict
    residuals: np.ndarray
    field_residuals: np.ndarray


def equation_residual(*terms):
    """Relative residual of a reconstructed discrete field equation."""
    numerator = np.max(abs(sum(terms)), axis=0)
    denominator = sum(np.max(abs(term), axis=0) for term in terms)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


class WaveguideKernel(CellMaterials):
    """Material, conductor and surface-impedance state shared by 1D and 2D kernels."""

    def __init__(self, frequency, lengths, cells):
        self.frequency = float(frequency)
        self.k0 = 2 * np.pi * self.frequency / C0
        self.cells = tuple(int(n) for n in cells)
        self.steps = tuple(length / n for length, n in zip(lengths, self.cells, strict=True))
        self.normalized_steps = tuple(self.k0 * step for step in self.steps)
        self._init_materials(self.cells)
        self.field_shapes = {name: self._component_shape(name) for name in FIELDS}
        for kind, field in (("pec", "E"), ("pmc", "H")):
            for axis, component in zip("xyz", COMPONENTS, strict=True):
                setattr(
                    self, f"{kind}_{component}_mask", np.zeros(self.field_shapes[field + axis], dtype=bool)
                )
        self._pec_cell_mask = np.zeros(self.cells, dtype=bool)
        self._pmc_cell_mask = np.zeros(self.cells, dtype=bool)
        self._surface_impedance_owner = np.full(self.cells, -1, dtype=np.int32)
        self._surface_impedance_definitions = []
        self._compiled_impedance_boundary = None

    @property
    def _axes(self):
        return ("x", "y")[: len(self.cells)]

    def _component_shape(self, name):
        return yee.staggered_shape(self.cells, name[1].lower(), self._axes, name[0])

    def component_masks(self, cell_mask, field):
        """Expand a cell mask onto the ``(xx, yy, zz)`` samples of ``"E"`` or ``"H"``."""
        return tuple(yee.footprint(cell_mask, c, self._axes, field) for c in "xyz")

    def _on_fields(self, eps, mu):
        """Average cell tensors onto the staggered component locations."""
        result = {}
        for index, (axis, component) in enumerate(zip("xyz", COMPONENTS, strict=True)):
            result["eps_" + component] = yee.staggered_average(eps[index], axis, self._axes, "E")
            result["mu_" + component] = yee.staggered_average(mu[index], axis, self._axes, "H")
        return result

    @property
    def impedance_surface_mask(self):
        """Cells owned by a surface-impedance region."""
        return self._surface_impedance_owner >= 0

    def add_pec(self, cell_mask):
        """Constrain tangential E and normal H on a PEC cell region."""
        cell_mask = np.asarray(cell_mask, dtype=bool)
        if np.any(cell_mask & self.impedance_surface_mask):
            raise ValueError("PEC region overlaps a surface-impedance region.")
        self._pec_cell_mask |= cell_mask
        for target, mask in zip(
            (self.pec_xx_mask, self.pec_yy_mask, self.pec_zz_mask),
            self.component_masks(cell_mask, "E"),
            strict=True,
        ):
            target |= mask

    def add_pmc(self, cell_mask):
        """Constrain tangential H and normal E on a PMC cell region."""
        cell_mask = np.asarray(cell_mask, dtype=bool)
        if np.any(cell_mask & self.impedance_surface_mask):
            raise ValueError("PMC region overlaps a surface-impedance region.")
        self._pmc_cell_mask |= cell_mask
        for target, mask in zip(
            (self.pmc_xx_mask, self.pmc_yy_mask, self.pmc_zz_mask),
            self.component_masks(cell_mask, "H"),
            strict=True,
        ):
            target |= mask

    def add_impedance_surface(self, cell_mask, impedance):
        """Mark opaque cells whose exposed faces obey the scalar SIBC ``Zs = impedance``."""
        from ._impedance_1d import SurfaceImpedanceDefinition

        region = np.asarray(cell_mask, dtype=bool)
        impedance = complex(impedance)
        if not np.isfinite(impedance) or impedance == 0 or impedance.real < 0:
            raise ValueError("Surface impedance must be finite, nonzero and passive (Re(Zs) >= 0).")
        for label, conflict in (
            ("PEC", self._pec_cell_mask),
            ("PMC", self._pmc_cell_mask),
            ("PML", self._pml_cell_mask),
        ):
            if np.any(region & conflict):
                raise ValueError(f"Surface-impedance region overlaps an existing {label} region.")
        key = ("constant", impedance.real, impedance.imag)
        index = next((i for i, d in enumerate(self._surface_impedance_definitions) if d.key == key), None)
        owners = np.unique(self._surface_impedance_owner[region])
        owners = owners[owners >= 0]
        if owners.size and (index is None or np.any(owners != index)):
            raise ValueError("Surface-impedance regions with different impedances overlap.")
        if np.all(self.impedance_surface_mask | region):
            raise ValueError("Surface-impedance geometry leaves no retained field cells.")
        self._validate_pml_separation(self.impedance_surface_mask | region, self._pml_cell_mask)
        if index is None:
            index = len(self._surface_impedance_definitions)
            self._surface_impedance_definitions.append(
                SurfaceImpedanceDefinition(key=key, impedance=impedance, label=f"Zs={impedance!r}")
            )
        self._surface_impedance_owner[region & (self._surface_impedance_owner < 0)] = index

    def set_pml_cells(self, mask):
        mask = np.asarray(mask, dtype=bool)
        self._validate_pml_separation(self.impedance_surface_mask, mask)
        self._pml_cell_mask = mask

    def update_component_materials(self):
        """Interpolate materials to the Yee components and compile constraints (validates SIBC)."""
        return self._effective_materials_and_masks()

    def _effective_materials_and_masks(self):
        eps, mu = (tuple(np.array(v, copy=True) for v in tensor) for tensor in self.cell_tensors())
        pec = [m.copy() for m in (self.pec_xx_mask, self.pec_yy_mask, self.pec_zz_mask)]
        pmc = [m.copy() for m in (self.pmc_xx_mask, self.pmc_yy_mask, self.pmc_zz_mask)]
        # A tangential PEC sample on Ex (Ey) also fixes the collocated normal
        # Hy (Hx) as PMC, and vice versa, so each pair uses the union.
        ex_hy, ey_hx = pec[0] | pmc[1], pec[1] | pmc[0]
        pec[0], pmc[1], pec[1], pmc[0] = ex_hy, ex_hy.copy(), ey_hx, ey_hx.copy()
        materials = self._on_fields(eps, mu)
        if np.any(self.impedance_surface_mask):
            boundary = self._compile_impedance(eps, mu, materials, pec, pmc)
            self._compiled_impedance_boundary = boundary
            for constraint, retained in zip(pec, boundary.electric_retained, strict=True):
                constraint |= ~retained
            for constraint, retained in zip(pmc, boundary.magnetic_retained, strict=True):
                constraint |= ~retained
        else:
            self._compiled_impedance_boundary = None
        for component, mask in zip(COMPONENTS, pec, strict=True):
            materials["eps_" + component][mask] = 1.0
        for component, mask in zip(COMPONENTS, pmc, strict=True):
            materials["mu_" + component][mask] = 1.0
        return materials, (*pec, *pmc)

    def _zero_constrained(self, fields, masks):
        for name, mask in zip(FIELDS, masks, strict=True):
            fields[name][mask, :] = 0.0

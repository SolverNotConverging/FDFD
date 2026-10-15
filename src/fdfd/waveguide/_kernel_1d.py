"""1D waveguide kernel: TE (Ey, Hx, Hz) and TM (Hy, Ex, Ez) slab modes.

Lengths are normalized by ``k0``, so ``neff**2`` is the eigenvalue of the
reduced transverse operator. Fields vary as ``exp(+i*omega*t - i*beta*z)``.
"""

import numpy as np

from ..core import eigs, yee
from ..core.constants import EPSILON_0
from ._impedance_1d import compile_impedance_boundary, validate_impedance_pml_separation
from ._kernel import FIELDS, ModeSolution, WaveguideKernel, equation_residual


class WaveguideKernel1D(WaveguideKernel):
    """Slab cross-section of ``Nx`` cells along x."""

    def __init__(self, frequency, length, nx):
        super().__init__(frequency, (length,), (nx,))
        self.Nx = self.cells[0]
        self.dx = self.steps[0]

    @staticmethod
    def _validate_pml_separation(opaque, pml_cells):
        validate_impedance_pml_separation(opaque, pml_cells)

    def _compile_impedance(self, eps, mu, materials, pec, pmc):
        boundary = compile_impedance_boundary(
            owner=self._surface_impedance_owner,
            definitions=tuple(self._surface_impedance_definitions),
            cell_eps_r_yy=eps[1],
            cell_eps_r_zz=eps[2],
            dx=self.dx,
            frequency=self.frequency,
            epsilon0=EPSILON_0,
            pml_cells=self._pml_cell_mask,
        )
        for row in boundary.rows:
            node, cell = row.electric_index, row.retained_cell_index
            if pec[1][node] or pec[2][node]:
                raise ValueError(
                    f"Surface-impedance boundary conflicts with a PEC/PMC constraint at node {node}."
                )
            if pmc[1][cell] or pmc[2][cell]:
                raise ValueError(
                    f"Surface-impedance boundary has a PEC/PMC constraint on magnetic cell {cell}."
                )
            materials["eps_yy"][node] = row.relative_permittivity_yy
            materials["eps_zz"][node] = row.relative_permittivity_zz
            materials["mu_xx"][node] = mu[0][cell]
        return boundary

    def derivatives(self):
        """Normalized node-to-cell and cell-to-node derivatives, with SIBC Ampere rows."""
        e_to_h = yee.difference((self.Nx + 1,), (self.Nx,), 0, self.normalized_steps[0])
        h_to_e = -e_to_h.conj().T
        boundary = self._compiled_impedance_boundary
        if boundary is not None and boundary.rows:
            # Replace the full-cell Ampere difference by the clipped half-cell row.
            editable = h_to_e.tolil(copy=True)
            for row in boundary.rows:
                editable.rows[row.electric_index] = [row.retained_cell_index]
                editable.data[row.electric_index] = [row.magnetic_coefficient / self.k0]
            h_to_e = editable.tocsr()
        return e_to_h, h_to_e

    def solve(self, num_modes, sigma, *, polarizations=("TE", "TM"), tol=0.0, device="cpu", method="arpack"):
        """Return a :class:`ModeSolution` per requested polarization."""
        materials, masks = self._effective_materials_and_masks()
        _, pec_yy, pec_zz, _, pmc_yy, pmc_zz = masks
        e_to_h, h_to_e = self.derivatives()
        eps_xx, eps_yy = yee.diag(materials["eps_xx"]), yee.diag(materials["eps_yy"])
        mu_xx, mu_yy = yee.diag(materials["mu_xx"]), yee.diag(materials["mu_yy"])
        eps_zz_inv = yee.inverse_diag(materials["eps_zz"], pec_zz)
        mu_zz_inv = yee.inverse_diag(materials["mu_zz"], pmc_zz)
        operators = {
            "TE": (-mu_xx @ (h_to_e @ mu_zz_inv @ e_to_h + eps_yy), ~pec_yy, self.Nx + 1),
            "TM": (-eps_xx @ (e_to_h @ eps_zz_inv @ h_to_e + mu_yy), ~pmc_yy, self.Nx),
        }
        solutions = []
        for polarization in polarizations:
            omega, free, size = operators[polarization]
            reduced = omega[free][:, free]
            pairs = eigs.solve(reduced, k=num_modes, sigma=sigma, tol=tol, device=device, method=method)
            order = np.argsort(np.real(pairs.values))
            values = pairs.values[order]
            primary = np.zeros((size, num_modes), dtype=complex)
            primary[free] = pairs.vectors[:, order]
            neff = yee.passive_root(-values)
            fields = {
                name: np.zeros((self.field_shapes[name][0], num_modes), dtype=complex) for name in FIELDS
            }
            # Stored H is -i*eta0*H, so the propagation-derived transverse
            # component carries +i*neff relative to the primary field.
            if polarization == "TE":
                fields["Ey"] = primary
                fields["Hz"] = np.asarray(mu_zz_inv @ (e_to_h @ primary))
                fields["Hx"] = 1j * neff[None, :] * primary / materials["mu_xx"][:, None]
                group = ("Ey", "Hx", "Hz")
            else:
                fields["Hy"] = primary
                fields["Ez"] = np.asarray(eps_zz_inv @ (h_to_e @ primary))
                fields["Ex"] = 1j * neff[None, :] * primary / materials["eps_xx"][:, None]
                group = ("Hy", "Ex", "Ez")
            self._zero_constrained(fields, masks)
            for mode in range(num_modes):
                phase = yee.most_real_phase(*(fields[name][:, mode] for name in group))
                for name in group:
                    fields[name][:, mode] *= phase
            if polarization == "TE":
                field_residuals = equation_residual(
                    (h_to_e @ fields["Hz"])[~pec_yy],
                    (eps_yy @ fields["Ey"])[~pec_yy],
                    (1j * neff * fields["Hx"])[~pec_yy],
                )
            else:
                field_residuals = equation_residual(
                    (e_to_h @ fields["Ez"])[~pmc_yy],
                    (mu_yy @ fields["Hy"])[~pmc_yy],
                    (1j * neff * fields["Ex"])[~pmc_yy],
                )
            solutions.append(
                ModeSolution(polarization, neff, fields, pairs.residuals[order], field_residuals)
            )
        return solutions

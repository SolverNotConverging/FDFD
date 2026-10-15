"""2D full-vector waveguide kernel on an (x, y) Yee cross-section.

The eigenproblem is posed for the transverse electric field (Ex, Ey); the
eigenvalue is ``-neff**2`` with lengths normalized by ``k0``. Transverse H
follows from the Q-block, and Ez, Hz from the longitudinal Maxwell equations.
"""

import numpy as np
from scipy.sparse import bmat, diags

from ..core import eigs, yee
from ..core.constants import EPSILON_0
from ._impedance_2d import compile_impedance_boundary, validate_impedance_pml_separation
from ._kernel import FIELDS, ModeSolution, WaveguideKernel, equation_residual


def _flat_index(i, j, nx):
    return i + j * nx


class WaveguideKernel2D(WaveguideKernel):
    """Cross-section of ``Nx`` by ``Ny`` cells."""

    def __init__(self, frequency, x_length, y_length, nx, ny):
        super().__init__(frequency, (x_length, y_length), (nx, ny))
        self.Nx, self.Ny = self.cells
        self.dx, self.dy = self.steps
        for name in FIELDS:
            setattr(self, "shape_" + name.lower(), self.field_shapes[name])
        self.n_ex, self.n_ey = (int(np.prod(self.field_shapes[n])) for n in ("Ex", "Ey"))
        self.n_hx, self.n_hy = self.n_ey, self.n_ex

    @staticmethod
    def _validate_pml_separation(opaque, pml_cells):
        validate_impedance_pml_separation(opaque, pml_cells)

    def _compile_impedance(self, eps, mu, materials, pec, pmc):
        boundary = compile_impedance_boundary(
            owner=self._surface_impedance_owner,
            definitions=tuple(self._surface_impedance_definitions),
            cell_eps_r_xx=eps[0],
            cell_eps_r_yy=eps[1],
            cell_eps_r_zz=eps[2],
            dx=self.dx,
            dy=self.dy,
            frequency=self.frequency,
            epsilon0=EPSILON_0,
            pml_cells=self._pml_cell_mask,
        )
        occupied = set()
        for row in boundary.rows:
            key = (row.electric_component, row.electric_index)
            if key in occupied:
                raise ValueError(f"Duplicate surface-impedance Ampere row {key}.")
            occupied.add(key)
            if pec[row.electric_component][row.electric_index]:
                raise ValueError(f"Surface-impedance boundary conflicts with a PEC/PMC constraint at {key}.")
            for term in row.magnetic_terms:
                if pmc[term.component][term.index]:
                    raise ValueError(
                        f"Surface-impedance boundary references a PEC/PMC-constrained H{term.index}."
                    )
            materials[("eps_xx", "eps_yy", "eps_zz")[row.electric_component]][row.electric_index] = (
                row.relative_permittivity
            )
            self._restore_normal_permeability(materials, mu, row)
        return boundary

    def _restore_normal_permeability(self, materials, mu, row):
        """Sample interface-normal H permeability only from the retained cell."""
        owner = self._surface_impedance_owner
        i, j = row.electric_index
        if row.electric_component == 0:
            candidates = [(i, jj) for jj in (j - 1, j) if 0 <= jj < self.Ny and owner[i, jj] < 0]
            if len(candidates) != 1:
                raise ValueError(f"Invalid normal-Hy sampling topology at Ex({i}, {j}).")
            materials["mu_yy"][i, j] = mu[1][candidates[0]]
        elif row.electric_component == 1:
            candidates = [(ii, j) for ii in (i - 1, i) if 0 <= ii < self.Nx and owner[ii, j] < 0]
            if len(candidates) != 1:
                raise ValueError(f"Invalid normal-Hx sampling topology at Ey({i}, {j}).")
            materials["mu_xx"][i, j] = mu[0][candidates[0]]

    def derivatives(self):
        """Normalized Yee derivatives keyed ``"ez_ex"`` (Ez -> Ex lattice) etc., with SIBC rows."""
        dx, dy = self.normalized_steps
        s = self.field_shapes
        d = {
            "ez_ex": yee.difference(s["Ez"], s["Ex"], 0, dx),
            "ez_ey": yee.difference(s["Ez"], s["Ey"], 1, dy),
            "ey_hz": yee.difference(s["Ey"], s["Hz"], 0, dx),
            "ex_hz": yee.difference(s["Ex"], s["Hz"], 1, dy),
        }
        d["hy_ez"] = -d["ez_ex"].conj().T
        d["hx_ez"] = -d["ez_ey"].conj().T
        d["hz_hx"] = -d["ey_hz"].conj().T
        d["hz_hy"] = -d["ex_hz"].conj().T
        self._apply_impedance_rows(d)
        return d

    def _apply_impedance_rows(self, d):
        """Replace rectangular Ampere rows by the clipped integral SIBC rows."""
        boundary = self._compiled_impedance_boundary
        if boundary is None or not boundary.rows:
            return
        replacements = {"hy_ez": {}, "hx_ez": {}, "hz_hx": {}, "hz_hy": {}}
        electric = (self.field_shapes["Ex"], self.field_shapes["Ey"], self.field_shapes["Ez"])
        magnetic = (self.field_shapes["Hx"], self.field_shapes["Hy"], self.field_shapes["Hz"])
        for row in boundary.rows:
            component = row.electric_component
            index = _flat_index(*row.electric_index, electric[component][0])
            denominator = row.retained_dual_area * self.k0
            if component == 2:
                replacements["hy_ez"][index] = {}
                replacements["hx_ez"][index] = {}
            for term in row.magnetic_terms:
                coefficient = term.length / denominator
                if component == 0:
                    if term.component != 2:
                        raise ValueError("Ex impedance rows may reference only Hz.")
                    name = "hz_hy"
                elif component == 1:
                    if term.component != 2:
                        raise ValueError("Ey impedance rows may reference only Hz.")
                    name, coefficient = "hz_hx", -coefficient
                elif term.component == 0:
                    name, coefficient = "hx_ez", -coefficient
                elif term.component == 1:
                    name = "hy_ez"
                else:
                    raise ValueError("Ez impedance rows may reference only Hx and Hy.")
                column = _flat_index(*term.index, magnetic[term.component][0])
                values = replacements[name].setdefault(index, {})
                values[column] = values.get(column, 0.0) + coefficient
        for name, rows in replacements.items():
            if not rows:
                continue
            editable = d[name].tolil(copy=True)
            for row, values in rows.items():
                nonzero = sorted((column, value) for column, value in values.items() if value != 0.0)
                editable.rows[row] = [column for column, _ in nonzero]
                editable.data[row] = [value for _, value in nonzero]
            d[name] = editable.tocsr()

    def solve(
        self,
        num_modes,
        sigma,
        *,
        tol=0.0,
        device="cpu",
        method="arpack",
        wide_search=False,
        cutoff_tolerance=1e-8,
    ):
        """Return a one-element list with the vector :class:`ModeSolution`.

        Fields of modes with ``|neff| <= cutoff_tolerance`` cannot be
        reconstructed (the transverse H divides by ``neff``) and are NaN.
        ``wide_search`` enlarges the Krylov space to keep repeated eigenvalues.
        """
        materials, masks = self._effective_materials_and_masks()
        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = masks
        eps_xx, eps_yy = yee.diag(materials["eps_xx"]), yee.diag(materials["eps_yy"])
        mu_xx, mu_yy = yee.diag(materials["mu_xx"]), yee.diag(materials["mu_yy"])
        eps_zz_inv = yee.inverse_diag(materials["eps_zz"], pec_zz)
        mu_zz_inv = yee.inverse_diag(materials["mu_zz"], pmc_zz)
        d = self.derivatives()

        P = bmat(
            [
                [d["ez_ex"] @ eps_zz_inv @ d["hx_ez"], -(d["ez_ex"] @ eps_zz_inv @ d["hy_ez"] + mu_yy)],
                [d["ez_ey"] @ eps_zz_inv @ d["hx_ez"] + mu_xx, -d["ez_ey"] @ eps_zz_inv @ d["hy_ez"]],
            ],
            format="csr",
        )
        Q = bmat(
            [
                [d["hz_hx"] @ mu_zz_inv @ d["ex_hz"], -(d["hz_hx"] @ mu_zz_inv @ d["ey_hz"] + eps_yy)],
                [d["hz_hy"] @ mu_zz_inv @ d["ex_hz"] + eps_xx, -d["hz_hy"] @ mu_zz_inv @ d["ey_hz"]],
            ],
            format="csr",
        )

        free_e = np.concatenate((~pec_xx.ravel(order="F"), ~pec_yy.ravel(order="F")))
        free_h = np.concatenate((~pmc_xx.ravel(order="F"), ~pmc_yy.ravel(order="F")))
        P_reduced, Q_reduced = P[:, free_h], Q[free_h, :]
        omega = (P_reduced @ Q_reduced)[free_e][:, free_e]
        ncv = min(omega.shape[0], max(6 * num_modes + 1, 64)) if wide_search else None
        pairs = eigs.solve(omega, k=num_modes, sigma=sigma, tol=tol, ncv=ncv, device=device, method=method)
        order = np.argsort(np.real(pairs.values))
        values = pairs.values[order]
        exy = np.zeros((self.n_ex + self.n_ey, num_modes), dtype=complex)
        exy[free_e] = pairs.vectors[:, order]
        neff = yee.passive_root(-values)

        # With exp(+i*omega*t - i*beta*z) the reconstruction uses +i*neff. At
        # cutoff it is singular, so keep the eigenpair but leave fields NaN.
        valid = abs(neff) > cutoff_tolerance
        gamma = 1j * neff
        inverse = np.divide(1.0, gamma, out=np.zeros_like(gamma), where=valid)
        hxy_reduced = Q_reduced @ exy @ diags(inverse, format="csr")
        hxy = np.zeros((self.n_hx + self.n_hy, num_modes), dtype=complex)
        hxy[free_h] = hxy_reduced
        field_residuals = np.maximum(
            equation_residual((P_reduced @ hxy_reduced)[free_e], (-1j * neff * exy)[free_e]),
            equation_residual(Q_reduced @ exy, -1j * neff * hxy_reduced),
        )
        ex, ey = exy[: self.n_ex], exy[self.n_ex :]
        hx, hy = hxy[: self.n_hx], hxy[self.n_hx :]
        flat = {
            "Ex": ex,
            "Ey": ey,
            "Hx": hx,
            "Hy": hy,
            "Ez": eps_zz_inv @ (d["hy_ez"] @ hy - d["hx_ez"] @ hx),
            "Hz": mu_zz_inv @ (d["ey_hz"] @ ey - d["ex_hz"] @ ex),
        }
        fields = {name: yee.unflatten(np.asarray(flat[name]), self.field_shapes[name]) for name in FIELDS}
        self._zero_constrained(fields, masks)
        for mode in range(num_modes):
            phase = yee.most_real_phase(*(fields[name][..., mode] for name in FIELDS))
            for name in FIELDS:
                fields[name][..., mode] *= phase
        for name in FIELDS:
            fields[name][..., ~valid] = np.nan
        return [ModeSolution("vector", neff, fields, pairs.residuals[order], field_residuals)]

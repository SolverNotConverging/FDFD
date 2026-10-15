"""2D periodic kernel: an (x, z) cell, periodic in z and invariant in y.

TE has (Hx, Ey) as unknowns and Hz reconstructed; TM has (Ex, Hy) and Ez.
"""

import numpy as np
from scipy.sparse import bmat, csr_matrix

from ..core import yee
from ..core.constants import EPSILON_0, MU_0
from ._kernel import FIELDS, PeriodicKernel


class PeriodicKernel2D(PeriodicKernel):
    """``Nx`` by ``Nz`` cells; ``boundary`` (``"pec"`` or ``"pmc"``) closes the x ends."""

    axes = ("x", "z")

    def __init__(self, frequency, x_length, z_length, nx, nz, *, boundary):
        super().__init__(frequency, (x_length, z_length), (nx, nz))
        self.Nx, self.Nz = self.cells
        self.dx, self.dz = self.steps
        self.boundary = boundary
        s = self.field_shapes
        self.DEX_EZ_TO_EX = yee.difference(s["Ez"], s["Ex"], 0, self.dx)
        self.DHX_HY_TO_EZ = -self.DEX_EZ_TO_EX.conj().T
        self.DEX_EY_TO_HZ = yee.difference(s["Ey"], s["Hz"], 0, self.dx)
        self.DHX_HZ_TO_HX = -self.DEX_EY_TO_HZ.conj().T
        self.DEZ_EX = yee.periodic_difference(s["Ex"], 1, self.dz)
        self.DHZ_HY = -self.DEZ_EX.conj().T
        self.DEZ_EY = yee.periodic_difference(s["Ey"], 1, self.dz)
        self.DHZ_HX = -self.DEZ_EY.conj().T
        self.AZ_EX = yee.periodic_average(s["Ex"], 1)
        self.AZ_HY = self.AZ_EX.conj().T
        self.AZ_EY = yee.periodic_average(s["Ey"], 1)
        self.AZ_HX = self.AZ_EY.conj().T

    def _outer_boundary(self, masks):
        # PEC x ends: tangential E (Ey, Ez) and normal H (Hx) vanish. The
        # unconstrained Yee closure is PMC (tangential H = 0).
        if self.boundary == "pec":
            for name in ("Ey", "Ez", "Hx"):
                masks[name][[0, -1], :] = True

    def _system(self, polarization, materials, masks):
        omega = self.omega
        # Equation rows start as [Faraday(Hy), Ampere(Ex)] for TM or
        # [Ampere(Ey), Faraday(Hx)] for TE; swap them to match the unknowns
        # so the same mask removes constrained rows and columns.
        if polarization == "TM":
            eps_zz_inv = yee.inverse_diag(materials["eps_zz"], masks["Ez"])
            d1 = 1j * omega * MU_0 * yee.diag(materials["mu_yy"])
            d1 = d1 + 1j / omega * self.DEX_EZ_TO_EX @ (1 / EPSILON_0 * eps_zz_inv @ self.DHX_HY_TO_EZ)
            d2 = 1j * omega * EPSILON_0 * yee.diag(materials["eps_xx"])
            zero = csr_matrix((self.sizes["Ex"], self.sizes["Hy"]), dtype=complex)
            A = bmat([[self.DEZ_EX, d1], [d2, self.DHZ_HY]], format="csr")
            B = bmat([[self.AZ_EX, zero], [zero, self.AZ_HY]], format="csr")
            free = np.concatenate((~masks["Ex"].ravel(order="F"), ~masks["Hy"].ravel(order="F")))
        else:
            mu_zz_inv = yee.inverse_diag(materials["mu_zz"], masks["Hz"])
            d1 = -1j * omega * EPSILON_0 * yee.diag(materials["eps_yy"])
            d1 = d1 - 1j / omega * self.DHX_HZ_TO_HX @ (1 / MU_0 * mu_zz_inv @ self.DEX_EY_TO_HZ)
            d2 = -1j * omega * MU_0 * yee.diag(materials["mu_xx"])
            zero = csr_matrix((self.sizes["Hx"], self.sizes["Ey"]), dtype=complex)
            A = bmat([[self.DHZ_HX, d1], [d2, self.DEZ_EY]], format="csr")
            B = bmat([[self.AZ_HX, zero], [zero, self.AZ_EY]], format="csr")
            free = np.concatenate((~masks["Hx"].ravel(order="F"), ~masks["Ey"].ravel(order="F")))
        n = A.shape[0] // 2
        rows = np.r_[np.arange(n, 2 * n), np.arange(n)]
        return A[rows], B[rows], free

    def solve(self, num_modes, sigma, *, polarization, **options):
        """Return ``(gamma, fields, residuals)`` for one polarization, nearest ``sigma`` first."""
        materials, masks = self.materials_and_masks()
        A, B, free = self._system(polarization, materials, masks)
        if options.get("method") == "arpack":
            options["v0"] = lambda n: np.ones(n, dtype=complex)
        gamma, vectors, residuals = self._eigenpairs(A, B, free, num_modes=num_modes, sigma=sigma, **options)
        flat = {name: np.zeros((self.sizes[name], num_modes), dtype=complex) for name in FIELDS}
        if polarization == "TM":
            flat["Ex"], flat["Hy"] = vectors[: self.sizes["Ex"]], vectors[self.sizes["Ex"] :]
            # Longitudinal Ampere law: d_x Hy = i*omega*epsilon*Ez.
            inverse = yee.inverse_diag(materials["eps_zz"], masks["Ez"])
            flat["Ez"] = (-1j / (self.omega * EPSILON_0)) * (inverse @ self.DHX_HY_TO_EZ @ flat["Hy"])
        else:
            flat["Hx"], flat["Ey"] = vectors[: self.sizes["Hx"]], vectors[self.sizes["Hx"] :]
            # Longitudinal Faraday law: d_x Ey = -i*omega*mu*Hz.
            inverse = yee.inverse_diag(materials["mu_zz"], masks["Hz"])
            flat["Hz"] = (1j / (self.omega * MU_0)) * (inverse @ self.DEX_EY_TO_HZ @ flat["Ey"])
        fields = {name: yee.unflatten(np.asarray(flat[name]), self.field_shapes[name]) for name in FIELDS}
        return gamma, fields, residuals

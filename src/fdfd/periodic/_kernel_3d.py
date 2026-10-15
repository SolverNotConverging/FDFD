"""3D periodic kernel: a full-vector (x, y, z) cell, periodic in z.

Unknowns are the transverse fields (Ex, Ey, Hx, Hy); Ez and Hz are
eliminated and reconstructed afterwards. The transverse outer boundary is the
natural Yee closure (PMC); use conductor objects or a PML to change it.
"""

import numpy as np
from scipy.sparse import bmat, csr_matrix

from ..core import yee
from ..core.constants import EPSILON_0, MU_0
from ._kernel import FIELDS, PeriodicKernel


class PeriodicKernel3D(PeriodicKernel):
    """``Nx`` by ``Ny`` by ``Nz`` cells."""

    axes = ("x", "y", "z")

    def __init__(self, frequency, x_length, y_length, z_length, nx, ny, nz):
        super().__init__(frequency, (x_length, y_length, z_length), (nx, ny, nz))
        self.Nx, self.Ny, self.Nz = self.cells
        self.dx, self.dy, self.dz = self.steps
        s = self.field_shapes
        self.DEX_EZ_TO_EX = yee.difference(s["Ez"], s["Ex"], 0, self.dx)
        self.DEY_EZ_TO_EY = yee.difference(s["Ez"], s["Ey"], 1, self.dy)
        self.DEX_EY_TO_HZ = yee.difference(s["Ey"], s["Hz"], 0, self.dx)
        self.DEY_EX_TO_HZ = yee.difference(s["Ex"], s["Hz"], 1, self.dy)
        self.DHX_HY_TO_EZ = -self.DEX_EZ_TO_EX.conj().T
        self.DHY_HX_TO_EZ = -self.DEY_EZ_TO_EY.conj().T
        self.DHX_HZ_TO_HX = -self.DEX_EY_TO_HZ.conj().T
        self.DHY_HZ_TO_HY = -self.DEY_EX_TO_HZ.conj().T
        self.DEZ_EX = yee.periodic_difference(s["Ex"], 2, self.dz)
        self.DEZ_EY = yee.periodic_difference(s["Ey"], 2, self.dz)
        self.DHZ_HX = -self.DEZ_EY.conj().T
        self.DHZ_HY = -self.DEZ_EX.conj().T
        self.AZ_EX = yee.periodic_average(s["Ex"], 2)
        self.AZ_EY = yee.periodic_average(s["Ey"], 2)
        self.AZ_HX = self.AZ_EY.conj().T
        self.AZ_HY = self.AZ_EX.conj().T

    def system(self, materials, masks):
        """The constrained-row-ordered pencil ``(A, B)`` and the free-unknown mask."""
        omega, n = self.omega, self.sizes
        eps_xx, eps_yy = yee.diag(materials["eps_xx"]), yee.diag(materials["eps_yy"])
        mu_xx, mu_yy = yee.diag(materials["mu_xx"]), yee.diag(materials["mu_yy"])
        eps_zz_inv = yee.inverse_diag(materials["eps_zz"], masks["Ez"])
        mu_zz_inv = yee.inverse_diag(materials["mu_zz"], masks["Hz"])
        e = -1j / (omega * EPSILON_0) * eps_zz_inv
        m = 1j / (omega * MU_0) * mu_zz_inv
        zero = csr_matrix
        A = bmat(
            [
                [
                    self.DEZ_EX,
                    zero((n["Ex"], n["Ey"])),
                    self.DEX_EZ_TO_EX @ e @ self.DHY_HX_TO_EZ,
                    -self.DEX_EZ_TO_EX @ e @ self.DHX_HY_TO_EZ + 1j * omega * MU_0 * mu_yy,
                ],
                [
                    zero((n["Ey"], n["Ex"])),
                    self.DEZ_EY,
                    self.DEY_EZ_TO_EY @ e @ self.DHY_HX_TO_EZ - 1j * omega * MU_0 * mu_xx,
                    -self.DEY_EZ_TO_EY @ e @ self.DHX_HY_TO_EZ,
                ],
                [
                    self.DHX_HZ_TO_HX @ m @ self.DEY_EX_TO_HZ,
                    -self.DHX_HZ_TO_HX @ m @ self.DEX_EY_TO_HZ - 1j * omega * EPSILON_0 * eps_yy,
                    self.DHZ_HX,
                    zero((n["Hx"], n["Hy"])),
                ],
                [
                    self.DHY_HZ_TO_HY @ m @ self.DEY_EX_TO_HZ + 1j * omega * EPSILON_0 * eps_xx,
                    -self.DHY_HZ_TO_HY @ m @ self.DEX_EY_TO_HZ,
                    zero((n["Hy"], n["Hx"])),
                    self.DHZ_HY,
                ],
            ],
            format="csr",
        )
        B = bmat(
            [
                [self.AZ_EX, zero((n["Ex"], n["Ey"])), None, None],
                [zero((n["Ey"], n["Ex"])), self.AZ_EY, None, None],
                [None, None, self.AZ_HX, zero((n["Hx"], n["Hy"]))],
                [None, None, zero((n["Hy"], n["Hx"])), self.AZ_HY],
            ],
            format="csr",
        )
        # Equation rows follow [Hy, Hx, Ey, Ex]; unknowns follow [Ex, Ey, Hx, Hy].
        offsets = np.cumsum([0, n["Ex"], n["Ey"], n["Hx"], n["Hy"]])
        rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in (3, 2, 1, 0)])
        free = np.concatenate([~masks[name].ravel(order="F") for name in ("Ex", "Ey", "Hx", "Hy")])
        return A[rows], B[rows], free

    def solve(self, num_modes, sigma, **options):
        """Return ``(gamma, fields, residuals)``, nearest ``sigma`` first."""
        materials, masks = self.materials_and_masks()
        A, B, free = self.system(materials, masks)
        gamma, vectors, residuals = self._eigenpairs(A, B, free, num_modes=num_modes, sigma=sigma, **options)
        offsets = np.cumsum([0, self.sizes["Ex"], self.sizes["Ey"], self.sizes["Hx"], self.sizes["Hy"]])
        flat = {name: vectors[offsets[i] : offsets[i + 1]] for i, name in enumerate(("Ex", "Ey", "Hx", "Hy"))}
        # Restore the longitudinal fields eliminated from the transverse pencil.
        inverse_eps = yee.inverse_diag(materials["eps_zz"], masks["Ez"])
        inverse_mu = yee.inverse_diag(materials["mu_zz"], masks["Hz"])
        flat["Ez"] = (-1j / (self.omega * EPSILON_0)) * (
            inverse_eps @ (self.DHX_HY_TO_EZ @ flat["Hy"] - self.DHY_HX_TO_EZ @ flat["Hx"])
        )
        flat["Hz"] = (1j / (self.omega * MU_0)) * (
            inverse_mu @ (self.DEX_EY_TO_HZ @ flat["Ey"] - self.DEY_EX_TO_HZ @ flat["Ex"])
        )
        fields = {name: yee.unflatten(np.asarray(flat[name]), self.field_shapes[name]) for name in FIELDS}
        return gamma, fields, residuals

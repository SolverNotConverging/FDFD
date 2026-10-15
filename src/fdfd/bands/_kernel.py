"""2D photonic band-structure kernel: frequency is the eigenvalue.

Materials live on a double-resolution helper grid whose even/odd indices are
cell nodes/centres, so each Yee component samples its own location. For a
Bloch vector ``k`` the fields vary as ``exp(-i*k*r)`` times a periodic
envelope, and the eigenvalue is ``(omega/c)**2``. Passive loss gives a
non-negative imaginary frequency under ``exp(+i*omega*t)``.
"""

import numpy as np
from tqdm.auto import tqdm

from ..core import eigs, yee
from ..core.constants import C0


class BandKernel:
    """Rectangular cell of ``Nx`` by ``Ny`` Yee cells, centred on the origin."""

    def __init__(self, x_length, y_length, nx, ny, *, background_epsilon=1.0, background_mu=1.0):
        self.Nx, self.Ny = int(nx), int(ny)
        self.dx, self.dy = x_length / self.Nx, y_length / self.Ny
        self.dx2, self.dy2 = self.dx / 2, self.dy / 2
        # Even helper indices are nodes, odd indices cell centres; periodicity
        # identifies the upper node with the lower one.
        xa = np.arange(2 * self.Nx) * self.dx2 - x_length / 2
        ya = np.arange(2 * self.Ny) * self.dy2 - y_length / 2
        self.X2, self.Y2 = np.meshgrid(xa, ya, indexing="ij")
        self.ER2 = np.full(self.X2.shape, background_epsilon, dtype=complex)
        self.UR2 = np.full(self.X2.shape, background_mu, dtype=complex)

    def apply_material(self, contains, epsilon, mu, subpixels):
        """Blend a scalar material into helper samples by the covered fraction.

        ``contains(x, y)`` is evaluated on ``subpixels``**2 points spread over
        the half-cell around each helper sample (only the sample itself when
        ``subpixels == 1``), using coordinates relative to the cell centre.
        """
        fraction = np.zeros(self.X2.shape)
        offsets = (np.arange(subpixels) + 0.5) / subpixels - 0.5 if subpixels > 1 else np.zeros(1)
        for ox in offsets:
            for oy in offsets:
                fraction += contains(self.X2 + ox * self.dx2, self.Y2 + oy * self.dy2)
        fraction /= len(offsets) ** 2
        self.ER2 = self.ER2 * (1 - fraction) + epsilon * fraction
        self.UR2 = self.UR2 * (1 - fraction) + mu * fraction

    def tensors(self):
        """Relative material values sampled at the six Yee component locations."""
        return {
            "eps_xx": self.ER2[1::2, ::2],
            "eps_yy": self.ER2[::2, 1::2],
            "eps_zz": self.ER2[::2, ::2],
            "mu_xx": self.UR2[::2, 1::2],
            "mu_yy": self.UR2[1::2, ::2],
            "mu_zz": self.UR2[1::2, 1::2],
        }

    def solve(self, beta_path, num_bands, *, polarizations, sigma, tol, device, method, progress=True):
        """Return ``(frequencies, residuals)``, each a dict of ``(num_bands, points)`` arrays."""
        t = {name: yee.diag(values) for name, values in self.tensors().items()}
        inverse = {name: matrix.power(-1) for name, matrix in t.items()}
        points = beta_path.shape[1]
        frequencies = {pol: np.zeros((num_bands, points), dtype=complex) for pol in polarizations}
        residuals = {pol: np.zeros((num_bands, points)) for pol in polarizations}
        for index in tqdm(range(points), desc="Band structure", unit="k-point", disable=not progress):
            kx, ky = beta_path[:, index]
            dex = yee.bloch_difference((self.Nx, self.Ny), 0, self.dx, kx)
            dey = yee.bloch_difference((self.Nx, self.Ny), 1, self.dy, ky)
            dhx, dhy = -dex.conj().T, -dey.conj().T
            operators = {
                "TM": inverse["eps_zz"] @ (-dhx @ inverse["mu_yy"] @ dex - dhy @ inverse["mu_xx"] @ dey),
                "TE": inverse["mu_zz"] @ (-dex @ inverse["eps_yy"] @ dhx - dey @ inverse["eps_xx"] @ dhy),
            }
            for pol in polarizations:
                pairs = eigs.solve(
                    operators[pol], k=num_bands, sigma=sigma, tol=tol, device=device, method=method
                )
                order = np.argsort(np.real(pairs.values))[:num_bands]
                # Principal root: passive loss keeps Im(omega) >= 0.
                frequencies[pol][:, index] = C0 * np.sqrt(pairs.values[order]) / (2 * np.pi)
                residuals[pol][:, index] = pairs.residuals[order]
        return frequencies, residuals

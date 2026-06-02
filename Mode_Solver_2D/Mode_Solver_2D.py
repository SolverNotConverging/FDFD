import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.sparse import bmat, diags, eye, kron
from scipy.sparse.linalg import eigs


class ModeSolver2D:
    """2D vector FDFD mode solver on an (Nx, Ny) Yee-style grid."""

    def __init__(self, frequency, x_range, y_range, Nx, Ny, num_modes, mode_filter=True):
        self.frequency = frequency
        self.x_range = x_range
        self.y_range = y_range
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.dx = x_range / self.Nx
        self.dy = y_range / self.Ny
        self.epsilon0 = 8.85e-12
        self.mu0 = 1.26e-6
        self.c = 1 / np.sqrt(self.epsilon0 * self.mu0)
        self.k_0 = 2 * np.pi * frequency / self.c
        self.dx_normalized = self.k_0 * self.dx
        self.dy_normalized = self.k_0 * self.dy

        shape = (self.Nx, self.Ny)
        self.eps_r_xx = np.ones(shape, dtype=complex)
        self.eps_r_yy = np.ones(shape, dtype=complex)
        self.eps_r_zz = np.ones(shape, dtype=complex)
        self.mu_r_xx = np.ones(shape, dtype=complex)
        self.mu_r_yy = np.ones(shape, dtype=complex)
        self.mu_r_zz = np.ones(shape, dtype=complex)

        self.pec_xx_mask = np.zeros(shape, dtype=bool)
        self.pec_yy_mask = np.zeros(shape, dtype=bool)
        self.pec_zz_mask = np.zeros(shape, dtype=bool)
        self.pmc_xx_mask = np.zeros(shape, dtype=bool)
        self.pmc_yy_mask = np.zeros(shape, dtype=bool)
        self.pmc_zz_mask = np.zeros(shape, dtype=bool)

        self.num_modes = int(num_modes)
        self.mode_filter = bool(mode_filter)
        self.eigenvalues = None
        self.eigenvectors = None
        self.neff = None
        self.propagation_constant = None
        self.attenuation_constant = None
        self.spurious_scores = None
        self.accepted_candidate_indices = None
        self.rejected_candidate_indices = None
        self.unselected_candidate_indices = None

    def _invalidate_solution(self):
        self.eigenvalues = None
        self.eigenvectors = None
        self.neff = None
        self.propagation_constant = None
        self.attenuation_constant = None
        self.spurious_scores = None
        self.accepted_candidate_indices = None
        self.rejected_candidate_indices = None
        self.unselected_candidate_indices = None
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            if hasattr(self, name):
                delattr(self, name)

    def _normalise_three(self, name, value):
        if np.isscalar(value):
            return np.full(3, value, dtype=complex)
        array = np.asarray(value, dtype=complex)
        if array.ndim == 1 and array.size == 3:
            return array
        raise ValueError(f"{name} must be a scalar or a length-3 1D array (xx, yy, zz).")

    def _validate_components(self, components):
        if isinstance(components, str):
            components = (components,)
        components = tuple(components)
        invalid = set(components) - {"xx", "yy", "zz"}
        if invalid:
            raise ValueError(f"components contains invalid tensor component(s): {sorted(invalid)}.")
        return components

    def _bound_to_index(self, value, axis):
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            step = self.dx if axis == "x" else self.dy
            return int(round(float(value) / step))
        raise ValueError("Region bounds must be int grid indices or float physical positions in metres.")

    def _region_slices(self, x_range, y_range):
        try:
            x0 = self._bound_to_index(x_range[0], "x")
            x1 = self._bound_to_index(x_range[1], "x")
            y0 = self._bound_to_index(y_range[0], "y")
            y1 = self._bound_to_index(y_range[1], "y")
        except (TypeError, IndexError):
            raise ValueError("x_range and y_range must be (min, max) pairs.")

        if not (x1 > x0 and y1 > y0):
            raise ValueError("x_range and y_range must satisfy max > min.")
        if not (0 <= x0 < x1 <= self.Nx and 0 <= y0 < y1 <= self.Ny):
            raise ValueError("Region is out of bounds of the simulation grid.")
        return slice(x0, x1), slice(y0, y1)

    def _component_array(self, prefix, component):
        if prefix == "eps":
            if component == "xx":
                return self.eps_r_xx
            if component == "yy":
                return self.eps_r_yy
            if component == "zz":
                return self.eps_r_zz
        if prefix == "mu":
            if component == "xx":
                return self.mu_r_xx
            if component == "yy":
                return self.mu_r_yy
            if component == "zz":
                return self.mu_r_zz
        if prefix == "pec":
            if component == "xx":
                return self.pec_xx_mask
            if component == "yy":
                return self.pec_yy_mask
            if component == "zz":
                return self.pec_zz_mask
        if prefix == "pmc":
            if component == "xx":
                return self.pmc_xx_mask
            if component == "yy":
                return self.pmc_yy_mask
            if component == "zz":
                return self.pmc_zz_mask
        raise ValueError(f"Unknown {prefix} component {component!r}.")

    def add_object(self, epsilon, mu, x_range, y_range):
        """Add a rectangular isotropic or diagonal-anisotropic material region."""
        sl_x, sl_y = self._region_slices(x_range, y_range)
        epsilon = self._normalise_three("epsilon", epsilon)
        mu = self._normalise_three("mu", mu)

        self.eps_r_xx[sl_x, sl_y] = epsilon[0]
        self.eps_r_yy[sl_x, sl_y] = epsilon[1]
        self.eps_r_zz[sl_x, sl_y] = epsilon[2]
        self.mu_r_xx[sl_x, sl_y] = mu[0]
        self.mu_r_yy[sl_x, sl_y] = mu[1]
        self.mu_r_zz[sl_x, sl_y] = mu[2]
        self._invalidate_solution()

    def add_pec(self, x_range, y_range, components=None):
        """Add a PEC region.

        By default the region is interpreted as a cell-centered PEC object and
        converted to component-specific Yee masks, matching the gprMax solver.
        Pass components=(...) to constrain selected electric components directly.
        """
        sl_x, sl_y = self._region_slices(x_range, y_range)
        cell_mask = np.zeros((self.Nx, self.Ny), dtype=bool)
        cell_mask[sl_x, sl_y] = True
        if components is None:
            xx_mask, yy_mask, zz_mask = self.component_masks_from_cell_mask(cell_mask)
            self.pec_xx_mask |= xx_mask
            self.pec_yy_mask |= yy_mask
            self.pec_zz_mask |= zz_mask
        else:
            for comp in self._validate_components(components):
                self._component_array("pec", comp)[sl_x, sl_y] = True
        self._invalidate_solution()

    def add_pmc(self, x_range, y_range, components=None):
        """Add a PMC region.

        By default the region is interpreted as a cell-centered PMC object and
        converted to component-specific Yee masks. Pass components=(...) to
        constrain selected magnetic components directly.
        """
        sl_x, sl_y = self._region_slices(x_range, y_range)
        cell_mask = np.zeros((self.Nx, self.Ny), dtype=bool)
        cell_mask[sl_x, sl_y] = True
        if components is None:
            xx_mask, yy_mask, zz_mask = self.component_masks_from_cell_mask(cell_mask)
            self.pmc_xx_mask |= xx_mask
            self.pmc_yy_mask |= yy_mask
            self.pmc_zz_mask |= zz_mask
        else:
            for comp in self._validate_components(components):
                self._component_array("pmc", comp)[sl_x, sl_y] = True
        self._invalidate_solution()

    @staticmethod
    def component_masks_from_cell_mask(cell_mask):
        mask = np.asarray(cell_mask, dtype=bool)

        xx_mask = mask.copy()
        xx_mask[:, 1:] |= mask[:, :-1] & ~mask[:, 1:]
        xx_mask[:, :-1] |= mask[:, 1:] & ~mask[:, :-1]

        yy_mask = mask.copy()
        yy_mask[1:, :] |= mask[:-1, :] & ~mask[1:, :]
        yy_mask[:-1, :] |= mask[1:, :] & ~mask[:-1, :]

        zz_mask = xx_mask | yy_mask
        return xx_mask, yy_mask, zz_mask

    def add_pml(self, pml_width=50, n=3, sigma_max=5, direction="both"):
        """Add a simple uniaxial PML by stretching epsilon and mu tensors."""
        pml_width = int(pml_width)
        if pml_width <= 0:
            raise ValueError("pml_width must be positive.")
        if direction not in ("x-", "x+", "x", "y-", "y+", "y", "both"):
            raise ValueError("direction must be one of 'x-', 'x+', 'x', 'y-', 'y+', 'y', or 'both'.")

        sigma_x = np.zeros((self.Nx, self.Ny), dtype=float)
        sigma_y = np.zeros((self.Nx, self.Ny), dtype=float)

        if direction in ("x-", "x", "both"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[i, :] = sigma_max * ((pml_width - i) / pml_width) ** n
        if direction in ("x+", "x", "both"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[-i - 1, :] = sigma_max * ((pml_width - i) / pml_width) ** n
        if direction in ("y-", "y", "both"):
            for i in range(min(pml_width, self.Ny)):
                sigma_y[:, i] = sigma_max * ((pml_width - i) / pml_width) ** n
        if direction in ("y+", "y", "both"):
            for i in range(min(pml_width, self.Ny)):
                sigma_y[:, -i - 1] = sigma_max * ((pml_width - i) / pml_width) ** n

        eps0 = 8.854187817e-12
        omega = 2 * np.pi * self.frequency
        Sx = 1.0 + 1j * sigma_x / (eps0 * omega)
        Sy = 1.0 + 1j * sigma_y / (eps0 * omega)

        self.eps_r_xx *= Sy / Sx
        self.eps_r_yy *= Sx / Sy
        self.eps_r_zz *= Sx * Sy
        self.mu_r_xx *= Sy / Sx
        self.mu_r_yy *= Sx / Sy
        self.mu_r_zz *= Sx * Sy
        self._invalidate_solution()

    def add_UPML(self, pml_width=50, n=3, sigma_max=5, direction="both"):
        """Backward-compatible alias for add_pml()."""
        self.add_pml(pml_width=pml_width, n=n, sigma_max=sigma_max, direction=direction)

    def add_impedance_surface(
            self,
            Zs: complex,
            position: float | int,
            *,
            orientation: str = "x",
            thickness_cells: int = 1,
            eps_components=("xx", "yy", "zz"),
    ):
        if orientation not in ("x", "y"):
            raise ValueError("orientation must be 'x' or 'y'.")
        thickness_cells = int(thickness_cells)
        if thickness_cells <= 0:
            raise ValueError("thickness_cells must be positive.")

        if orientation == "x":
            idx = self._bound_to_index(position, "x")
            x_range = (idx, idx + thickness_cells)
            y_range = (0, self.Ny)
            thickness = thickness_cells * self.dx
        else:
            idy = self._bound_to_index(position, "y")
            x_range = (0, self.Nx)
            y_range = (idy, idy + thickness_cells)
            thickness = thickness_cells * self.dy

        sl_x, sl_y = self._region_slices(x_range, y_range)
        eps0 = 8.854187817e-12
        delta_eps = -1j / (2 * np.pi * self.frequency * eps0 * thickness * Zs)
        for comp in self._validate_components(eps_components):
            self._component_array("eps", comp)[sl_x, sl_y] += delta_eps
        self._invalidate_solution()

    def _yeeder2d(self):
        """Generate derivative matrices on a 2D Yee grid."""
        def diff_operator(n):
            values = np.ones(n)
            return diags([-values, values], [0, 1], shape=(n, n)).tocsr()

        Ix = eye(self.Nx, format="csr")
        Iy = eye(self.Ny, format="csr")
        self.DEX = kron(Iy, diff_operator(self.Nx), format="csr") / self.dx_normalized
        self.DEY = kron(diff_operator(self.Ny), Ix, format="csr") / self.dy_normalized
        self.DHX = -self.DEX.conj().T
        self.DHY = -self.DEY.conj().T
        return self.DEX, self.DEY, self.DHX, self.DHY

    def _effective_materials_and_masks(self):
        eps_r_xx = self.eps_r_xx.copy()
        eps_r_yy = self.eps_r_yy.copy()
        eps_r_zz = self.eps_r_zz.copy()
        mu_r_xx = self.mu_r_xx.copy()
        mu_r_yy = self.mu_r_yy.copy()
        mu_r_zz = self.mu_r_zz.copy()

        pec_xx_mask = self.pec_xx_mask | ~np.isfinite(eps_r_xx)
        pec_yy_mask = self.pec_yy_mask | ~np.isfinite(eps_r_yy)
        pec_zz_mask = self.pec_zz_mask | ~np.isfinite(eps_r_zz)
        pmc_xx_mask = self.pmc_xx_mask | ~np.isfinite(mu_r_xx)
        pmc_yy_mask = self.pmc_yy_mask | ~np.isfinite(mu_r_yy)
        pmc_zz_mask = self.pmc_zz_mask | ~np.isfinite(mu_r_zz)

        eps_r_xx[pec_xx_mask] = 1.0 + 0j
        eps_r_yy[pec_yy_mask] = 1.0 + 0j
        eps_r_zz[pec_zz_mask] = 1.0 + 0j
        mu_r_xx[pmc_xx_mask] = 1.0 + 0j
        mu_r_yy[pmc_yy_mask] = 1.0 + 0j
        mu_r_zz[pmc_zz_mask] = 1.0 + 0j

        return (
            eps_r_xx,
            eps_r_yy,
            eps_r_zz,
            mu_r_xx,
            mu_r_yy,
            mu_r_zz,
            pec_xx_mask,
            pec_yy_mask,
            pec_zz_mask,
            pmc_xx_mask,
            pmc_yy_mask,
            pmc_zz_mask,
        )

    def _inverse_diag_on_free(self, values, constrained_mask):
        flat = values.ravel(order="F")
        constrained = constrained_mask.ravel(order="F")
        inverse = np.zeros_like(flat, dtype=complex)
        inverse[~constrained] = 1.0 / flat[~constrained]
        return diags(inverse)

    def _zero_constrained_fields(self, pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask):
        self.Ex[pec_xx_mask.ravel(order="F"), :] = 0.0
        self.Ey[pec_yy_mask.ravel(order="F"), :] = 0.0
        self.Ez[pec_zz_mask.ravel(order="F"), :] = 0.0
        self.Hx[pmc_xx_mask.ravel(order="F"), :] = 0.0
        self.Hy[pmc_yy_mask.ravel(order="F"), :] = 0.0
        self.Hz[pmc_zz_mask.ravel(order="F"), :] = 0.0

    def solve(self, sigma=-13, extra_modes=8, max_pec_neighbor_energy_fraction=0.35):
        """Solve for transverse modes and recover all six field components."""
        (
            eps_r_xx,
            eps_r_yy,
            eps_r_zz,
            mu_r_xx,
            mu_r_yy,
            mu_r_zz,
            pec_xx_mask,
            pec_yy_mask,
            pec_zz_mask,
            pmc_xx_mask,
            pmc_yy_mask,
            pmc_zz_mask,
        ) = self._effective_materials_and_masks()

        eps_r_xx_diag = diags(eps_r_xx.ravel(order="F"))
        eps_r_yy_diag = diags(eps_r_yy.ravel(order="F"))
        mu_r_xx_diag = diags(mu_r_xx.ravel(order="F"))
        mu_r_yy_diag = diags(mu_r_yy.ravel(order="F"))

        DEX, DEY, DHX, DHY = self._yeeder2d()
        eps_r_zz_inv = self._inverse_diag_on_free(eps_r_zz, pec_zz_mask)
        mu_r_zz_inv = self._inverse_diag_on_free(mu_r_zz, pmc_zz_mask)

        P11 = DEX @ eps_r_zz_inv @ DHY
        P12 = -(DEX @ eps_r_zz_inv @ DHX + mu_r_yy_diag)
        P21 = DEY @ eps_r_zz_inv @ DHY + mu_r_xx_diag
        P22 = -DEY @ eps_r_zz_inv @ DHX
        P = bmat([[P11, P12], [P21, P22]], format="csr")

        Q11 = DHX @ mu_r_zz_inv @ DEY
        Q12 = -(DHX @ mu_r_zz_inv @ DEX + eps_r_yy_diag)
        Q21 = DHY @ mu_r_zz_inv @ DEY + eps_r_xx_diag
        Q22 = -DHY @ mu_r_zz_inv @ DEX
        Q = bmat([[Q11, Q12], [Q21, Q22]], format="csr")

        free_ex = ~pec_xx_mask.ravel(order="F")
        free_ey = ~pec_yy_mask.ravel(order="F")
        free_hx = ~pmc_xx_mask.ravel(order="F")
        free_hy = ~pmc_yy_mask.ravel(order="F")
        free_exy = np.concatenate((free_ex, free_ey))
        free_hxy = np.concatenate((free_hx, free_hy))

        P_reduced = P[:, free_hxy]
        Q_reduced = Q[free_hxy, :]
        Omega = P_reduced @ Q_reduced
        Omega = Omega[free_exy, :][:, free_exy]
        if Omega.shape[0] <= self.num_modes:
            raise ValueError(
                f"Not enough unconstrained electric DOFs ({Omega.shape[0]}) to solve {self.num_modes} modes."
            )

        candidate_modes = self.num_modes + max(0, int(extra_modes)) if self.mode_filter else self.num_modes
        candidate_modes = min(candidate_modes, Omega.shape[0] - 1)
        if candidate_modes < self.num_modes:
            raise ValueError(
                f"Not enough unconstrained electric DOFs ({Omega.shape[0]}) to solve {self.num_modes} modes."
            )

        eigenvalues, eigenvectors_reduced = eigs(Omega, k=candidate_modes, sigma=sigma)
        eigenvectors = np.zeros((2 * self.Nx * self.Ny, candidate_modes), dtype=complex)
        eigenvectors[free_exy, :] = eigenvectors_reduced

        order = np.argsort(np.real(eigenvalues))
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        if self.mode_filter:
            keep_indices = self._select_physical_candidates(
                eigenvectors,
                self.num_modes,
                max_pec_neighbor_energy_fraction,
                pec_xx_mask,
                pec_yy_mask,
            )
        else:
            keep_indices = np.arange(self.num_modes)
            self.spurious_scores = np.zeros(candidate_modes, dtype=np.float64)
            self.accepted_candidate_indices = keep_indices.copy()
            self.rejected_candidate_indices = np.array([], dtype=int)
            self.unselected_candidate_indices = np.array([], dtype=int)

        self.eigenvalues = eigenvalues[keep_indices]
        Exy = eigenvectors[:, keep_indices]
        self.eigenvectors = Exy
        self.neff = self._passive_positive_neff(-self.eigenvalues)
        self.propagation_constant = np.real(self.neff)
        self.attenuation_constant = np.imag(self.neff)

        eigenvalues_inv = diags(np.sqrt(self.eigenvalues)).power(-1)
        self.Ex = np.asarray(Exy[: self.Nx * self.Ny, :], dtype=complex)
        self.Ey = np.asarray(Exy[self.Nx * self.Ny:, :], dtype=complex)

        Hxy_reduced = Q_reduced @ Exy @ eigenvalues_inv
        Hxy = np.zeros((2 * self.Nx * self.Ny, self.num_modes), dtype=complex)
        Hxy[free_hxy, :] = Hxy_reduced
        self.Hx = np.asarray(Hxy[: self.Nx * self.Ny, :], dtype=complex)
        self.Hy = np.asarray(Hxy[self.Nx * self.Ny:, :], dtype=complex)
        self.Ez = np.asarray(eps_r_zz_inv @ (DHX @ self.Hy - DHY @ self.Hx), dtype=complex)
        self.Hz = np.asarray(mu_r_zz_inv @ (DEX @ self.Ey - DEY @ self.Ex), dtype=complex)
        self._zero_constrained_fields(pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask)

    def _select_physical_candidates(self, eigenvectors, num_modes, max_pec_neighbor_energy_fraction, pec_xx_mask, pec_yy_mask):
        scores = self._pec_neighbor_energy_scores(eigenvectors, pec_xx_mask, pec_yy_mask)
        candidate_indices = np.arange(eigenvectors.shape[1])
        accepted = candidate_indices[scores <= max_pec_neighbor_energy_fraction]
        rejected = candidate_indices[scores > max_pec_neighbor_energy_fraction]
        if accepted.size < num_modes:
            rejected_by_score = rejected[np.argsort(scores[rejected])]
            accepted = np.concatenate((accepted, rejected_by_score[:num_modes - accepted.size]))
        keep_indices = np.sort(accepted[:num_modes])
        self.spurious_scores = scores
        self.accepted_candidate_indices = keep_indices.copy()
        self.rejected_candidate_indices = rejected.copy()
        self.unselected_candidate_indices = np.setdiff1d(candidate_indices, keep_indices, assume_unique=True)
        return keep_indices

    def _pec_neighbor_energy_scores(self, eigenvectors, pec_xx_mask, pec_yy_mask):
        ex = eigenvectors[: self.Nx * self.Ny, :]
        ey = eigenvectors[self.Nx * self.Ny:, :]
        total_energy = np.sum(np.abs(ex) ** 2, axis=0) + np.sum(np.abs(ey) ** 2, axis=0)
        near_xx = self._dilate_mask(pec_xx_mask).ravel(order="F")
        near_yy = self._dilate_mask(pec_yy_mask).ravel(order="F")
        pec_neighbor_energy = np.sum(np.abs(ex[near_xx, :]) ** 2, axis=0)
        pec_neighbor_energy += np.sum(np.abs(ey[near_yy, :]) ** 2, axis=0)
        scores = np.zeros(eigenvectors.shape[1], dtype=np.float64)
        valid = total_energy > 1e-300
        scores[valid] = np.real(pec_neighbor_energy[valid] / total_energy[valid])
        scores[~valid] = np.inf
        return scores

    @staticmethod
    def _dilate_mask(mask):
        mask = np.asarray(mask, dtype=bool)
        dilated = mask.copy()
        dilated[1:, :] |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
        dilated[:, 1:] |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
        return dilated

    def _has_lossy_material(self):
        for values in (
                self.eps_r_xx,
                self.eps_r_yy,
                self.eps_r_zz,
                self.mu_r_xx,
                self.mu_r_yy,
                self.mu_r_zz,
        ):
            finite = np.isfinite(values)
            if np.any(np.abs(np.imag(values[finite])) > 1e-14):
                return True
        return False

    def _passive_positive_neff(self, neff_squared):
        sqrt = np.sqrt(neff_squared)
        neff = np.where(np.real(sqrt) < 0, -sqrt, sqrt)
        real = np.real(neff)
        imag = np.imag(neff)
        tolerance = 1e-12 * np.maximum(1.0, np.abs(neff))
        real = np.where(np.abs(real) <= tolerance, 0.0, real)
        imag = np.where(np.abs(imag) <= tolerance, 0.0, np.abs(imag))
        if not self._has_lossy_material():
            imag = np.zeros_like(imag)
        return real + 1j * imag

    def _field_array(self, field, mode):
        return field[:, mode].reshape((self.Nx, self.Ny), order="F")

    def visualize(self, mode=1, **kwargs):
        """Visualize selected field components for a given one-based mode index."""
        if self.eigenvalues is None:
            raise RuntimeError("solve() must be called before visualize().")
        mode -= 1
        if not (0 <= mode < self.num_modes):
            raise ValueError("mode is out of range.")

        fields = {
            "ex": (self._field_array(self.Ex, mode), "Ex"),
            "ey": (self._field_array(self.Ey, mode), "Ey"),
            "ez": (self._field_array(self.Ez, mode), "Ez"),
            "hx": (self._field_array(self.Hx, mode), "Hx"),
            "hy": (self._field_array(self.Hy, mode), "Hy"),
            "hz": (self._field_array(self.Hz, mode), "Hz"),
        }
        e_abs = np.sqrt(sum(np.abs(fields[key][0]) ** 2 for key in ("ex", "ey", "ez")))
        h_abs = np.sqrt(sum(np.abs(fields[key][0]) ** 2 for key in ("hx", "hy", "hz")))
        fields["eabs"] = (e_abs, "|E|")
        fields["habs"] = (h_abs, "|H|")

        selected = [key for key in fields if kwargs.get(key)]
        if not selected:
            selected = ["ex", "ey", "ez", "hx", "hy", "hz"]

        ncols = 3
        nrows = int(np.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), layout="compressed")
        axes = np.array(axes).reshape(-1)

        last_image = None
        for i, field_name in enumerate(selected):
            field_data, title = fields[field_name]
            norm = np.max(np.abs(field_data))
            if norm > 0:
                field_data = field_data / norm
            ax = axes[i]
            last_image = ax.imshow(
                np.abs(field_data).T,
                cmap="viridis",
                origin="lower",
                extent=[0, self.x_range * 1e3, 0, self.y_range * 1e3],
            )
            ax.imshow(
                np.abs(self.eps_r_xx).T,
                cmap="inferno",
                origin="lower",
                extent=[0, self.x_range * 1e3, 0, self.y_range * 1e3],
                vmax=20,
                alpha=0.2,
            )
            ax.set_title(title)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")

        for j in range(len(selected), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(
            rf"Mode {mode + 1}: $n_{{eff}}$ = {self.neff[mode]:.4g}",
            fontsize=16,
        )
        if last_image is not None:
            fig.colorbar(last_image, ax=axes[:len(selected)], location="right", shrink=1, pad=0.02)
        plt.show()

    def visualize_with_gui(self):
        """Visualize field components with a dropdown menu for mode selection."""
        if self.eigenvalues is None:
            raise RuntimeError("solve() must be called before visualize_with_gui().")

        root = tk.Tk()
        root.title("Field Visualization")

        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        controls_frame = tk.Frame(root)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=10)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        colorbar = [None]

        def plot_mode(selected_mode):
            mode = int(selected_mode) - 1
            data = [
                self._field_array(self.Ex, mode),
                self._field_array(self.Ey, mode),
                self._field_array(self.Ez, mode),
                self._field_array(self.Hx, mode),
                self._field_array(self.Hy, mode),
                self._field_array(self.Hz, mode),
            ]
            titles = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
            e_norm = max(np.max(np.abs(field)) for field in data[:3])
            h_norm = max(np.max(np.abs(field)) for field in data[3:])

            for ax in axes.flat:
                ax.clear()
            if colorbar[0] is not None:
                colorbar[0].remove()
                colorbar[0] = None

            for i, ax in enumerate(axes.flat):
                norm = e_norm if i < 3 else h_norm
                field = data[i] / norm if norm > 0 else data[i]
                ax.imshow(
                    np.abs(field).T,
                    cmap="viridis",
                    origin="lower",
                    extent=[0, self.x_range * 1e3, 0, self.y_range * 1e3],
                    vmin=0,
                    vmax=1,
                )
                ax.imshow(
                    np.abs(self.eps_r_zz).T,
                    cmap="inferno",
                    origin="lower",
                    extent=[0, self.x_range * 1e3, 0, self.y_range * 1e3],
                    vmax=20,
                    alpha=0.2,
                )
                ax.set_title(titles[i])
                ax.set_xlabel("x (mm)")
                ax.set_ylabel("y (mm)")

            fig.subplots_adjust(right=0.86)
            cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
            colorbar[0] = fig.colorbar(sm, cax=cbar_ax)
            colorbar[0].set_label("Normalized Magnitude")
            fig.suptitle(rf"Mode {mode + 1}: $n_{{eff}}$ = {self.neff[mode]:.4g}", fontsize=16)
            canvas.draw()

        mode_var = tk.StringVar(value="1")
        mode_menu = ttk.Combobox(controls_frame, textvariable=mode_var, values=list(range(1, self.num_modes + 1)))
        mode_menu.grid(row=0, column=0, padx=10, sticky="w")
        mode_menu.bind("<<ComboboxSelected>>", lambda event: plot_mode(mode_var.get()))
        tk.Button(controls_frame, text="Quit", command=root.destroy).grid(row=0, column=1, padx=10, sticky="e")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        plot_mode(mode_var.get())
        root.mainloop()




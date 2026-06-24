import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigs


class PeriodicModeSolver2D:
    """2D Bloch-periodic TE/TM mode solver on a periodic Yee grid."""

    def __init__(
            self,
            polarization,
            freq,
            x_range,
            z_range,
            Nx,
            Nz,
            num_modes,
            mode_filter=True,
            guess=0,
            tol=0,
            ncv=None,
    ):
        polarization = str(polarization).upper()
        if polarization not in ("TE", "TM"):
            raise ValueError("polarization must be 'TE' or 'TM'.")

        self.polarization = polarization
        self.freq = freq
        self.frequency = freq
        self.x_range = x_range
        self.z_range = z_range
        self.Nx = int(Nx)
        self.Nz = int(Nz)
        self.dx = x_range / self.Nx
        self.dz = z_range / self.Nz
        self.N = self.Nx * self.Nz
        self.num_modes = int(num_modes)
        self.mode_filter = bool(mode_filter)

        self.epsilon0 = 8.85e-12
        self.mu0 = 1.26e-6
        self.c = 1 / np.sqrt(self.epsilon0 * self.mu0)
        self.omega = 2 * np.pi * freq
        self.k0 = self.omega / self.c

        self.guess = guess
        self.tol = tol
        self.ncv = ncv

        self.shape_cell = (self.Nx, self.Nz)
        self.shape_ex = self.shape_cell
        self.shape_hy = self.shape_ex
        self.shape_ez = (self.Nx + 1, self.Nz)
        self.shape_hx = self.shape_ez
        self.shape_ey = self.shape_hx
        self.shape_hz = self.shape_cell

        self.n_ex = int(np.prod(self.shape_ex))
        self.n_hy = self.n_ex
        self.n_ez = int(np.prod(self.shape_ez))
        self.n_hx = self.n_ez
        self.n_ey = self.n_hx
        self.n_hz = self.N

        shape = self.shape_cell
        self.cell_eps_r_xx = np.ones(shape, dtype=complex)
        self.cell_eps_r_yy = np.ones(shape, dtype=complex)
        self.cell_eps_r_zz = np.ones(shape, dtype=complex)
        self.cell_mu_r_xx = np.ones(shape, dtype=complex)
        self.cell_mu_r_yy = np.ones(shape, dtype=complex)
        self.cell_mu_r_zz = np.ones(shape, dtype=complex)
        self.material_no_average_mask = np.zeros(shape, dtype=bool)

        self.eps_r_xx = np.ones(self.shape_ex, dtype=complex)
        self.eps_r_yy = np.ones(self.shape_ey, dtype=complex)
        self.eps_r_zz = np.ones(self.shape_ez, dtype=complex)
        self.mu_r_xx = np.ones(self.shape_hx, dtype=complex)
        self.mu_r_yy = np.ones(self.shape_hy, dtype=complex)
        self.mu_r_zz = np.ones(self.shape_hz, dtype=complex)

        self.pec_xx_mask = np.zeros(self.shape_ex, dtype=bool)
        self.pec_yy_mask = np.zeros(self.shape_ey, dtype=bool)
        self.pec_zz_mask = np.zeros(self.shape_ez, dtype=bool)
        self.pmc_xx_mask = np.zeros(self.shape_hx, dtype=bool)
        self.pmc_yy_mask = np.zeros(self.shape_hy, dtype=bool)
        self.pmc_zz_mask = np.zeros(self.shape_hz, dtype=bool)
        self._pec_regions = []
        self._pmc_regions = []

        self._init_operators()
        self.update_component_materials()
        self._invalidate_solution()

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
        for name in ("Ex", "Ey", "Hx", "Hy"):
            if hasattr(self, name):
                delattr(self, name)

    @staticmethod
    def _normalise_three(name, value):
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
            step = self.dx if axis == "x" else self.dz
            return int(round(float(value) / step))
        raise ValueError("Region bounds must be int grid indices or float physical positions in metres.")

    def _coordinate_to_length(self, value, axis):
        if isinstance(value, (int, np.integer)):
            step = self.dx if axis == "x" else self.dz
            return int(value) * step
        if isinstance(value, (float, np.floating)):
            return float(value)
        raise ValueError("Coordinates must be int grid indices or float physical positions in metres.")

    def _range_to_lengths(self, name, values, axis):
        try:
            start, stop = values
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a (min, max) pair.")
        start = self._coordinate_to_length(start, axis)
        stop = self._coordinate_to_length(stop, axis)
        if stop <= start:
            raise ValueError(f"{name} must satisfy max > min.")
        limit = self.x_range if axis == "x" else self.z_range
        if start < 0 or stop > limit:
            raise ValueError(f"{name} is out of bounds of the simulation grid.")
        return start, stop

    @staticmethod
    def _validate_subpixels(subpixels):
        subpixels = int(subpixels)
        if subpixels <= 0:
            raise ValueError("subpixels must be positive.")
        return subpixels

    def _region_slices(self, x_range, z_range):
        try:
            x0 = self._bound_to_index(x_range[0], "x")
            x1 = self._bound_to_index(x_range[1], "x")
            z0 = self._bound_to_index(z_range[0], "z")
            z1 = self._bound_to_index(z_range[1], "z")
        except (TypeError, IndexError):
            raise ValueError("x_range and z_range must be (min, max) pairs.")

        if not (x1 > x0 and z1 > z0):
            raise ValueError("x_range and z_range must satisfy max > min.")
        if not (0 <= x0 < x1 <= self.Nx and 0 <= z0 < z1 <= self.Nz):
            raise ValueError("Region is out of bounds of the simulation grid.")
        return slice(x0, x1), slice(z0, z1)

    def _cell_material_array(self, prefix, component):
        if prefix == "eps":
            return {"xx": self.cell_eps_r_xx, "yy": self.cell_eps_r_yy, "zz": self.cell_eps_r_zz}[component]
        if prefix == "mu":
            return {"xx": self.cell_mu_r_xx, "yy": self.cell_mu_r_yy, "zz": self.cell_mu_r_zz}[component]
        raise ValueError(f"Unknown {prefix} component {component!r}.")

    def _component_array(self, prefix, component):
        if prefix == "pec":
            return {"xx": self.pec_xx_mask, "yy": self.pec_yy_mask, "zz": self.pec_zz_mask}[component]
        if prefix == "pmc":
            return {"xx": self.pmc_xx_mask, "yy": self.pmc_yy_mask, "zz": self.pmc_zz_mask}[component]
        raise ValueError(f"Unknown {prefix} component {component!r}.")

    def _init_operators(self):
        self.DEX_EZ_TO_EX = self._difference_matrix_x(self.shape_ez, self.shape_ex, self.dx, forward=True)
        self.DHX_HY_TO_EZ = -self.DEX_EZ_TO_EX.conj().T
        self.DEX_EY_TO_HZ = self._difference_matrix_x(self.shape_ey, self.shape_hz, self.dx, forward=True)
        self.DHX_HZ_TO_HX = -self.DEX_EY_TO_HZ.conj().T

        self.DEZ_EX = self._periodic_difference_matrix_z(self.shape_ex, self.shape_ex, self.dz, forward=True)
        self.DHZ_HY = -self.DEZ_EX.conj().T
        self.DEZ_EY = self._periodic_difference_matrix_z(self.shape_ey, self.shape_ey, self.dz, forward=True)
        self.DHZ_HX = -self.DEZ_EY.conj().T

        self.AZ_EX = self._periodic_forward_average_z(self.shape_ex)
        self.AZ_HY = self.AZ_EX.conj().T
        self.AZ_EY = self._periodic_forward_average_z(self.shape_ey)
        self.AZ_HX = self.AZ_EY.conj().T

    @staticmethod
    def _flat_index(i, j, nx):
        return i + j * nx

    def _difference_matrix_x(self, in_shape, out_shape, scale, forward=True):
        rows = []
        cols = []
        data = []
        in_nx, _ = in_shape
        out_nx, out_nz = out_shape
        for j in range(out_nz):
            for i in range(out_nx):
                row = self._flat_index(i, j, out_nx)
                entries = ((i + 1, j, 1.0), (i, j, -1.0)) if forward else ((i, j, 1.0), (i - 1, j, -1.0))
                for ci, cj, value in entries:
                    if 0 <= ci < in_shape[0] and 0 <= cj < in_shape[1]:
                        rows.append(row)
                        cols.append(self._flat_index(ci, cj, in_nx))
                        data.append(value / scale)
        return coo_matrix((data, (rows, cols)), shape=(out_nx * out_nz, in_shape[0] * in_shape[1])).tocsr()

    def _periodic_difference_matrix_z(self, in_shape, out_shape, scale, forward=True):
        if in_shape != out_shape:
            raise ValueError("Periodic z derivatives should not change Yee component shape.")
        rows = []
        cols = []
        data = []
        nx, nz = in_shape
        for j in range(nz):
            for i in range(nx):
                row = self._flat_index(i, j, nx)
                if forward:
                    entries = ((i, (j + 1) % nz, 1.0), (i, j, -1.0))
                else:
                    entries = ((i, j, 1.0), (i, (j - 1) % nz, -1.0))
                for ci, cj, value in entries:
                    rows.append(row)
                    cols.append(self._flat_index(ci, cj, nx))
                    data.append(value / scale)
        return coo_matrix((data, (rows, cols)), shape=(nx * nz, nx * nz)).tocsr()

    def _periodic_forward_average_z(self, shape):
        rows = []
        cols = []
        data = []
        nx, nz = shape
        for j in range(nz):
            for i in range(nx):
                row = self._flat_index(i, j, nx)
                for cj in (j, (j + 1) % nz):
                    rows.append(row)
                    cols.append(self._flat_index(i, cj, nx))
                    data.append(0.5)
        return coo_matrix((data, (rows, cols)), shape=(nx * nz, nx * nz)).tocsr()

    def _subpixel_axis(self, start, stop, step, subpixels):
        indices = np.arange(start, stop, dtype=float)
        offsets = (np.arange(subpixels, dtype=float) + 0.5) / subpixels
        return (indices[:, None] + offsets[None, :]) * step

    def _clipped_cell_bbox(self, xmin, xmax, zmin, zmax):
        x0 = max(0, int(np.floor(xmin / self.dx)))
        x1 = min(self.Nx, int(np.ceil(xmax / self.dx)))
        z0 = max(0, int(np.floor(zmin / self.dz)))
        z1 = min(self.Nz, int(np.ceil(zmax / self.dz)))
        return x0, x1, z0, z1

    def _apply_fractional_material(self, epsilon, mu, fraction, sl_x, sl_z):
        epsilon = self._normalise_three("epsilon", epsilon)
        mu = self._normalise_three("mu", mu)
        fraction = np.asarray(fraction, dtype=float)
        if fraction.shape != self.cell_eps_r_xx[sl_x, sl_z].shape:
            raise ValueError("fraction shape does not match target cell region.")

        no_average = self.material_no_average_mask[sl_x, sl_z]
        covered = (fraction > 0.0) & ~no_average
        if not np.any(covered):
            return

        for component, value in zip(("xx", "yy", "zz"), epsilon):
            target = self._cell_material_array("eps", component)[sl_x, sl_z]
            target[covered] = target[covered] * (1.0 - fraction[covered]) + value * fraction[covered]
        for component, value in zip(("xx", "yy", "zz"), mu):
            target = self._cell_material_array("mu", component)[sl_x, sl_z]
            target[covered] = target[covered] * (1.0 - fraction[covered]) + value * fraction[covered]

        self.update_component_materials()
        self._invalidate_solution()

    def add_rectangle(self, epsilon, mu, x_range, z_range, *, subpixels=8):
        """Add a subpixel-smoothed rectangular material region on the cell grid."""
        x_min, x_max = self._range_to_lengths("x_range", x_range, "x")
        z_min, z_max = self._range_to_lengths("z_range", z_range, "z")
        subpixels = self._validate_subpixels(subpixels)
        x0, x1, z0, z1 = self._clipped_cell_bbox(x_min, x_max, z_min, z_max)
        if x0 >= x1 or z0 >= z1:
            return

        xs = self._subpixel_axis(x0, x1, self.dx, subpixels)
        zs = self._subpixel_axis(z0, z1, self.dz, subpixels)
        x_inside = (xs >= x_min) & (xs <= x_max)
        z_inside = (zs >= z_min) & (zs <= z_max)
        fraction = (x_inside[:, :, None, None] & z_inside[None, None, :, :]).mean(axis=(1, 3))
        self._apply_fractional_material(epsilon, mu, fraction, slice(x0, x1), slice(z0, z1))

    def add_pec(self, x_range, z_range, components=None, epsilon=1e8):
        """Add a PEC-like region using a large permittivity penalty.

        Hard DOF elimination makes the generalized periodic TE/TM eigenproblem
        fragile. A large material penalty preserves the periodic coupling and
        matches the historical examples that used high-epsilon metal regions.
        """
        sl_x, sl_z = self._region_slices(x_range, z_range)
        if components is None:
            components = ("xx", "yy", "zz")
        for comp in self._validate_components(components):
            self._cell_material_array("eps", comp)[sl_x, sl_z] = epsilon
        self.material_no_average_mask[sl_x, sl_z] = True
        self._pec_regions.append((sl_x, sl_z))
        self.update_component_materials()
        self._invalidate_solution()

    def add_pmc(self, x_range, z_range, components=None, mu=1e8):
        """Add a PMC-like region using a large permeability penalty."""
        sl_x, sl_z = self._region_slices(x_range, z_range)
        if components is None:
            components = ("xx", "yy", "zz")
        for comp in self._validate_components(components):
            self._cell_material_array("mu", comp)[sl_x, sl_z] = mu
        self.material_no_average_mask[sl_x, sl_z] = True
        self._pmc_regions.append((sl_x, sl_z))
        self.update_component_materials()
        self._invalidate_solution()

    def add_pml(self, pml_width=30, n=3, sigma_max=5.0, direction="all"):
        """Add a simple x-directed uniaxial PML."""
        pml_width = int(pml_width)
        if pml_width <= 0:
            raise ValueError("pml_width must be positive.")
        if direction not in ("x-", "x+", "x", "all"):
            raise ValueError("direction must be one of 'x-', 'x+', 'x', or 'all'.")

        sigma_x = np.zeros(self.shape_cell, dtype=float)

        if direction in ("x-", "x", "all"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[i, :] = sigma_max * ((pml_width - i) / pml_width) ** n
        if direction in ("x+", "x", "all"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[-i - 1, :] = sigma_max * ((pml_width - i) / pml_width) ** n

        Sx = 1.0 + 1j * sigma_x / (self.epsilon0 * self.omega)
        self.cell_eps_r_xx *= 1 / Sx
        self.cell_eps_r_yy *= Sx
        self.cell_eps_r_zz *= Sx
        self.cell_mu_r_xx *= 1 / Sx
        self.cell_mu_r_yy *= Sx
        self.cell_mu_r_zz *= Sx
        self.update_component_materials()
        self._invalidate_solution()

    def _flat(self, values):
        return values.ravel(order="F")

    @staticmethod
    def _diag(values):
        return diags(values.ravel(order="F"), format="csr")

    def _average_periodic_z(self, values, no_average_mask=None):
        out = 0.5 * (values + np.roll(values, -1, axis=1))
        if no_average_mask is not None:
            ii, jj = np.nonzero(no_average_mask)
            out[ii, jj] = values[ii, jj]
            out[ii, (jj - 1) % self.Nz] = values[ii, jj]
        return out

    def _average_x(self, values, no_average_mask=None):
        out = np.zeros((self.Nx + 1, self.Nz), dtype=complex)
        counts = np.zeros((self.Nx + 1, self.Nz), dtype=float)
        out[:self.Nx, :] += values
        counts[:self.Nx, :] += 1
        out[1:, :] += values
        counts[1:, :] += 1
        out = out / counts
        if no_average_mask is not None:
            ii, jj = np.nonzero(no_average_mask)
            out[ii, jj] = values[ii, jj]
            out[ii + 1, jj] = values[ii, jj]
        return out

    def _material_on_fields(self, eps_r_xx, eps_r_yy, eps_r_zz, mu_r_xx, mu_r_yy, mu_r_zz, no_average_mask):
        return {
            "eps_xx": self._average_periodic_z(eps_r_xx, no_average_mask),
            "eps_yy": self._average_x(eps_r_yy, no_average_mask),
            "eps_zz": self._average_x(eps_r_zz, no_average_mask),
            "mu_xx": self._average_x(mu_r_xx, no_average_mask),
            "mu_yy": self._average_periodic_z(mu_r_yy, no_average_mask),
            "mu_zz": mu_r_zz.copy(),
        }

    def _set_component_materials(self, materials):
        self.eps_r_xx = materials["eps_xx"].copy()
        self.eps_r_yy = materials["eps_yy"].copy()
        self.eps_r_zz = materials["eps_zz"].copy()
        self.mu_r_xx = materials["mu_xx"].copy()
        self.mu_r_yy = materials["mu_yy"].copy()
        self.mu_r_zz = materials["mu_zz"].copy()

    def update_component_materials(self):
        """Refresh component-location tensors from cell-centered material grids."""
        materials = self._material_on_fields(
            self.cell_eps_r_xx,
            self.cell_eps_r_yy,
            self.cell_eps_r_zz,
            self.cell_mu_r_xx,
            self.cell_mu_r_yy,
            self.cell_mu_r_zz,
            self.material_no_average_mask,
        )
        self._set_component_materials(materials)
        return materials

    def _effective_materials_and_masks(self):
        materials = self._material_on_fields(
            self.cell_eps_r_xx.copy(),
            self.cell_eps_r_yy.copy(),
            self.cell_eps_r_zz.copy(),
            self.cell_mu_r_xx.copy(),
            self.cell_mu_r_yy.copy(),
            self.cell_mu_r_zz.copy(),
            self.material_no_average_mask.copy(),
        )

        eps_r_xx = materials["eps_xx"]
        eps_r_yy = materials["eps_yy"]
        eps_r_zz = materials["eps_zz"]
        mu_r_xx = materials["mu_xx"]
        mu_r_yy = materials["mu_yy"]
        mu_r_zz = materials["mu_zz"]

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
        self._set_component_materials(materials)

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

    def _build_tm_system(self, eps_r_xx, eps_r_zz, mu_r_yy):
        eps_r_xx_diag = self._diag(eps_r_xx)
        eps_r_zz_diag = self._diag(eps_r_zz)
        mu_r_yy_diag = self._diag(mu_r_yy)
        zero = csr_matrix((self.n_ex, self.n_hy), dtype=complex)

        D1 = 1j * self.omega * self.mu0 * mu_r_yy_diag
        D1 += 1j / self.omega * self.DEX_EZ_TO_EX @ (
            1 / self.epsilon0 * eps_r_zz_diag.power(-1) @ self.DHX_HY_TO_EZ
        )
        D2 = 1j * self.omega * self.epsilon0 * eps_r_xx_diag
        A = bmat([[self.DEZ_EX, D1], [D2, self.DHZ_HY]], format="csr")
        B = bmat([[self.AZ_EX, zero], [zero, self.AZ_HY]], format="csr")
        return A, B

    def _build_te_system(self, eps_r_yy, mu_r_xx, mu_r_zz):
        eps_r_yy_diag = self._diag(eps_r_yy)
        mu_r_xx_diag = self._diag(mu_r_xx)
        mu_r_zz_diag = self._diag(mu_r_zz)
        zero = csr_matrix((self.n_hx, self.n_ey), dtype=complex)

        D1 = -1j * self.omega * self.epsilon0 * eps_r_yy_diag
        D1 -= 1j / self.omega * self.DHX_HZ_TO_HX @ (
            1 / self.mu0 * mu_r_zz_diag.power(-1) @ self.DEX_EY_TO_HZ
        )
        D2 = -1j * self.omega * self.mu0 * mu_r_xx_diag
        A = bmat([[self.DHZ_HX, D1], [D2, self.DEZ_EY]], format="csr")
        B = bmat([[self.AZ_HX, zero], [zero, self.AZ_EY]], format="csr")
        return A, B

    def _free_mask(self, pec_xx_mask, pec_yy_mask, pmc_xx_mask, pmc_yy_mask):
        if self.polarization == "TM":
            first = ~self._flat(pec_xx_mask)
            second = ~self._flat(pmc_yy_mask)
        else:
            first = ~self._flat(pmc_xx_mask)
            second = ~self._flat(pec_yy_mask)
        return np.concatenate((first, second))

    def solve(self, guess=None, tol=None, ncv=None):
        """Solve periodic modes, optionally overriding eigs controls for this call."""
        guess = self.guess if guess is None else guess
        tol = self.tol if tol is None else tol
        ncv = self.ncv if ncv is None else ncv

        (
            eps_r_xx,
            eps_r_yy,
            eps_r_zz,
            mu_r_xx,
            mu_r_yy,
            mu_r_zz,
            pec_xx_mask,
            pec_yy_mask,
            _pec_zz_mask,
            pmc_xx_mask,
            pmc_yy_mask,
            _pmc_zz_mask,
        ) = self._effective_materials_and_masks()

        if self.polarization == "TM":
            A, B = self._build_tm_system(eps_r_xx, eps_r_zz, mu_r_yy)
        else:
            A, B = self._build_te_system(eps_r_yy, mu_r_xx, mu_r_zz)

        free = self._free_mask(pec_xx_mask, pec_yy_mask, pmc_xx_mask, pmc_yy_mask)
        if np.count_nonzero(free) <= self.num_modes:
            raise ValueError(f"Not enough unconstrained DOFs to solve {self.num_modes} modes.")

        A = A[free, :][:, free]
        B = B[free, :][:, free]
        v0 = np.ones(A.shape[0], dtype=complex)
        eigenvalues, eigenvectors_reduced = eigs(
            A,
            M=B,
            k=self.num_modes,
            sigma=guess,
            tol=tol,
            ncv=ncv,
            v0=v0,
        )
        order = np.argsort(np.abs(eigenvalues - guess))
        eigenvalues = eigenvalues[order]
        eigenvectors_reduced = eigenvectors_reduced[:, order]

        eigenvectors = np.zeros((free.size, self.num_modes), dtype=complex)
        eigenvectors[free, :] = eigenvectors_reduced

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.neff = eigenvalues / self.k0
        self.propagation_constant = np.imag(self.neff)
        self.attenuation_constant = np.real(self.neff)

        if self.polarization == "TM":
            self.Ex = eigenvectors[:self.n_ex, :]
            self.Hy = eigenvectors[self.n_ex:, :]
        else:
            self.Hx = eigenvectors[:self.n_hx, :]
            self.Ey = eigenvectors[self.n_hx:, :]

        self.spurious_scores = np.zeros(self.num_modes, dtype=float)
        self.accepted_candidate_indices = np.arange(self.num_modes)
        self.rejected_candidate_indices = np.array([], dtype=int)
        self.unselected_candidate_indices = np.array([], dtype=int)

    def _field_array(self, field, mode, shape):
        return field[:, mode].reshape(shape, order="F")

    def _material_plot_data(self):
        data = np.abs(self.cell_eps_r_yy).astype(float)
        if self._pec_regions or self._pmc_regions:
            data = data.copy()
            for sl_x, sl_z in self._pec_regions + self._pmc_regions:
                data[sl_x, sl_z] = np.nan
        return data

    def _add_boundary_rectangles(self, ax):
        for sl_x, sl_z in self._pec_regions:
            self._add_boundary_rectangle(ax, sl_x, sl_z, label="PEC")
        for sl_x, sl_z in self._pmc_regions:
            self._add_boundary_rectangle(ax, sl_x, sl_z, label="PMC")

    def _add_boundary_rectangle(self, ax, sl_x, sl_z, label):
        x0 = sl_x.start * self.dx * 1e3
        x1 = sl_x.stop * self.dx * 1e3
        z0 = sl_z.start * self.dz * 1e3
        z1 = sl_z.stop * self.dz * 1e3
        patch = Rectangle(
            (z0, x0),
            z1 - z0,
            x1 - x0,
            facecolor="yellow",
            edgecolor="goldenrod",
            linewidth=1.0,
            alpha=0.75,
            label=label,
        )
        ax.add_patch(patch)

    def visualize_with_gui(self):
        if self.eigenvectors is None:
            raise RuntimeError("solve() must be called before visualize_with_gui().")
        import sys

        root = tk.Tk()
        root.title(f"{self.polarization} Periodic Mode Viewer")
        if sys.platform == "darwin":
            root.tk.call("tk", "scaling", 1.0)

        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        controls_frame = tk.Frame(root)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=10)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=100)
        colorbars = []
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def plot_mode(selected_mode):
            mode = int(selected_mode) - 1
            if self.polarization == "TM":
                field_data = [
                    self._material_plot_data(),
                    self._field_array(self.Ex, mode, self.shape_ex),
                    self._field_array(self.Hy, mode, self.shape_hy),
                ]
                titles = [r"Structure (Abs($\epsilon$))", "Ex (norm.)", "Hy (norm.)"]
            else:
                field_data = [
                    self._material_plot_data(),
                    self._field_array(self.Hx, mode, self.shape_hx),
                    self._field_array(self.Ey, mode, self.shape_ey),
                ]
                titles = [r"Structure (Abs($\epsilon$))", "Hx (norm.)", "Ey (norm.)"]

            for colorbar in colorbars:
                colorbar.remove()
            colorbars.clear()
            for ax in axes:
                ax.clear()

            for i, ax in enumerate(axes):
                data = np.abs(field_data[i])
                if i > 0:
                    norm = np.max(data)
                    data = data / norm if norm > 0 else data
                cmap = plt.get_cmap("viridis").copy()
                if i == 0:
                    cmap.set_bad(color="white", alpha=0.0)
                im = ax.imshow(
                    data,
                    cmap=cmap,
                    origin="lower",
                    extent=[0, self.z_range * 1e3, 0, self.x_range * 1e3],
                )
                if i == 0:
                    self._add_boundary_rectangles(ax)
                ax.set_title(titles[i])
                ax.set_xlabel("z (mm)")
                ax.set_ylabel("x (mm)")
                colorbars.append(plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04))

            fig.suptitle(
                f"{self.freq / 1e9:.2f} GHz, Mode {mode + 1}: "
                f"beta = {self.propagation_constant[mode]:.4f}, "
                f"alpha = {self.attenuation_constant[mode]:.4f}",
                fontsize=14,
            )
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

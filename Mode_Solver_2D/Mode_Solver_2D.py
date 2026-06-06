import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from scipy.sparse import bmat, coo_matrix, diags
from scipy.sparse.linalg import eigs


class ModeSolver2D:
    """2D full-vector FDFD mode solver on a true staggered Yee grid."""

    def __init__(self, frequency, x_range, y_range, Nx, Ny, num_modes, guess=None):
        self.frequency = frequency
        self.x_range = x_range
        self.y_range = y_range
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        if self.Nx <= 0 or self.Ny <= 0:
            raise ValueError("Nx and Ny must be positive.")

        self.dx = x_range / self.Nx
        self.dy = y_range / self.Ny
        self.epsilon0 = 8.854187817e-12
        self.mu0 = 4e-7 * np.pi
        self.c = 1 / np.sqrt(self.epsilon0 * self.mu0)
        self.k_0 = 2 * np.pi * frequency / self.c
        self.dx_normalized = self.k_0 * self.dx
        self.dy_normalized = self.k_0 * self.dy

        self.shape_cell = (self.Nx, self.Ny)
        self.shape_ex = (self.Nx, self.Ny + 1)
        self.shape_ey = (self.Nx + 1, self.Ny)
        self.shape_ez = (self.Nx + 1, self.Ny + 1)
        self.shape_hx = self.shape_ey
        self.shape_hy = self.shape_ex
        self.shape_hz = self.shape_cell

        self.n_ex = int(np.prod(self.shape_ex))
        self.n_ey = int(np.prod(self.shape_ey))
        self.n_ez = int(np.prod(self.shape_ez))
        self.n_hx = self.n_ey
        self.n_hy = self.n_ex
        self.n_hz = self.Nx * self.Ny
        self.n_e = self.n_ex + self.n_ey
        self.n_h = self.n_hx + self.n_hy

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

        self.num_modes = int(num_modes)
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive.")
        self.guess = guess
        self._auto_guess = guess is None
        if self._auto_guess:
            self.guess = self._default_guess()
        self._invalidate_solution()

    @staticmethod
    def _max_magnitude(arr):
        values = np.abs(np.asarray(arr))
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return 0.0
        return np.max(finite_values)

    def _default_guess(self):
        return -max(
            self._max_magnitude(arr)
            for arr in [self.cell_eps_r_xx, self.cell_eps_r_yy, self.cell_eps_r_zz,
                        self.cell_mu_r_xx, self.cell_mu_r_yy, self.cell_mu_r_zz]
        )

    def _resolve_eigs_guess(self, sigma):
        if sigma is not None:
            return sigma
        if self._auto_guess:
            self.guess = self._default_guess()
        return self.guess

    def _invalidate_solution(self):
        self.eigenvalues = None
        self.eigenvectors = None
        self.neff = None
        self.propagation_constant = None
        self.attenuation_constant = None
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

    def add_rectangle(self, epsilon, mu, x_range, y_range, *, average=True):
        """Add a rectangular isotropic or diagonal-anisotropic material region on the cell grid."""
        sl_x, sl_y = self._region_slices(x_range, y_range)
        epsilon = self._normalise_three("epsilon", epsilon)
        mu = self._normalise_three("mu", mu)

        self.cell_eps_r_xx[sl_x, sl_y] = epsilon[0]
        self.cell_eps_r_yy[sl_x, sl_y] = epsilon[1]
        self.cell_eps_r_zz[sl_x, sl_y] = epsilon[2]
        self.cell_mu_r_xx[sl_x, sl_y] = mu[0]
        self.cell_mu_r_yy[sl_x, sl_y] = mu[1]
        self.cell_mu_r_zz[sl_x, sl_y] = mu[2]
        if average:
            self.material_no_average_mask[sl_x, sl_y] = False
        else:
            self.material_no_average_mask[sl_x, sl_y] = True
        self.update_component_materials()
        self._invalidate_solution()

    def add_pec(self, x_range, y_range, components=None):
        """Add a PEC cell region and expand it onto surrounding Yee electric components."""
        sl_x, sl_y = self._region_slices(x_range, y_range)
        self._pec_regions.append((sl_x, sl_y))
        cell_mask = np.zeros(self.shape_cell, dtype=bool)
        cell_mask[sl_x, sl_y] = True
        xx_mask, yy_mask, zz_mask = self.component_masks_from_cell_mask(cell_mask, field="electric")
        selected = ("xx", "yy", "zz") if components is None else self._validate_components(components)
        for comp, mask in (("xx", xx_mask), ("yy", yy_mask), ("zz", zz_mask)):
            if comp in selected:
                self._component_array("pec", comp)[:] |= mask
        self._effective_materials_and_masks()
        self._invalidate_solution()

    def add_pmc(self, x_range, y_range, components=None):
        """Add a PMC cell region and expand it onto surrounding Yee magnetic components."""
        sl_x, sl_y = self._region_slices(x_range, y_range)
        self._pmc_regions.append((sl_x, sl_y))
        cell_mask = np.zeros(self.shape_cell, dtype=bool)
        cell_mask[sl_x, sl_y] = True
        xx_mask, yy_mask, zz_mask = self.component_masks_from_cell_mask(cell_mask, field="magnetic")
        selected = ("xx", "yy", "zz") if components is None else self._validate_components(components)
        for comp, mask in (("xx", xx_mask), ("yy", yy_mask), ("zz", zz_mask)):
            if comp in selected:
                self._component_array("pmc", comp)[:] |= mask
        self._effective_materials_and_masks()
        self._invalidate_solution()

    def component_masks_from_cell_mask(self, cell_mask, field="electric"):
        """Expand a cell mask to the component locations surrounding each marked cell."""
        mask = np.asarray(cell_mask, dtype=bool)
        if mask.shape != self.shape_cell:
            raise ValueError(f"cell_mask must have shape {self.shape_cell}.")

        if field == "electric":
            xx_mask = np.zeros(self.shape_ex, dtype=bool)
            yy_mask = np.zeros(self.shape_ey, dtype=bool)
            zz_mask = np.zeros(self.shape_ez, dtype=bool)
            ii, jj = np.nonzero(mask)
            xx_mask[ii, jj] = True
            xx_mask[ii, jj + 1] = True
            yy_mask[ii, jj] = True
            yy_mask[ii + 1, jj] = True
            zz_mask[ii, jj] = True
            zz_mask[ii + 1, jj] = True
            zz_mask[ii, jj + 1] = True
            zz_mask[ii + 1, jj + 1] = True
            return xx_mask, yy_mask, zz_mask

        if field == "magnetic":
            xx_mask = np.zeros(self.shape_hx, dtype=bool)
            yy_mask = np.zeros(self.shape_hy, dtype=bool)
            zz_mask = np.zeros(self.shape_hz, dtype=bool)
            ii, jj = np.nonzero(mask)
            xx_mask[ii, jj] = True
            xx_mask[ii + 1, jj] = True
            yy_mask[ii, jj] = True
            yy_mask[ii, jj + 1] = True
            zz_mask[ii, jj] = True
            return xx_mask, yy_mask, zz_mask

        raise ValueError("field must be 'electric' or 'magnetic'.")

    def add_pml(self, pml_width=50, n=3, sigma_max=5, direction="all"):
        """Add a simple uniaxial PML by stretching cell-centered epsilon and mu tensors."""
        pml_width = int(pml_width)
        if pml_width <= 0:
            raise ValueError("pml_width must be positive.")
        if direction not in ("x-", "x+", "x", "y-", "y+", "y", "all"):
            raise ValueError("direction must be one of 'x-', 'x+', 'x', 'y-', 'y+', 'y', or 'all'.")

        sigma_x = np.zeros(self.shape_cell, dtype=float)
        sigma_y = np.zeros(self.shape_cell, dtype=float)

        if direction in ("x-", "x", "all"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[i, :] = sigma_max * ((pml_width - i) / pml_width) ** n
        if direction in ("x+", "x", "all"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[-i - 1, :] = sigma_max * ((pml_width - i) / pml_width) ** n
        if direction in ("y-", "y", "all"):
            for j in range(min(pml_width, self.Ny)):
                sigma_y[:, j] = sigma_max * ((pml_width - j) / pml_width) ** n
        if direction in ("y+", "y", "all"):
            for j in range(min(pml_width, self.Ny)):
                sigma_y[:, -j - 1] = sigma_max * ((pml_width - j) / pml_width) ** n

        omega = 2 * np.pi * self.frequency
        Sx = 1.0 + 1j * sigma_x / (self.epsilon0 * omega)
        Sy = 1.0 + 1j * sigma_y / (self.epsilon0 * omega)

        self.cell_eps_r_xx *= Sy / Sx
        self.cell_eps_r_yy *= Sx / Sy
        self.cell_eps_r_zz *= Sx * Sy
        self.cell_mu_r_xx *= Sy / Sx
        self.cell_mu_r_yy *= Sx / Sy
        self.cell_mu_r_zz *= Sx * Sy
        self.update_component_materials()
        self._invalidate_solution()

    def add_UPML(self, pml_width=50, n=3, sigma_max=5, direction="all"):
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
        delta_eps = -1j / (2 * np.pi * self.frequency * self.epsilon0 * thickness * Zs)
        for comp in self._validate_components(eps_components):
            self._cell_material_array("eps", comp)[sl_x, sl_y] += delta_eps
        self.update_component_materials()
        self._invalidate_solution()

    @staticmethod
    def _flat_index(i, j, nx):
        return i + j * nx

    def _difference_matrix_x(self, in_shape, out_shape, scale, forward=True):
        rows = []
        cols = []
        data = []
        in_nx, _ = in_shape
        out_nx, out_ny = out_shape
        for j in range(out_ny):
            for i in range(out_nx):
                row = self._flat_index(i, j, out_nx)
                entries = ((i + 1, j, 1.0), (i, j, -1.0)) if forward else ((i, j, 1.0), (i - 1, j, -1.0))
                for ci, cj, value in entries:
                    if 0 <= ci < in_shape[0] and 0 <= cj < in_shape[1]:
                        rows.append(row)
                        cols.append(self._flat_index(ci, cj, in_nx))
                        data.append(value / scale)
        return coo_matrix((data, (rows, cols)), shape=(out_nx * out_ny, in_shape[0] * in_shape[1])).tocsr()

    def _difference_matrix_y(self, in_shape, out_shape, scale, forward=True):
        rows = []
        cols = []
        data = []
        in_nx, _ = in_shape
        out_nx, out_ny = out_shape
        for j in range(out_ny):
            for i in range(out_nx):
                row = self._flat_index(i, j, out_nx)
                entries = ((i, j + 1, 1.0), (i, j, -1.0)) if forward else ((i, j, 1.0), (i, j - 1, -1.0))
                for ci, cj, value in entries:
                    if 0 <= ci < in_shape[0] and 0 <= cj < in_shape[1]:
                        rows.append(row)
                        cols.append(self._flat_index(ci, cj, in_nx))
                        data.append(value / scale)
        return coo_matrix((data, (rows, cols)), shape=(out_nx * out_ny, in_shape[0] * in_shape[1])).tocsr()

    def _yeeder2d(self):
        """Generate rectangular derivative matrices between true Yee component grids."""
        dx = self.dx_normalized
        dy = self.dy_normalized

        self.DEX_EZ_TO_EX = self._difference_matrix_x(self.shape_ez, self.shape_ex, dx, forward=True)
        self.DEY_EZ_TO_EY = self._difference_matrix_y(self.shape_ez, self.shape_ey, dy, forward=True)
        self.DEX_EY_TO_HZ = self._difference_matrix_x(self.shape_ey, self.shape_hz, dx, forward=True)
        self.DEY_EX_TO_HZ = self._difference_matrix_y(self.shape_ex, self.shape_hz, dy, forward=True)

        self.DHX_HY_TO_EZ = -self.DEX_EZ_TO_EX.conj().T
        self.DHY_HX_TO_EZ = -self.DEY_EZ_TO_EY.conj().T
        self.DHX_HZ_TO_HX = -self.DEX_EY_TO_HZ.conj().T
        self.DHY_HZ_TO_HY = -self.DEY_EX_TO_HZ.conj().T

        self.DEX = self.DEX_EY_TO_HZ
        self.DEY = self.DEY_EX_TO_HZ
        self.DHX = self.DHX_HY_TO_EZ
        self.DHY = self.DHY_HX_TO_EZ
        return (
            self.DEX_EZ_TO_EX,
            self.DEY_EZ_TO_EY,
            self.DEX_EY_TO_HZ,
            self.DEY_EX_TO_HZ,
            self.DHX_HY_TO_EZ,
            self.DHY_HX_TO_EZ,
            self.DHX_HZ_TO_HX,
            self.DHY_HZ_TO_HY,
        )

    def _average_to_ex(self, values, no_average_mask=None):
        out = np.zeros(self.shape_ex, dtype=complex)
        counts = np.zeros(self.shape_ex, dtype=float)
        out[:, :self.Ny] += values
        counts[:, :self.Ny] += 1
        out[:, 1:] += values
        counts[:, 1:] += 1
        out = out / counts
        if no_average_mask is not None:
            ii, jj = np.nonzero(no_average_mask)
            out[ii, jj] = values[ii, jj]
            out[ii, jj + 1] = values[ii, jj]
        return out

    def _average_to_ey(self, values, no_average_mask=None):
        out = np.zeros(self.shape_ey, dtype=complex)
        counts = np.zeros(self.shape_ey, dtype=float)
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

    def _average_to_ez(self, values, no_average_mask=None):
        out = np.zeros(self.shape_ez, dtype=complex)
        counts = np.zeros(self.shape_ez, dtype=float)
        out[:self.Nx, :self.Ny] += values
        counts[:self.Nx, :self.Ny] += 1
        out[1:, :self.Ny] += values
        counts[1:, :self.Ny] += 1
        out[:self.Nx, 1:] += values
        counts[:self.Nx, 1:] += 1
        out[1:, 1:] += values
        counts[1:, 1:] += 1
        out = out / counts
        if no_average_mask is not None:
            ii, jj = np.nonzero(no_average_mask)
            out[ii, jj] = values[ii, jj]
            out[ii + 1, jj] = values[ii, jj]
            out[ii, jj + 1] = values[ii, jj]
            out[ii + 1, jj + 1] = values[ii, jj]
        return out

    def _material_on_fields(self, eps_r_xx, eps_r_yy, eps_r_zz, mu_r_xx, mu_r_yy, mu_r_zz, no_average_mask):
        return {
            "eps_xx": self._average_to_ex(eps_r_xx, no_average_mask),
            "eps_yy": self._average_to_ey(eps_r_yy, no_average_mask),
            "eps_zz": self._average_to_ez(eps_r_zz, no_average_mask),
            "mu_xx": self._average_to_ey(mu_r_xx, no_average_mask),
            "mu_yy": self._average_to_ex(mu_r_yy, no_average_mask),
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
        """Refresh component-location material tensors from the cell-centered source grid."""
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
        eps_r_xx = self.cell_eps_r_xx.copy()
        eps_r_yy = self.cell_eps_r_yy.copy()
        eps_r_zz = self.cell_eps_r_zz.copy()
        mu_r_xx = self.cell_mu_r_xx.copy()
        mu_r_yy = self.cell_mu_r_yy.copy()
        mu_r_zz = self.cell_mu_r_zz.copy()
        no_average_mask = self.material_no_average_mask.copy()

        pec_xx_mask = self.pec_xx_mask.copy()
        pec_yy_mask = self.pec_yy_mask.copy()
        pec_zz_mask = self.pec_zz_mask.copy()
        pmc_xx_mask = self.pmc_xx_mask.copy()
        pmc_yy_mask = self.pmc_yy_mask.copy()
        pmc_zz_mask = self.pmc_zz_mask.copy()

        electric_targets = {"xx": pec_xx_mask, "yy": pec_yy_mask, "zz": pec_zz_mask}
        magnetic_targets = {"xx": pmc_xx_mask, "yy": pmc_yy_mask, "zz": pmc_zz_mask}
        mask_index = {"xx": 0, "yy": 1, "zz": 2}
        for component, values in (("xx", eps_r_xx), ("yy", eps_r_yy), ("zz", eps_r_zz)):
            bad_cells = ~np.isfinite(values)
            if np.any(bad_cells):
                masks = self.component_masks_from_cell_mask(bad_cells, field="electric")
                electric_targets[component][:] |= masks[mask_index[component]]
                values[bad_cells] = 1.0 + 0j

        for component, values in (("xx", mu_r_xx), ("yy", mu_r_yy), ("zz", mu_r_zz)):
            bad_cells = ~np.isfinite(values)
            if np.any(bad_cells):
                masks = self.component_masks_from_cell_mask(bad_cells, field="magnetic")
                magnetic_targets[component][:] |= masks[mask_index[component]]
                values[bad_cells] = 1.0 + 0j

        materials = self._material_on_fields(
            eps_r_xx,
            eps_r_yy,
            eps_r_zz,
            mu_r_xx,
            mu_r_yy,
            mu_r_zz,
            no_average_mask,
        )
        materials["eps_xx"][pec_xx_mask] = 1.0 + 0j
        materials["eps_yy"][pec_yy_mask] = 1.0 + 0j
        materials["eps_zz"][pec_zz_mask] = 1.0 + 0j
        materials["mu_xx"][pmc_xx_mask] = 1.0 + 0j
        materials["mu_yy"][pmc_yy_mask] = 1.0 + 0j
        materials["mu_zz"][pmc_zz_mask] = 1.0 + 0j
        self._set_component_materials(materials)

        return materials, pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask

    def _inverse_diag_on_free(self, values, constrained_mask):
        flat = values.ravel(order="F")
        constrained = constrained_mask.ravel(order="F")
        inverse = np.zeros_like(flat, dtype=complex)
        inverse[~constrained] = 1.0 / flat[~constrained]
        return diags(inverse, format="csr")

    @staticmethod
    def _diag(values):
        return diags(values.ravel(order="F"), format="csr")

    def _unflatten_modes(self, flat_modes, shape):
        return np.asarray(flat_modes, dtype=complex).reshape((*shape, flat_modes.shape[1]), order="F")

    def _zero_constrained_fields(self, pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask):
        self.Ex[pec_xx_mask, :] = 0.0
        self.Ey[pec_yy_mask, :] = 0.0
        self.Ez[pec_zz_mask, :] = 0.0
        self.Hx[pmc_xx_mask, :] = 0.0
        self.Hy[pmc_yy_mask, :] = 0.0
        self.Hz[pmc_zz_mask, :] = 0.0

    def solve(self, sigma=None):
        """Solve for transverse modes and recover all six staggered field components."""
        sigma = self._resolve_eigs_guess(sigma)
        materials, pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask = (
            self._effective_materials_and_masks()
        )

        eps_xx_diag = self._diag(materials["eps_xx"])
        eps_yy_diag = self._diag(materials["eps_yy"])
        mu_xx_diag = self._diag(materials["mu_xx"])
        mu_yy_diag = self._diag(materials["mu_yy"])
        eps_zz_inv = self._inverse_diag_on_free(materials["eps_zz"], pec_zz_mask)
        mu_zz_inv = self._inverse_diag_on_free(materials["mu_zz"], pmc_zz_mask)

        (
            d_ez_ex,
            d_ez_ey,
            d_ey_hz,
            d_ex_hz,
            d_hy_ez,
            d_hx_ez,
            d_hz_hx,
            d_hz_hy,
        ) = self._yeeder2d()

        P11 = d_ez_ex @ eps_zz_inv @ d_hx_ez
        P12 = -(d_ez_ex @ eps_zz_inv @ d_hy_ez + mu_yy_diag)
        P21 = d_ez_ey @ eps_zz_inv @ d_hx_ez + mu_xx_diag
        P22 = -d_ez_ey @ eps_zz_inv @ d_hy_ez
        P = bmat([[P11, P12], [P21, P22]], format="csr")

        Q11 = d_hz_hx @ mu_zz_inv @ d_ex_hz
        Q12 = -(d_hz_hx @ mu_zz_inv @ d_ey_hz + eps_yy_diag)
        Q21 = d_hz_hy @ mu_zz_inv @ d_ex_hz + eps_xx_diag
        Q22 = -d_hz_hy @ mu_zz_inv @ d_ey_hz
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

        eigenvalues, eigenvectors_reduced = eigs(Omega, k=self.num_modes, sigma=sigma)
        eigenvectors = np.zeros((self.n_e, self.num_modes), dtype=complex)
        eigenvectors[free_exy, :] = eigenvectors_reduced

        order = np.argsort(np.real(eigenvalues))
        self.eigenvalues = eigenvalues[order]
        Exy_flat = eigenvectors[:, order]
        self.eigenvectors = Exy_flat
        self.neff = self._passive_positive_neff(-self.eigenvalues)
        self.propagation_constant = np.real(self.neff)
        self.attenuation_constant = np.imag(self.neff)

        sqrt_eigenvalues = np.sqrt(self.eigenvalues)
        if np.any(np.abs(sqrt_eigenvalues) < 1e-300):
            raise ValueError("Encountered a near-zero eigenvalue while reconstructing magnetic fields.")
        eigenvalues_inv = diags(1.0 / sqrt_eigenvalues, format="csr")

        ex_flat = np.asarray(Exy_flat[:self.n_ex, :], dtype=complex)
        ey_flat = np.asarray(Exy_flat[self.n_ex:, :], dtype=complex)
        Hxy_reduced = Q_reduced @ Exy_flat @ eigenvalues_inv
        Hxy_flat = np.zeros((self.n_h, self.num_modes), dtype=complex)
        Hxy_flat[free_hxy, :] = Hxy_reduced
        hx_flat = np.asarray(Hxy_flat[:self.n_hx, :], dtype=complex)
        hy_flat = np.asarray(Hxy_flat[self.n_hx:, :], dtype=complex)
        ez_flat = np.asarray(eps_zz_inv @ (d_hy_ez @ hy_flat - d_hx_ez @ hx_flat), dtype=complex)
        hz_flat = np.asarray(mu_zz_inv @ (d_ey_hz @ ey_flat - d_ex_hz @ ex_flat), dtype=complex)

        self.Ex = self._unflatten_modes(ex_flat, self.shape_ex)
        self.Ey = self._unflatten_modes(ey_flat, self.shape_ey)
        self.Ez = self._unflatten_modes(ez_flat, self.shape_ez)
        self.Hx = self._unflatten_modes(hx_flat, self.shape_hx)
        self.Hy = self._unflatten_modes(hy_flat, self.shape_hy)
        self.Hz = self._unflatten_modes(hz_flat, self.shape_hz)
        self._zero_constrained_fields(pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask)

    def _has_lossy_material(self):
        for values in (
                self.cell_eps_r_xx,
                self.cell_eps_r_yy,
                self.cell_eps_r_zz,
                self.cell_mu_r_xx,
                self.cell_mu_r_yy,
                self.cell_mu_r_zz,
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
        return field[:, :, mode]

    def _center_to_cells(self, name, data):
        if name in ("ex", "hy"):
            return 0.5 * (data[:, :self.Ny] + data[:, 1:])
        if name in ("ey", "hx"):
            return 0.5 * (data[:self.Nx, :] + data[1:, :])
        if name == "ez":
            return 0.25 * (data[:self.Nx, :self.Ny] + data[1:, :self.Ny] + data[:self.Nx, 1:] + data[1:, 1:])
        if name == "hz":
            return data
        raise ValueError(f"Unknown field name {name!r}.")

    def _add_boundary_background(self, ax):
        for sl_x, sl_y in self._pec_regions:
            self._add_boundary_rectangle(ax, sl_x, sl_y, label="PEC")
        for sl_x, sl_y in self._pmc_regions:
            self._add_boundary_rectangle(ax, sl_x, sl_y, label="PMC")

    def _add_boundary_rectangle(self, ax, sl_x, sl_y, label):
        x0 = sl_x.start * self.dx * 1e3
        x1 = sl_x.stop * self.dx * 1e3
        y0 = sl_y.start * self.dy * 1e3
        y1 = sl_y.stop * self.dy * 1e3
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="yellow",
                edgecolor="goldenrod",
                linewidth=1.0,
                alpha=0.35,
                label=label,
                zorder=0,
            )
        )

    def _component_fields_for_mode(self, mode):
        return {
            "ex": (self._field_array(self.Ex, mode), "Ex"),
            "ey": (self._field_array(self.Ey, mode), "Ey"),
            "ez": (self._field_array(self.Ez, mode), "Ez"),
            "hx": (self._field_array(self.Hx, mode), "Hx"),
            "hy": (self._field_array(self.Hy, mode), "Hy"),
            "hz": (self._field_array(self.Hz, mode), "Hz"),
        }

    def visualize(self, mode=1, **kwargs):
        """Visualize selected field components for a given one-based mode index."""
        if self.eigenvalues is None:
            raise RuntimeError("solve() must be called before visualize().")
        mode -= 1
        if not (0 <= mode < self.num_modes):
            raise ValueError("mode is out of range.")

        fields = self._component_fields_for_mode(mode)
        e_abs = np.sqrt(sum(np.abs(self._center_to_cells(key, fields[key][0])) ** 2 for key in ("ex", "ey", "ez")))
        h_abs = np.sqrt(sum(np.abs(self._center_to_cells(key, fields[key][0])) ** 2 for key in ("hx", "hy", "hz")))
        fields["eabs"] = (e_abs, "|E| cell-centered")
        fields["habs"] = (h_abs, "|H| cell-centered")

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
            self._add_boundary_background(ax)
            last_image = ax.imshow(
                np.abs(field_data).T,
                cmap="viridis",
                origin="lower",
                extent=[0, self.x_range * 1e3, 0, self.y_range * 1e3],
                alpha=0.9,
                zorder=1,
            )
            ax.set_title(title)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")

        for j in range(len(selected), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(rf"Mode {mode + 1}: $n_{{eff}}$ = {self.neff[mode]:.4g}", fontsize=16)
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
                self._add_boundary_background(ax)
                ax.imshow(
                    np.abs(field).T,
                    cmap="viridis",
                    origin="lower",
                    extent=[0, self.x_range * 1e3, 0, self.y_range * 1e3],
                    vmin=0,
                    vmax=1,
                    alpha=0.9,
                    zorder=1,
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

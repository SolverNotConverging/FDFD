import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags, eye, kron
from scipy.sparse.linalg import eigs


class PeriodicModeSolver2D:
    """2D Bloch-periodic TE/TM mode solver on a compact periodic Yee grid.

    The structure varies in transverse ``x`` and periodic ``z``.  User
    materials live on cell centers with shape ``(Nx, Nz)`` and are averaged to
    component-specific Yee locations before each solve.
    """

    def __init__(
            self,
            polarization,
            freq,
            x_range,
            z_range,
            Nx,
            Nz,
            num_modes,
            guess=5,
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
        if self.Nx <= 0 or self.Nz <= 0:
            raise ValueError("Nx and Nz must be positive.")

        self.dx = x_range / self.Nx
        self.dz = z_range / self.Nz
        self.num_modes = int(num_modes)
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive.")

        self.epsilon0 = 8.854187817e-12
        self.mu0 = 4e-7 * np.pi
        self.c = 1 / np.sqrt(self.epsilon0 * self.mu0)
        self.omega = 2 * np.pi * freq
        self.k0 = self.omega / self.c

        self.guess = guess
        self.tol = tol
        self.ncv = ncv

        self.shape_cell = (self.Nx, self.Nz)
        self.shape_ex = (self.Nx, self.Nz)
        self.shape_ey = (self.Nx + 1, self.Nz)
        self.shape_ez = (self.Nx + 1, self.Nz)
        self.shape_hx = (self.Nx + 1, self.Nz)
        self.shape_hy = (self.Nx, self.Nz)
        self.shape_hz = (self.Nx, self.Nz)

        self.cell_eps_r_xx = np.ones(self.shape_cell, dtype=complex)
        self.cell_eps_r_yy = np.ones(self.shape_cell, dtype=complex)
        self.cell_eps_r_zz = np.ones(self.shape_cell, dtype=complex)
        self.cell_mu_r_xx = np.ones(self.shape_cell, dtype=complex)
        self.cell_mu_r_yy = np.ones(self.shape_cell, dtype=complex)
        self.cell_mu_r_zz = np.ones(self.shape_cell, dtype=complex)
        self.material_no_average_mask = np.zeros(self.shape_cell, dtype=bool)

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
        self._invalidate_solution()

    def _invalidate_solution(self):
        self.eigenvalues = None
        self.eigenvectors = None
        self.neff = None
        self.propagation_constant = None
        self.attenuation_constant = None
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
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

    @staticmethod
    def _validate_subpixels(subpixels):
        subpixels = int(subpixels)
        if subpixels <= 0:
            raise ValueError("subpixels must be positive.")
        return subpixels

    def _cell_material_array(self, prefix, component):
        if prefix == "eps":
            return {"xx": self.cell_eps_r_xx, "yy": self.cell_eps_r_yy, "zz": self.cell_eps_r_zz}[component]
        if prefix == "mu":
            return {"xx": self.cell_mu_r_xx, "yy": self.cell_mu_r_yy, "zz": self.cell_mu_r_zz}[component]
        raise ValueError(f"Unknown {prefix} component {component!r}.")

    def _component_mask(self, prefix, component):
        if prefix == "pec":
            return {"xx": self.pec_xx_mask, "yy": self.pec_yy_mask, "zz": self.pec_zz_mask}[component]
        if prefix == "pmc":
            return {"xx": self.pmc_xx_mask, "yy": self.pmc_yy_mask, "zz": self.pmc_zz_mask}[component]
        raise ValueError(f"Unknown {prefix} component {component!r}.")

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

        covered = fraction > 0.0
        if not np.any(covered):
            return

        for component, value in zip(("xx", "yy", "zz"), epsilon):
            target = self._cell_material_array("eps", component)[sl_x, sl_z]
            target[covered] = target[covered] * (1.0 - fraction[covered]) + value * fraction[covered]
        for component, value in zip(("xx", "yy", "zz"), mu):
            target = self._cell_material_array("mu", component)[sl_x, sl_z]
            target[covered] = target[covered] * (1.0 - fraction[covered]) + value * fraction[covered]

        local_no_average = self.material_no_average_mask[sl_x, sl_z]
        local_no_average[covered] = False
        self.update_component_materials()
        self._invalidate_solution()

    def add_rectangle(self, epsilon, mu, x_range, z_range, *, subpixels=8):
        """Add a rectangular isotropic or diagonal-anisotropic material region."""
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

    def add_pec(self, x_range, z_range, components=None):
        """Add a PEC cell region and expand it onto surrounding electric components."""
        sl_x, sl_z = self._region_slices(x_range, z_range)
        self._pec_regions.append((sl_x, sl_z))
        cell_mask = np.zeros(self.shape_cell, dtype=bool)
        cell_mask[sl_x, sl_z] = True
        xx_mask, yy_mask, zz_mask = self.component_masks_from_cell_mask(cell_mask, field="electric")
        selected = ("xx", "yy", "zz") if components is None else self._validate_components(components)
        for comp, mask in (("xx", xx_mask), ("yy", yy_mask), ("zz", zz_mask)):
            if comp in selected:
                self._component_mask("pec", comp)[:] |= mask
        self.update_component_materials()
        self._invalidate_solution()

    def add_pmc(self, x_range, z_range, components=None):
        """Add a PMC cell region and expand it onto surrounding magnetic components."""
        sl_x, sl_z = self._region_slices(x_range, z_range)
        self._pmc_regions.append((sl_x, sl_z))
        cell_mask = np.zeros(self.shape_cell, dtype=bool)
        cell_mask[sl_x, sl_z] = True
        xx_mask, yy_mask, zz_mask = self.component_masks_from_cell_mask(cell_mask, field="magnetic")
        selected = ("xx", "yy", "zz") if components is None else self._validate_components(components)
        for comp, mask in (("xx", xx_mask), ("yy", yy_mask), ("zz", zz_mask)):
            if comp in selected:
                self._component_mask("pmc", comp)[:] |= mask
        self.update_component_materials()
        self._invalidate_solution()

    def add_PMC(self, x_range, z_range, components=None):
        """Compatibility alias for add_pmc()."""
        self.add_pmc(x_range, z_range, components=components)

    def component_masks_from_cell_mask(self, cell_mask, field="electric"):
        """Expand a cell mask to compact-periodic Yee component locations."""
        mask = np.asarray(cell_mask, dtype=bool)
        if mask.shape != self.shape_cell:
            raise ValueError(f"cell_mask must have shape {self.shape_cell}.")
        ii, jj = np.nonzero(mask)
        jp = (jj + 1) % self.Nz

        if field == "electric":
            xx_mask = np.zeros(self.shape_ex, dtype=bool)
            yy_mask = np.zeros(self.shape_ey, dtype=bool)
            zz_mask = np.zeros(self.shape_ez, dtype=bool)
            xx_mask[ii, jj] = True
            xx_mask[ii, jp] = True
            yy_mask[ii, jj] = True
            yy_mask[ii + 1, jj] = True
            yy_mask[ii, jp] = True
            yy_mask[ii + 1, jp] = True
            zz_mask[ii, jj] = True
            zz_mask[ii + 1, jj] = True
            return xx_mask, yy_mask, zz_mask

        if field == "magnetic":
            xx_mask = np.zeros(self.shape_hx, dtype=bool)
            yy_mask = np.zeros(self.shape_hy, dtype=bool)
            zz_mask = np.zeros(self.shape_hz, dtype=bool)
            xx_mask[ii, jj] = True
            xx_mask[ii + 1, jj] = True
            yy_mask[ii, jj] = True
            zz_mask[ii, jj] = True
            zz_mask[ii, jp] = True
            return xx_mask, yy_mask, zz_mask

        raise ValueError("field must be 'electric' or 'magnetic'.")

    def add_pml(self, pml_width=30, n=3, sigma_max=5.0, direction="all"):
        """Add a simple x-directed uniaxial PML on the cell material tensors."""
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

    def add_UPML(self, pml_width=30, n=3, sigma_max=5.0, direction="all"):
        """Backward-compatible alias for add_pml()."""
        self.add_pml(pml_width=pml_width, n=n, sigma_max=sigma_max, direction=direction)

    @staticmethod
    def _flat(values):
        return np.asarray(values).ravel(order="F")

    @staticmethod
    def _flat_index(i, j, nx):
        return i + j * nx

    def _difference_matrix_x_node_to_cell(self, nz):
        rows = []
        cols = []
        data = []
        in_nx = self.Nx + 1
        out_nx = self.Nx
        for j in range(nz):
            for i in range(self.Nx):
                row = self._flat_index(i, j, out_nx)
                rows.extend((row, row))
                cols.extend((self._flat_index(i + 1, j, in_nx), self._flat_index(i, j, in_nx)))
                data.extend((1.0 / self.dx, -1.0 / self.dx))
        return coo_matrix((data, (rows, cols)), shape=(out_nx * nz, in_nx * nz)).tocsr()

    def _init_operators(self):
        values = np.ones(self.Nz)
        dz_node_to_cell = coo_matrix(
            (
                np.ravel(np.column_stack((-values, values))),
                (
                    np.repeat(np.arange(self.Nz), 2),
                    np.ravel(np.column_stack((np.arange(self.Nz), (np.arange(self.Nz) + 1) % self.Nz))),
                ),
            ),
            shape=(self.Nz, self.Nz),
        ).tocsr() / self.dz

        az_node_to_cell = coo_matrix(
            (
                0.5 * np.ones(2 * self.Nz),
                (
                    np.repeat(np.arange(self.Nz), 2),
                    np.ravel(np.column_stack((np.arange(self.Nz), (np.arange(self.Nz) + 1) % self.Nz))),
                ),
            ),
            shape=(self.Nz, self.Nz),
        ).tocsr()

        ix_cell = eye(self.Nx, format="csr")
        ix_node = eye(self.Nx + 1, format="csr")

        self.DZ_EX_TO_HY = kron(dz_node_to_cell, ix_cell, format="csr")
        self.AZ_EX_TO_HY = kron(az_node_to_cell, ix_cell, format="csr")
        self.DZ_HY_TO_EX = -self.DZ_EX_TO_HY.conj().T
        self.AZ_HY_TO_EX = self.AZ_EX_TO_HY.conj().T

        self.DZ_EY_TO_HX = kron(dz_node_to_cell, ix_node, format="csr")
        self.AZ_EY_TO_HX = kron(az_node_to_cell, ix_node, format="csr")
        self.DZ_HX_TO_EY = -self.DZ_EY_TO_HX.conj().T
        self.AZ_HX_TO_EY = self.AZ_EY_TO_HX.conj().T

        self.DX_EZ_TO_HY = self._difference_matrix_x_node_to_cell(self.Nz)
        self.DX_HY_TO_EZ = -self.DX_EZ_TO_HY.conj().T
        self.DX_EY_TO_HZ = self._difference_matrix_x_node_to_cell(self.Nz)
        self.DX_HZ_TO_EY = -self.DX_EY_TO_HZ.conj().T

    def _average_to_x_node(self, values, no_average_mask=None):
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

    def _average_to_z_node(self, values, no_average_mask=None):
        out = 0.5 * (values + np.roll(values, 1, axis=1))
        if no_average_mask is not None:
            ii, jj = np.nonzero(no_average_mask)
            out[ii, jj] = values[ii, jj]
            out[ii, (jj + 1) % self.Nz] = values[ii, jj]
        return out

    def _average_to_xz_node(self, values, no_average_mask=None):
        out = np.zeros((self.Nx + 1, self.Nz), dtype=complex)
        counts = np.zeros((self.Nx + 1, self.Nz), dtype=float)
        for z_shift in (0, 1):
            target_z = (np.arange(self.Nz) + z_shift) % self.Nz
            out[:self.Nx, target_z] += values
            counts[:self.Nx, target_z] += 1
            out[1:, target_z] += values
            counts[1:, target_z] += 1
        out = out / counts
        if no_average_mask is not None:
            ii, jj = np.nonzero(no_average_mask)
            jp = (jj + 1) % self.Nz
            out[ii, jj] = values[ii, jj]
            out[ii + 1, jj] = values[ii, jj]
            out[ii, jp] = values[ii, jj]
            out[ii + 1, jp] = values[ii, jj]
        return out

    def _material_on_fields(self, eps_r_xx, eps_r_yy, eps_r_zz, mu_r_xx, mu_r_yy, mu_r_zz, no_average_mask):
        return {
            "eps_xx": self._average_to_z_node(eps_r_xx, no_average_mask),
            "eps_yy": self._average_to_xz_node(eps_r_yy, no_average_mask),
            "eps_zz": self._average_to_x_node(eps_r_zz, no_average_mask),
            "mu_xx": self._average_to_x_node(mu_r_xx, no_average_mask),
            "mu_yy": mu_r_yy.copy(),
            "mu_zz": self._average_to_z_node(mu_r_zz, no_average_mask),
        }

    def _set_component_materials(self, materials):
        self.eps_r_xx = materials["eps_xx"].copy()
        self.eps_r_yy = materials["eps_yy"].copy()
        self.eps_r_zz = materials["eps_zz"].copy()
        self.mu_r_xx = materials["mu_xx"].copy()
        self.mu_r_yy = materials["mu_yy"].copy()
        self.mu_r_zz = materials["mu_zz"].copy()

    def update_component_materials(self):
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

    @staticmethod
    def _diag(values):
        return diags(np.asarray(values).ravel(order="F"), format="csr")

    def _inverse_diag_on_free(self, values, constrained_mask):
        values = np.asarray(values, dtype=complex)
        inverse = np.zeros_like(values, dtype=complex)
        free = ~constrained_mask
        inverse[free] = 1.0 / values[free]
        return diags(inverse.ravel(order="F"), format="csr")

    def _build_tm_system(self, materials, pec_xx_mask, pec_zz_mask, pmc_yy_mask):
        eps_xx_diag = self._diag(materials["eps_xx"])
        mu_yy_diag = self._diag(materials["mu_yy"])
        eps_zz_inv = self._inverse_diag_on_free(materials["eps_zz"], pec_zz_mask)

        n_ex = int(np.prod(self.shape_ex))
        n_hy = int(np.prod(self.shape_hy))
        zero_1 = csr_matrix((n_hy, n_hy), dtype=complex)
        zero_2 = csr_matrix((n_ex, n_ex), dtype=complex)

        d1 = 1j * self.omega * self.mu0 * mu_yy_diag
        d1 += 1j / self.omega * self.DX_EZ_TO_HY @ (1 / self.epsilon0 * eps_zz_inv @ self.DX_HY_TO_EZ)
        d2 = 1j * self.omega * self.epsilon0 * eps_xx_diag

        a = bmat([[self.DZ_EX_TO_HY, d1], [d2, self.DZ_HY_TO_EX]], format="csr")
        b = bmat([[self.AZ_EX_TO_HY, zero_1], [zero_2, self.AZ_HY_TO_EX]], format="csr")
        free = np.concatenate((~self._flat(pec_xx_mask), ~self._flat(pmc_yy_mask)))
        return a, b, free

    def _build_te_system(self, materials, pec_yy_mask, pmc_xx_mask, pmc_zz_mask):
        eps_yy_diag = self._diag(materials["eps_yy"])
        mu_xx_diag = self._diag(materials["mu_xx"])
        mu_zz_inv = self._inverse_diag_on_free(materials["mu_zz"], pmc_zz_mask)

        n_hx = int(np.prod(self.shape_hx))
        n_ey = int(np.prod(self.shape_ey))
        zero_1 = csr_matrix((n_ey, n_ey), dtype=complex)
        zero_2 = csr_matrix((n_hx, n_hx), dtype=complex)

        d1 = -1j * self.omega * self.epsilon0 * eps_yy_diag
        d1 -= 1j / self.omega * self.DX_HZ_TO_EY @ (1 / self.mu0 * mu_zz_inv @ self.DX_EY_TO_HZ)
        d2 = -1j * self.omega * self.mu0 * mu_xx_diag

        a = bmat([[self.DZ_HX_TO_EY, d1], [d2, self.DZ_EY_TO_HX]], format="csr")
        b = bmat([[self.AZ_HX_TO_EY, zero_1], [zero_2, self.AZ_EY_TO_HX]], format="csr")
        free = np.concatenate((~self._flat(pmc_xx_mask), ~self._flat(pec_yy_mask)))
        return a, b, free

    def solve(self, guess=None, tol=None, ncv=None):
        """Solve Bloch-periodic modes on the compact Yee grid."""
        guess = self.guess if guess is None else guess
        tol = self.tol if tol is None else tol
        ncv = self.ncv if ncv is None else ncv

        materials, pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask = (
            self._effective_materials_and_masks()
        )

        if self.polarization == "TM":
            a, b, free = self._build_tm_system(materials, pec_xx_mask, pec_zz_mask, pmc_yy_mask)
        else:
            a, b, free = self._build_te_system(materials, pec_yy_mask, pmc_xx_mask, pmc_zz_mask)

        if np.count_nonzero(free) <= self.num_modes:
            raise ValueError(f"Not enough unconstrained DOFs to solve {self.num_modes} modes.")

        a_reduced = a[free, :][:, free]
        b_reduced = b[free, :][:, free]
        v0 = np.ones(a_reduced.shape[0], dtype=complex)
        eigenvalues, eigenvectors_reduced = eigs(
            a_reduced,
            M=b_reduced,
            k=self.num_modes,
            sigma=guess,
            tol=tol,
            ncv=ncv,
            v0=v0,
        )
        order = np.argsort(np.abs(eigenvalues - guess))
        eigenvalues = eigenvalues[order]
        eigenvectors_reduced = eigenvectors_reduced[:, order]

        eigenvectors = np.zeros((a.shape[1], self.num_modes), dtype=complex)
        eigenvectors[free, :] = eigenvectors_reduced

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.neff = eigenvalues / self.k0
        self.propagation_constant = np.imag(self.neff)
        self.attenuation_constant = np.real(self.neff)

        if self.polarization == "TM":
            self._store_tm_fields(eigenvectors, materials, pec_xx_mask, pec_zz_mask, pmc_yy_mask)
        else:
            self._store_te_fields(eigenvectors, materials, pec_yy_mask, pmc_xx_mask, pmc_zz_mask)

        self._rotate_modes_to_most_real()

    def _reshape_modes(self, values, shape):
        return values.reshape((*shape, self.num_modes), order="F")

    def _store_tm_fields(self, eigenvectors, materials, pec_xx_mask, pec_zz_mask, pmc_yy_mask):
        n_ex = int(np.prod(self.shape_ex))
        ex_flat = eigenvectors[:n_ex, :]
        hy_flat = eigenvectors[n_ex:, :]
        eps_zz_inv = self._inverse_diag_on_free(materials["eps_zz"], pec_zz_mask)
        ez_flat = (-1j / (self.omega * self.epsilon0)) * (eps_zz_inv @ (self.DX_HY_TO_EZ @ hy_flat))

        self.Ex = self._reshape_modes(ex_flat, self.shape_ex)
        self.Hy = self._reshape_modes(hy_flat, self.shape_hy)
        self.Ez = self._reshape_modes(ez_flat, self.shape_ez)
        self.Ex[pec_xx_mask, :] = 0.0
        self.Ez[pec_zz_mask, :] = 0.0
        self.Hy[pmc_yy_mask, :] = 0.0

    def _store_te_fields(self, eigenvectors, materials, pec_yy_mask, pmc_xx_mask, pmc_zz_mask):
        n_hx = int(np.prod(self.shape_hx))
        hx_flat = eigenvectors[:n_hx, :]
        ey_flat = eigenvectors[n_hx:, :]
        mu_zz_inv = self._inverse_diag_on_free(materials["mu_zz"], pmc_zz_mask)
        hz_flat = (1j / (self.omega * self.mu0)) * (mu_zz_inv @ (self.DX_EY_TO_HZ @ ey_flat))

        self.Hx = self._reshape_modes(hx_flat, self.shape_hx)
        self.Ey = self._reshape_modes(ey_flat, self.shape_ey)
        self.Hz = self._reshape_modes(hz_flat, self.shape_hz)
        self.Hx[pmc_xx_mask, :] = 0.0
        self.Hz[pmc_zz_mask, :] = 0.0
        self.Ey[pec_yy_mask, :] = 0.0

    @staticmethod
    def _most_real_phase(*fields):
        values = np.concatenate([np.asarray(field).ravel() for field in fields])
        finite = np.isfinite(values)
        values = values[finite]
        if values.size == 0 or np.max(np.abs(values)) == 0:
            return 1.0 + 0j
        return np.exp(-0.5j * np.angle(np.sum(values ** 2)))

    def _rotate_modes_to_most_real(self):
        if self.polarization == "TM":
            for mode in range(self.num_modes):
                phase = self._most_real_phase(self.Ex[:, :, mode], self.Ez[:, :, mode], self.Hy[:, :, mode])
                self.Ex[:, :, mode] *= phase
                self.Ez[:, :, mode] *= phase
                self.Hy[:, :, mode] *= phase
                self.eigenvectors[:, mode] *= phase
        else:
            for mode in range(self.num_modes):
                phase = self._most_real_phase(self.Ey[:, :, mode], self.Hx[:, :, mode], self.Hz[:, :, mode])
                self.Ey[:, :, mode] *= phase
                self.Hx[:, :, mode] *= phase
                self.Hz[:, :, mode] *= phase
                self.eigenvectors[:, mode] *= phase

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
        from matplotlib.patches import Rectangle

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

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        root = tk.Tk()
        root.title(f"{self.polarization} Periodic Yee Mode Viewer")
        if sys.platform == "darwin":
            root.tk.call("tk", "scaling", 1.0)

        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        controls_frame = tk.Frame(root)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=10)

        fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), dpi=100)
        colorbars = []
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def plot_mode(selected_mode):
            mode = int(selected_mode) - 1
            if self.polarization == "TM":
                field_data = [
                    self._material_plot_data(),
                    self.Ex[:, :, mode],
                    self.Ez[:, :, mode],
                    self.Hy[:, :, mode],
                ]
                titles = [r"Structure (Abs($\epsilon$))", "Ex", "Ez", "Hy"]
            else:
                field_data = [
                    self._material_plot_data(),
                    self.Ey[:, :, mode],
                    self.Hx[:, :, mode],
                    self.Hz[:, :, mode],
                ]
                titles = [r"Structure (Abs($\epsilon$))", "Ey", "Hx", "Hz"]

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
                    aspect="auto",
                    extent=[0, self.z_range * 1e3, 0, self.x_range * 1e3],
                )
                # self._add_boundary_rectangles(ax)
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

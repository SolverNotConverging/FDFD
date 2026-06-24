import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigs


class PeriodicModeSolver3D:
    def __init__(self, Nx, Ny, Nz, x_range, y_range, z_range, freq, num_modes, sigma_guess=None, tol=0, ncv=None):
        # Store parameters
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Nz = int(Nz)
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.dx = x_range / self.Nx
        self.dy = y_range / self.Ny
        self.dz = z_range / self.Nz
        self.N = self.Nx * self.Ny * self.Nz

        self.freq = freq
        self.omega = 2 * np.pi * freq
        self.k0 = self.omega / 3e8
        self.epsilon0 = 8.85e-12
        self.mu0 = 1.26e-6
        self.num_modes = int(num_modes)

        if sigma_guess is not None:
            self.sigma_guess = sigma_guess
        else:
            self.sigma_guess = 0

        self.tol = tol
        self.ncv = ncv

        self.shape_cell = (self.Nx, self.Ny, self.Nz)
        self.shape_ex = (self.Nx, self.Ny + 1, self.Nz)
        self.shape_ey = (self.Nx + 1, self.Ny, self.Nz)
        self.shape_ez = (self.Nx + 1, self.Ny + 1, self.Nz)
        self.shape_hx = self.shape_ey
        self.shape_hy = self.shape_ex
        self.shape_hz = self.shape_cell

        self.n_ex = int(np.prod(self.shape_ex))
        self.n_ey = int(np.prod(self.shape_ey))
        self.n_ez = int(np.prod(self.shape_ez))
        self.n_hx = self.n_ey
        self.n_hy = self.n_ex
        self.n_hz = self.N

        # Material arrays
        self.cell_Erxx_3D = np.ones(self.shape_cell, dtype=complex)
        self.cell_Eryy_3D = np.ones(self.shape_cell, dtype=complex)
        self.cell_Erzz_3D = np.ones(self.shape_cell, dtype=complex)
        self.cell_Mrxx_3D = np.ones(self.shape_cell, dtype=complex)
        self.cell_Mryy_3D = np.ones(self.shape_cell, dtype=complex)
        self.cell_Mrzz_3D = np.ones(self.shape_cell, dtype=complex)
        self.material_no_average_mask = np.zeros(self.shape_cell, dtype=bool)

        self.Erxx_3D = np.ones(self.shape_ex, dtype=complex)
        self.Eryy_3D = np.ones(self.shape_ey, dtype=complex)
        self.Erzz_3D = np.ones(self.shape_ez, dtype=complex)
        self.Mrxx_3D = np.ones(self.shape_hx, dtype=complex)
        self.Mryy_3D = np.ones(self.shape_hy, dtype=complex)
        self.Mrzz_3D = np.ones(self.shape_hz, dtype=complex)

        self._pec_regions = []
        self._pmc_regions = []
        self._init_operators()
        self.update_component_materials()

        # Storage
        self.fields = {}
        self.eigenvalues = None
        self.eigenvectors = None

    def _invalidate_solution(self):
        self.fields = {}
        self.eigenvalues = None
        self.eigenvectors = None
        if hasattr(self, "gammas"):
            delattr(self, "gammas")

    # --- Differentiation operators
    def _init_operators(self):
        self.DEX_EZ_TO_EX = self._difference_matrix_x(self.shape_ez, self.shape_ex, self.dx, forward=True)
        self.DEY_EZ_TO_EY = self._difference_matrix_y(self.shape_ez, self.shape_ey, self.dy, forward=True)
        self.DEX_EY_TO_HZ = self._difference_matrix_x(self.shape_ey, self.shape_hz, self.dx, forward=True)
        self.DEY_EX_TO_HZ = self._difference_matrix_y(self.shape_ex, self.shape_hz, self.dy, forward=True)

        self.DHX_HY_TO_EZ = -self.DEX_EZ_TO_EX.conj().T
        self.DHY_HX_TO_EZ = -self.DEY_EZ_TO_EY.conj().T
        self.DHX_HZ_TO_HX = -self.DEX_EY_TO_HZ.conj().T
        self.DHY_HZ_TO_HY = -self.DEY_EX_TO_HZ.conj().T

        self.DEZ_EX = self._periodic_difference_matrix_z(self.shape_ex, self.shape_ex, self.dz, forward=True)
        self.DEZ_EY = self._periodic_difference_matrix_z(self.shape_ey, self.shape_ey, self.dz, forward=True)
        self.DHZ_HX = -self.DEZ_EY.conj().T
        self.DHZ_HY = -self.DEZ_EX.conj().T

        self.AZ_EX = self._periodic_forward_average_z(self.shape_ex)
        self.AZ_EY = self._periodic_forward_average_z(self.shape_ey)
        self.AZ_HX = self.AZ_EY.conj().T
        self.AZ_HY = self.AZ_EX.conj().T

    @staticmethod
    def _flat_index(i, j, k, nx, ny):
        return i + nx * (j + ny * k)

    def _difference_matrix_x(self, in_shape, out_shape, scale, forward=True):
        rows = []
        cols = []
        data = []
        in_nx, in_ny, _ = in_shape
        out_nx, out_ny, out_nz = out_shape
        for k in range(out_nz):
            for j in range(out_ny):
                for i in range(out_nx):
                    row = self._flat_index(i, j, k, out_nx, out_ny)
                    entries = ((i + 1, j, k, 1.0), (i, j, k, -1.0)) if forward else (
                        (i, j, k, 1.0), (i - 1, j, k, -1.0)
                    )
                    for ci, cj, ck, value in entries:
                        if 0 <= ci < in_shape[0] and 0 <= cj < in_shape[1] and 0 <= ck < in_shape[2]:
                            rows.append(row)
                            cols.append(self._flat_index(ci, cj, ck, in_nx, in_ny))
                            data.append(value / scale)
        return coo_matrix((data, (rows, cols)), shape=(np.prod(out_shape), np.prod(in_shape))).tocsr()

    def _difference_matrix_y(self, in_shape, out_shape, scale, forward=True):
        rows = []
        cols = []
        data = []
        in_nx, in_ny, _ = in_shape
        out_nx, out_ny, out_nz = out_shape
        for k in range(out_nz):
            for j in range(out_ny):
                for i in range(out_nx):
                    row = self._flat_index(i, j, k, out_nx, out_ny)
                    entries = ((i, j + 1, k, 1.0), (i, j, k, -1.0)) if forward else (
                        (i, j, k, 1.0), (i, j - 1, k, -1.0)
                    )
                    for ci, cj, ck, value in entries:
                        if 0 <= ci < in_shape[0] and 0 <= cj < in_shape[1] and 0 <= ck < in_shape[2]:
                            rows.append(row)
                            cols.append(self._flat_index(ci, cj, ck, in_nx, in_ny))
                            data.append(value / scale)
        return coo_matrix((data, (rows, cols)), shape=(np.prod(out_shape), np.prod(in_shape))).tocsr()

    def _periodic_difference_matrix_z(self, in_shape, out_shape, scale, forward=True):
        if in_shape != out_shape:
            raise ValueError("Periodic z derivatives should not change Yee component shape.")
        rows = []
        cols = []
        data = []
        nx, ny, nz = in_shape
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    row = self._flat_index(i, j, k, nx, ny)
                    if forward:
                        entries = ((i, j, (k + 1) % nz, 1.0), (i, j, k, -1.0))
                    else:
                        entries = ((i, j, k, 1.0), (i, j, (k - 1) % nz, -1.0))
                    for ci, cj, ck, value in entries:
                        rows.append(row)
                        cols.append(self._flat_index(ci, cj, ck, nx, ny))
                        data.append(value / scale)
        return coo_matrix((data, (rows, cols)), shape=(np.prod(in_shape), np.prod(in_shape))).tocsr()

    def _periodic_forward_average_z(self, shape):
        rows = []
        cols = []
        data = []
        nx, ny, nz = shape
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    row = self._flat_index(i, j, k, nx, ny)
                    for ck in (k, (k + 1) % nz):
                        rows.append(row)
                        cols.append(self._flat_index(i, j, ck, nx, ny))
                        data.append(0.5)
        return coo_matrix((data, (rows, cols)), shape=(np.prod(shape), np.prod(shape))).tocsr()

    # --- Helper functions for modeling
    @staticmethod
    def _normalise_three(name, value):
        if np.isscalar(value):
            return np.full(3, value, dtype=complex)
        array = np.asarray(value, dtype=complex)
        if array.ndim == 1 and array.size == 3:
            return array
        raise ValueError(f"{name} must be a scalar or a length-3 1D array (xx, yy, zz).")

    @staticmethod
    def _validate_subpixels(subpixels):
        subpixels = int(subpixels)
        if subpixels <= 0:
            raise ValueError("subpixels must be positive.")
        return subpixels

    def _coordinate_to_length(self, value, axis):
        if isinstance(value, (int, np.integer)):
            step = {"x": self.dx, "y": self.dy, "z": self.dz}[axis]
            return int(value) * step
        if isinstance(value, (float, np.floating)):
            return float(value)
        raise ValueError("Coordinates must be int grid indices or float physical positions in metres.")

    def _range_to_lengths(self, name, values, axis):
        if isinstance(values, slice):
            start = 0 if values.start is None else values.start
            stop = {"x": self.Nx, "y": self.Ny, "z": self.Nz}[axis] if values.stop is None else values.stop
            if values.step not in (None, 1):
                raise ValueError(f"{name} slice step must be None or 1.")
            values = (start, stop)
        try:
            start, stop = values
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a slice or a (min, max) pair.")
        start = self._coordinate_to_length(start, axis)
        stop = self._coordinate_to_length(stop, axis)
        if stop <= start:
            raise ValueError(f"{name} must satisfy max > min.")
        limit = {"x": self.x_range, "y": self.y_range, "z": self.z_range}[axis]
        if start < 0 or stop > limit:
            raise ValueError(f"{name} is out of bounds of the simulation grid.")
        return start, stop

    def _region_slices(self, x_range, y_range, z_range):
        x_min, x_max = self._range_to_lengths("x_range", x_range, "x")
        y_min, y_max = self._range_to_lengths("y_range", y_range, "y")
        z_min, z_max = self._range_to_lengths("z_range", z_range, "z")
        x0, x1, y0, y1, z0, z1 = self._clipped_cell_bbox(x_min, x_max, y_min, y_max, z_min, z_max)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            raise ValueError("Region is outside the simulation grid.")
        return slice(x0, x1), slice(y0, y1), slice(z0, z1)

    def _cell_material_array(self, prefix, component):
        if prefix == "eps":
            return {"xx": self.cell_Erxx_3D, "yy": self.cell_Eryy_3D, "zz": self.cell_Erzz_3D}[component]
        if prefix == "mu":
            return {"xx": self.cell_Mrxx_3D, "yy": self.cell_Mryy_3D, "zz": self.cell_Mrzz_3D}[component]
        raise ValueError(f"Unknown {prefix} component {component!r}.")

    def _subpixel_axis(self, start, stop, step, subpixels):
        indices = np.arange(start, stop, dtype=float)
        offsets = (np.arange(subpixels, dtype=float) + 0.5) / subpixels
        return (indices[:, None] + offsets[None, :]) * step

    def _clipped_cell_bbox(self, xmin, xmax, ymin, ymax, zmin, zmax):
        x0 = max(0, int(np.floor(xmin / self.dx)))
        x1 = min(self.Nx, int(np.ceil(xmax / self.dx)))
        y0 = max(0, int(np.floor(ymin / self.dy)))
        y1 = min(self.Ny, int(np.ceil(ymax / self.dy)))
        z0 = max(0, int(np.floor(zmin / self.dz)))
        z1 = min(self.Nz, int(np.ceil(zmax / self.dz)))
        return x0, x1, y0, y1, z0, z1

    def _apply_fractional_material(self, er, mr, fraction, sl_x, sl_y, sl_z):
        er = self._normalise_three("er", er)
        mr = self._normalise_three("mr", mr)
        fraction = np.asarray(fraction, dtype=float)
        if fraction.shape != self.cell_Erxx_3D[sl_x, sl_y, sl_z].shape:
            raise ValueError("fraction shape does not match target cell region.")

        no_average = self.material_no_average_mask[sl_x, sl_y, sl_z]
        covered = (fraction > 0.0) & ~no_average
        if not np.any(covered):
            return

        for component, value in zip(("xx", "yy", "zz"), er):
            target = self._cell_material_array("eps", component)[sl_x, sl_y, sl_z]
            target[covered] = target[covered] * (1.0 - fraction[covered]) + value * fraction[covered]
        for component, value in zip(("xx", "yy", "zz"), mr):
            target = self._cell_material_array("mu", component)[sl_x, sl_y, sl_z]
            target[covered] = target[covered] * (1.0 - fraction[covered]) + value * fraction[covered]

        self.update_component_materials()
        self._invalidate_solution()

    def add_block(self, er, mr, x_range, y_range, z_range, *, subpixels=8):
        """Add a subpixel-smoothed rectangular block on the cell material grid.

        Range bounds accept either integer grid indices or float physical
        positions in metres, matching ``PeriodicModeSolver2D.add_rectangle``.
        Python slices are also accepted for concise index-based regions.
        """
        x_min, x_max = self._range_to_lengths("x_range", x_range, "x")
        y_min, y_max = self._range_to_lengths("y_range", y_range, "y")
        z_min, z_max = self._range_to_lengths("z_range", z_range, "z")
        subpixels = self._validate_subpixels(subpixels)
        x0, x1, y0, y1, z0, z1 = self._clipped_cell_bbox(x_min, x_max, y_min, y_max, z_min, z_max)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            return

        xs = self._subpixel_axis(x0, x1, self.dx, subpixels)
        ys = self._subpixel_axis(y0, y1, self.dy, subpixels)
        zs = self._subpixel_axis(z0, z1, self.dz, subpixels)
        x_inside = (xs >= x_min) & (xs <= x_max)
        y_inside = (ys >= y_min) & (ys <= y_max)
        z_inside = (zs >= z_min) & (zs <= z_max)
        fraction = (
            x_inside[:, :, None, None, None, None]
            & y_inside[None, None, :, :, None, None]
            & z_inside[None, None, None, None, :, :]
        ).mean(axis=(1, 3, 5))
        self._apply_fractional_material(er, mr, fraction, slice(x0, x1), slice(y0, y1), slice(z0, z1))

    def add_pec(self, x_range, y_range, z_range, components=None, epsilon=1e8):
        sl_x, sl_y, sl_z = self._region_slices(x_range, y_range, z_range)
        selected = ("xx", "yy", "zz") if components is None else tuple(components)
        for comp in selected:
            self._cell_material_array("eps", comp)[sl_x, sl_y, sl_z] = epsilon
        self.material_no_average_mask[sl_x, sl_y, sl_z] = True
        self._pec_regions.append((sl_x, sl_y, sl_z))
        self.update_component_materials()
        self._invalidate_solution()

    def add_pmc(self, x_range, y_range, z_range, components=None, mu=1e8):
        sl_x, sl_y, sl_z = self._region_slices(x_range, y_range, z_range)
        selected = ("xx", "yy", "zz") if components is None else tuple(components)
        for comp in selected:
            self._cell_material_array("mu", comp)[sl_x, sl_y, sl_z] = mu
        self.material_no_average_mask[sl_x, sl_y, sl_z] = True
        self._pmc_regions.append((sl_x, sl_y, sl_z))
        self.update_component_materials()
        self._invalidate_solution()

    def add_UPML(self, sides=('-x', '+x', '-y', '+y'), width=10, max_loss=5, n=3):
        # Assumes e^{+i ω t}. If using e^{-i ω t}, change +1j -> -1j.
        omega = self.omega
        e0 = self.epsilon0

        nx = self.Nx
        ny = self.Ny
        assert width > 0
        assert width <= nx // 2 and width <= ny // 2, "PML width too large for grid."

        for i in range(width):
            sigma = max_loss * ((width - i) / width) ** n
            S = 1 + 1j * sigma / (omega * e0)  # flip sign if using e^{-iωt}

            if '-x' in sides:
                sl = np.s_[i, :, :]
                self.cell_Erxx_3D[sl] /= S
                self.cell_Eryy_3D[sl] *= S
                self.cell_Erzz_3D[sl] *= S
                self.cell_Mrxx_3D[sl] /= S
                self.cell_Mryy_3D[sl] *= S
                self.cell_Mrzz_3D[sl] *= S

            if '+x' in sides:
                sl = np.s_[-1 - i, :, :]
                self.cell_Erxx_3D[sl] /= S
                self.cell_Eryy_3D[sl] *= S
                self.cell_Erzz_3D[sl] *= S
                self.cell_Mrxx_3D[sl] /= S
                self.cell_Mryy_3D[sl] *= S
                self.cell_Mrzz_3D[sl] *= S

            if '+y' in sides:
                sl = np.s_[:, i, :]
                self.cell_Erxx_3D[sl] *= S
                self.cell_Eryy_3D[sl] /= S
                self.cell_Erzz_3D[sl] *= S
                self.cell_Mrxx_3D[sl] *= S
                self.cell_Mryy_3D[sl] /= S
                self.cell_Mrzz_3D[sl] *= S

            if '-y' in sides:
                sl = np.s_[:, -1 - i, :]
                self.cell_Erxx_3D[sl] *= S
                self.cell_Eryy_3D[sl] /= S
                self.cell_Erzz_3D[sl] *= S
                self.cell_Mrxx_3D[sl] *= S
                self.cell_Mryy_3D[sl] /= S
                self.cell_Mrzz_3D[sl] *= S

        self.update_component_materials()
        self._invalidate_solution()

    @staticmethod
    def _diag(values):
        return diags(values.ravel(order="F"), format="csr")

    def _average_x(self, values, no_average_mask=None):
        out = np.zeros((self.Nx + 1, self.Ny, self.Nz), dtype=complex)
        counts = np.zeros((self.Nx + 1, self.Ny, self.Nz), dtype=float)
        out[:self.Nx, :, :] += values
        counts[:self.Nx, :, :] += 1
        out[1:, :, :] += values
        counts[1:, :, :] += 1
        out = out / counts
        if no_average_mask is not None:
            ii, jj, kk = np.nonzero(no_average_mask)
            out[ii, jj, kk] = values[ii, jj, kk]
            out[ii + 1, jj, kk] = values[ii, jj, kk]
        return out

    def _average_y(self, values, no_average_mask=None):
        out = np.zeros((self.Nx, self.Ny + 1, self.Nz), dtype=complex)
        counts = np.zeros((self.Nx, self.Ny + 1, self.Nz), dtype=float)
        out[:, :self.Ny, :] += values
        counts[:, :self.Ny, :] += 1
        out[:, 1:, :] += values
        counts[:, 1:, :] += 1
        out = out / counts
        if no_average_mask is not None:
            ii, jj, kk = np.nonzero(no_average_mask)
            out[ii, jj, kk] = values[ii, jj, kk]
            out[ii, jj + 1, kk] = values[ii, jj, kk]
        return out

    def _average_xy(self, values, no_average_mask=None):
        out = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz), dtype=complex)
        counts = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz), dtype=float)
        out[:self.Nx, :self.Ny, :] += values
        counts[:self.Nx, :self.Ny, :] += 1
        out[1:, :self.Ny, :] += values
        counts[1:, :self.Ny, :] += 1
        out[:self.Nx, 1:, :] += values
        counts[:self.Nx, 1:, :] += 1
        out[1:, 1:, :] += values
        counts[1:, 1:, :] += 1
        out = out / counts
        if no_average_mask is not None:
            ii, jj, kk = np.nonzero(no_average_mask)
            out[ii, jj, kk] = values[ii, jj, kk]
            out[ii + 1, jj, kk] = values[ii, jj, kk]
            out[ii, jj + 1, kk] = values[ii, jj, kk]
            out[ii + 1, jj + 1, kk] = values[ii, jj, kk]
        return out

    def _material_on_fields(self, erxx, eryy, erzz, mrxx, mryy, mrzz, no_average_mask):
        return {
            "erxx": self._average_y(erxx, no_average_mask),
            "eryy": self._average_x(eryy, no_average_mask),
            "erzz": self._average_xy(erzz, no_average_mask),
            "mrxx": self._average_x(mrxx, no_average_mask),
            "mryy": self._average_y(mryy, no_average_mask),
            "mrzz": mrzz.copy(),
        }

    def _set_component_materials(self, materials):
        self.Erxx_3D = materials["erxx"].copy()
        self.Eryy_3D = materials["eryy"].copy()
        self.Erzz_3D = materials["erzz"].copy()
        self.Mrxx_3D = materials["mrxx"].copy()
        self.Mryy_3D = materials["mryy"].copy()
        self.Mrzz_3D = materials["mrzz"].copy()

    def update_component_materials(self):
        materials = self._material_on_fields(
            self.cell_Erxx_3D,
            self.cell_Eryy_3D,
            self.cell_Erzz_3D,
            self.cell_Mrxx_3D,
            self.cell_Mryy_3D,
            self.cell_Mrzz_3D,
            self.material_no_average_mask,
        )
        self._set_component_materials(materials)
        return materials

    # --- Solver
    def solve(self):
        omega, epsilon0, mu0 = self.omega, self.epsilon0, self.mu0

        # Build diagonal sparse matrices
        Erxx = self._diag(self.Erxx_3D)
        Eryy = self._diag(self.Eryy_3D)
        Erzz = self._diag(self.Erzz_3D)
        Mrxx = self._diag(self.Mrxx_3D)
        Mryy = self._diag(self.Mryy_3D)
        Mrzz = self._diag(self.Mrzz_3D)

        zero_ex_ey = csr_matrix((self.n_ex, self.n_ey), dtype=complex)
        zero_ey_ex = csr_matrix((self.n_ey, self.n_ex), dtype=complex)
        zero_hx_hy = csr_matrix((self.n_hx, self.n_hy), dtype=complex)
        zero_hy_hx = csr_matrix((self.n_hy, self.n_hx), dtype=complex)

        # Build system matrices
        A = bmat([
            [
                self.DEZ_EX,
                zero_ex_ey,
                self.DEX_EZ_TO_EX @ (-1j / (omega * epsilon0) * Erzz.power(-1) @ self.DHY_HX_TO_EZ),
                self.DEX_EZ_TO_EX @ (1j / (omega * epsilon0) * Erzz.power(-1) @ self.DHX_HY_TO_EZ)
                + 1j * omega * mu0 * Mryy,
            ],
            [
                zero_ey_ex,
                self.DEZ_EY,
                self.DEY_EZ_TO_EY @ (-1j / (omega * epsilon0) * Erzz.power(-1) @ self.DHY_HX_TO_EZ)
                - 1j * omega * mu0 * Mrxx,
                self.DEY_EZ_TO_EY @ (1j / (omega * epsilon0) * Erzz.power(-1) @ self.DHX_HY_TO_EZ),
            ],
            [
                self.DHX_HZ_TO_HX @ (1j / (omega * mu0) * Mrzz.power(-1) @ self.DEY_EX_TO_HZ),
                self.DHX_HZ_TO_HX @ (-1j / (omega * mu0) * Mrzz.power(-1) @ self.DEX_EY_TO_HZ)
                - 1j * omega * epsilon0 * Eryy,
                self.DHZ_HX,
                zero_hx_hy,
            ],
            [
                self.DHY_HZ_TO_HY @ (1j / (omega * mu0) * Mrzz.power(-1) @ self.DEY_EX_TO_HZ)
                + 1j * omega * epsilon0 * Erxx,
                self.DHY_HZ_TO_HY @ (-1j / (omega * mu0) * Mrzz.power(-1) @ self.DEX_EY_TO_HZ),
                zero_hy_hx,
                self.DHZ_HY,
            ],
        ]).tocsr()

        B = bmat([
            [self.AZ_EX, zero_ex_ey, None, None],
            [zero_ey_ex, self.AZ_EY, None, None],
            [None, None, self.AZ_HX, zero_hx_hy],
            [None, None, zero_hy_hx, self.AZ_HY],
        ], format="csr")

        # Solve
        self.eigenvalues, self.eigenvectors = eigs(A, M=B, k=self.num_modes, sigma=self.sigma_guess, tol=self.tol,
                                                   ncv=self.ncv)
        self.gammas = self.eigenvalues / self.k0
        self.store_fields()

    def store_fields(self):
        n0 = 0
        n1 = n0 + self.n_ex
        n2 = n1 + self.n_ey
        n3 = n2 + self.n_hx
        n4 = n3 + self.n_hy
        self.fields['Ex'] = np.array([
            self.eigenvectors[n0:n1, i].reshape(self.shape_ex, order="F") for i in range(self.num_modes)
        ])
        self.fields['Ey'] = np.array([
            self.eigenvectors[n1:n2, i].reshape(self.shape_ey, order="F") for i in range(self.num_modes)
        ])
        self.fields['Hx'] = np.array([
            self.eigenvectors[n2:n3, i].reshape(self.shape_hx, order="F") for i in range(self.num_modes)
        ])
        self.fields['Hy'] = np.array([
            self.eigenvectors[n3:n4, i].reshape(self.shape_hy, order="F") for i in range(self.num_modes)
        ])

    # --- Plotting
    def plot_field_plane(self, axis, index, mode_index=0, field='Ex'):
        field_data = np.abs(self.fields[field][mode_index])
        imdata, extent, xlabel, ylabel, _ = self._field_slice_plot_data(field_data, axis, index)
        plt.imshow(imdata, cmap='hot', origin='lower', extent=extent, aspect='auto')
        plt.colorbar()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'{field} at {axis}={index} | Mode {mode_index}')
        plt.show()

    def plot(self, mode=0, x=None, y=None, z=None, *, save=None, show=True):
        """
        Plot |Ex|, |Ey|, |Hx|, |Hy| for a 2D slice of the 3D field.

        Usage:
            self.plot(mode=0, x=3)  # slice at x_index=3
            self.plot(mode=1, y=5)  # slice at y_index=5
            self.plot(mode=2, z=7)  # slice at z_index=7

        Args:
            mode (int): mode index.
            x (int|None): x-slice index (plane is z–y).
            y (int|None): y-slice index (plane is z–x).
            z (int|None): z-slice index (plane is x–y).
            save (str|None): optional filepath to save the figure (e.g. "slice.png").
            show (bool): whether to call plt.show() at the end.

        Returns:
            matplotlib.figure.Figure
        """
        if self.eigenvalues is None or not self.fields:
            raise RuntimeError("Call solve() before plot().")

        # Determine axis/index from exactly one of x/y/z
        axes_specified = [(ax, idx) for ax, idx in (('x', x), ('y', y), ('z', z)) if idx is not None]
        if len(axes_specified) != 1:
            raise ValueError("Specify exactly one of x=, y=, or z= (e.g. plot(mode=0, x=3)).")

        axis, index = axes_specified[0]

        # Clamp mode
        mode = int(np.clip(mode, 0, self.num_modes - 1))

        # Make figure via existing helper
        fig = self._create_slice_figure(mode_index=mode, axis=axis, index=int(index))

        if save:
            fig.savefig(save, dpi=200, bbox_inches='tight')
        if show:
            plt.show()

        return fig

    def visualize_with_gui(self):
        if not hasattr(self, 'fields') or not self.fields or self.eigenvalues is None:
            raise RuntimeError("Call solve() before visualize_with_gui().")

        root = tk.Tk()
        root.title("3D Periodic Mode Viewer — Slice Explorer")
        if sys.platform == "darwin":
            root.tk.call("tk", "scaling", 1.0)

        def _configure_window():
            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()
            w = int(sw * 0.9)
            h = int(sh * 0.85)
            root.geometry(f"{w}x{h}")
            root.minsize(900, 600)
            return w, h

        _configure_window()

        # ==== Controls ====
        ctrl = ttk.Frame(root)
        ctrl.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Mode selector
        ttk.Label(ctrl, text="Mode:").grid(row=0, column=0, padx=(0, 6), sticky='w')
        mode_var = tk.IntVar(value=0)
        mode_dd = ttk.Combobox(ctrl, textvariable=mode_var, values=list(range(self.num_modes)), width=6,
                               state="readonly")
        mode_dd.grid(row=0, column=1, padx=(0, 12), sticky='w')

        # Axis selector
        ttk.Label(ctrl, text="Axis:").grid(row=0, column=2, padx=(0, 6), sticky='w')
        axis_var = tk.StringVar(value='z')
        rb_frame = ttk.Frame(ctrl)
        rb_frame.grid(row=0, column=3, padx=(0, 12), sticky='w')
        for ax in ('x', 'y', 'z'):
            ttk.Radiobutton(rb_frame, text=ax, value=ax, variable=axis_var).pack(side='left')

        # Index selector
        ttk.Label(ctrl, text="Index:").grid(row=0, column=4, padx=(0, 6), sticky='w')
        index_var = tk.IntVar(value=self.Nz // 2)  # default mid-plane for z
        idx_spin = ttk.Spinbox(ctrl, from_=0, to=max(self.Nx, self.Ny, self.Nz) - 1, textvariable=index_var, width=8)
        idx_spin.grid(row=0, column=5, padx=(0, 12), sticky='w')

        # Range hint label
        range_hint = ttk.Label(ctrl, text=f"(valid: 0 … {self.Nz - 1}) for axis z")
        range_hint.grid(row=0, column=6, sticky='w')

        # ==== Plot area ====
        plot_frame = ttk.Frame(root)
        plot_frame.grid(row=1, column=0, sticky="nsew")
        canvas = None

        def update_spin_range():
            ax = axis_var.get()
            if ax == 'x':
                max_idx = self.Nx - 1
            elif ax == 'y':
                max_idx = self.Ny - 1
            else:
                max_idx = self.Nz - 1
            # Update spin bounds and clamp current
            idx_spin.config(from_=0, to=max_idx)
            if index_var.get() > max_idx:
                index_var.set(max_idx)
            range_hint.config(text=f"(valid: 0 … {max_idx}) for axis {ax}")

        def redraw(*_):
            nonlocal canvas
            # Clean previous canvas
            if canvas:
                canvas.get_tk_widget().destroy()

            # Clamp mode/index for safety
            mode = max(0, min(self.num_modes - 1, mode_var.get()))
            update_spin_range()
            idx = index_var.get()

            fig = self._create_slice_figure(mode_index=mode, axis=axis_var.get(), index=idx)
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

        # Bind changes
        mode_dd.bind("<<ComboboxSelected>>", redraw)
        axis_var.trace_add('write', lambda *_: redraw())
        index_var.trace_add('write', lambda *_: redraw())

        # Initial setup
        update_spin_range()
        redraw()

        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        root.mainloop()

    def _slice_extent_labels(self, axis):
        """
        Returns (extent, xlabel, ylabel) for imshow depending on the slice axis.
        Units: mm.
        """
        if axis == 'x':
            # plane is (z, y)
            extent = [0, self.Nz * self.dz * 1e3, 0, self.Ny * self.dy * 1e3]
            return extent, 'z (mm)', 'y (mm)'
        elif axis == 'y':
            # plane is (z, x)
            extent = [0, self.Nz * self.dz * 1e3, 0, self.Nx * self.dx * 1e3]
            return extent, 'z (mm)', 'x (mm)'
        else:  # 'z'
            # plane is (x, y)
            extent = [0, self.Nx * self.dx * 1e3, 0, self.Ny * self.dy * 1e3]
            return extent, 'x (mm)', 'y (mm)'

    def _field_slice_plot_data(self, field_data, axis, index):
        if axis == 'x':
            index = int(np.clip(index, 0, field_data.shape[0] - 1))
            imdata = field_data[index, :, :]
            extent, xlabel, ylabel = self._slice_extent_labels(axis)
        elif axis == 'y':
            index = int(np.clip(index, 0, field_data.shape[1] - 1))
            imdata = field_data[:, index, :]
            extent, xlabel, ylabel = self._slice_extent_labels(axis)
        else:
            index = int(np.clip(index, 0, field_data.shape[2] - 1))
            imdata = field_data[:, :, index].T
            extent, xlabel, ylabel = self._slice_extent_labels(axis)
        return imdata, extent, xlabel, ylabel, index

    def _create_slice_figure(self, mode_index, axis, index):
        """
        Create a 2x2 figure of |Ex|, |Ey|, |Hx|, |Hy| for a 2D slice of the 3D fields.
        axis ∈ {'x','y','z'}, index is the slice index along that axis.
        """
        # Pull fields for the selected mode
        Ex = np.abs(self.fields['Ex'][mode_index])
        Ey = np.abs(self.fields['Ey'][mode_index])
        Hx = np.abs(self.fields['Hx'][mode_index])
        Hy = np.abs(self.fields['Hy'][mode_index])

        # Normalize safely (avoid division by zero)
        def norm(a):
            m = np.max(a)
            return a / m if m > 0 else a

        Ex, Ey, Hx, Hy = map(norm, (Ex, Ey, Hx, Hy))

        # Figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.25, hspace=0.25)

        panels = [
            ('Ex', Ex, axes[0, 0]),
            ('Ey', Ey, axes[0, 1]),
            ('Hx', Hx, axes[1, 0]),
            ('Hy', Hy, axes[1, 1]),
        ]

        plotted_index = int(index)
        for title, data3d, ax in panels:
            imdata, extent, xlabel, ylabel, plotted_index = self._field_slice_plot_data(data3d, axis, index)
            im = ax.imshow(imdata, cmap='viridis', origin='lower', extent=extent, aspect='auto')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Mode info
        beta = float(np.imag(self.gammas[mode_index]))
        alpha = float(np.real(self.gammas[mode_index]))
        fig.suptitle(f"Mode {mode_index} | Slice {axis}={plotted_index} | Beta={beta:.4f}, Alpha={alpha:.4f}", fontsize=12)

        # >>> Add these two lines <<<
        fig.tight_layout()
        fig.canvas.draw_idle()
        plt.close(fig)  # prevent accumulation of hidden figures

        return fig

    def _results_dict(self, include_eigenvectors=False):
        """
        Collect all calculated results and key inputs into a dict of plain numpy arrays
        suitable for saving. By default omits eigenvectors to keep files small.
        """
        if self.eigenvalues is None or not self.fields:
            raise RuntimeError("No results to store yet. Run solve() first.")

        out = dict(
            # grid + problem setup
            Nx=np.int64(self.Nx), Ny=np.int64(self.Ny), Nz=np.int64(self.Nz),
            x_range=np.float64(self.dx * self.Nx),
            y_range=np.float64(self.dy * self.Ny),
            z_range=np.float64(self.dz * self.Nz),
            freq=np.float64(self.freq),
            num_modes=np.int64(self.num_modes),
            epsilon0=np.float64(self.epsilon0),
            mu0=np.float64(self.mu0),
            sigma_guess=np.complex128(self.sigma_guess),
            tol=np.float64(self.tol),

            # materials
            cell_Erxx_3D=self.cell_Erxx_3D,
            cell_Eryy_3D=self.cell_Eryy_3D,
            cell_Erzz_3D=self.cell_Erzz_3D,
            cell_Mrxx_3D=self.cell_Mrxx_3D,
            cell_Mryy_3D=self.cell_Mryy_3D,
            cell_Mrzz_3D=self.cell_Mrzz_3D,
            material_no_average_mask=self.material_no_average_mask,

            # modal results
            eigenvalues=self.eigenvalues,
            gammas=self.gammas,
            Ex=self.fields['Ex'],
            Ey=self.fields['Ey'],
            Hx=self.fields['Hx'],
            Hy=self.fields['Hy'],
        )

        if include_eigenvectors and self.eigenvectors is not None:
            out['eigenvectors'] = self.eigenvectors

        return out

    def save_results(self, path, include_eigenvectors=False, compressed=True):
        """
        Save all calculated results + inputs to a single NPZ file.

        Args:
            path (str): e.g. 'results.npz'
            include_eigenvectors (bool): store the raw eigenvectors (large!).
            compressed (bool): use np.savez_compressed (default) or plain savez.
        """
        data = self._results_dict(include_eigenvectors=include_eigenvectors)
        if compressed:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)
        return path

    @classmethod
    def load_results(cls, path):
        """
        Recreate a solver instance from a saved NPZ (from save_results).
        Returns a fully populated solver ready for plotting.
        """
        with np.load(path, allow_pickle=False) as d:
            Nx = int(d['Nx'])
            Ny = int(d['Ny'])
            Nz = int(d['Nz'])
            x_range = float(d['x_range'])
            y_range = float(d['y_range'])
            z_range = float(d['z_range'])
            freq = float(d['freq'])
            num_modes = int(d['num_modes'])
            tol = float(d['tol']) if 'tol' in d else 0.0

            # Build instance
            inst = cls(Nx, Ny, Nz, x_range, y_range, z_range, freq, num_modes, tol=tol)

            # Restore constants/params that might differ
            inst.epsilon0 = float(d['epsilon0'])
            inst.mu0 = float(d['mu0'])
            inst.sigma_guess = complex(d['sigma_guess'])

            # Materials
            inst.cell_Erxx_3D = d['cell_Erxx_3D']
            inst.cell_Eryy_3D = d['cell_Eryy_3D']
            inst.cell_Erzz_3D = d['cell_Erzz_3D']
            inst.cell_Mrxx_3D = d['cell_Mrxx_3D']
            inst.cell_Mryy_3D = d['cell_Mryy_3D']
            inst.cell_Mrzz_3D = d['cell_Mrzz_3D']
            inst.material_no_average_mask = d['material_no_average_mask'].astype(bool)
            inst.update_component_materials()

            # Modal results
            inst.eigenvalues = d['eigenvalues']
            inst.gammas = d['gammas']

            # Fields dict (keep same shape/order)
            inst.fields = dict(
                Ex=d['Ex'],
                Ey=d['Ey'],
                Hx=d['Hx'],
                Hy=d['Hy'],
            )

            # Optional eigenvectors
            if 'eigenvectors' in d.files:
                inst.eigenvectors = d['eigenvectors']
            else:
                inst.eigenvectors = None

        return inst

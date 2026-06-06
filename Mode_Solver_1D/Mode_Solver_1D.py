import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigs


class ModeSolver1D:
    """1D FDFD mode solver on a true staggered Yee grid."""

    def __init__(self, frequency, x_range, Nx, num_modes, guess=None):
        self.frequency = frequency
        self.x_range = x_range
        self.Nx = int(Nx)
        if self.Nx <= 0:
            raise ValueError("Nx must be positive.")

        self.dx = x_range / self.Nx
        self.epsilon0 = 8.854187817e-12
        self.mu0 = 4e-7 * np.pi
        self.c = 1 / np.sqrt(self.epsilon0 * self.mu0)
        self.k_0 = 2 * np.pi * frequency / self.c
        self.dx_normalized = self.k_0 * self.dx

        self.shape_cell = (self.Nx,)
        self.shape_node = (self.Nx + 1,)

        self.cell_eps_r_xx = np.ones(self.shape_cell, dtype=complex)
        self.cell_eps_r_yy = np.ones(self.shape_cell, dtype=complex)
        self.cell_eps_r_zz = np.ones(self.shape_cell, dtype=complex)
        self.cell_mu_r_xx = np.ones(self.shape_cell, dtype=complex)
        self.cell_mu_r_yy = np.ones(self.shape_cell, dtype=complex)
        self.cell_mu_r_zz = np.ones(self.shape_cell, dtype=complex)
        self.material_no_average_mask = np.zeros(self.shape_cell, dtype=bool)

        self.eps_r_xx = np.ones(self.shape_cell, dtype=complex)
        self.eps_r_yy = np.ones(self.shape_node, dtype=complex)
        self.eps_r_zz = np.ones(self.shape_node, dtype=complex)
        self.mu_r_xx = np.ones(self.shape_node, dtype=complex)
        self.mu_r_yy = np.ones(self.shape_cell, dtype=complex)
        self.mu_r_zz = np.ones(self.shape_cell, dtype=complex)

        self.pec_xx_mask = np.zeros(self.shape_cell, dtype=bool)
        self.pec_yy_mask = np.zeros(self.shape_node, dtype=bool)
        self.pec_zz_mask = np.zeros(self.shape_node, dtype=bool)
        self.pmc_xx_mask = np.zeros(self.shape_node, dtype=bool)
        self.pmc_yy_mask = np.zeros(self.shape_cell, dtype=bool)
        self.pmc_zz_mask = np.zeros(self.shape_cell, dtype=bool)

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
            for arr in (
                self.cell_eps_r_xx,
                self.cell_eps_r_yy,
                self.cell_eps_r_zz,
                self.cell_mu_r_xx,
                self.cell_mu_r_yy,
                self.cell_mu_r_zz,
            )
        )

    def _resolve_eigs_guess(self, sigma):
        if sigma is not None:
            return sigma
        if self._auto_guess:
            self.guess = self._default_guess()
        return self.guess

    def _invalidate_solution(self):
        self.eigenvalues_TE = None
        self.eigenvalues_TM = None
        self.eigenvectors_TE = None
        self.eigenvectors_TM = None
        self.neff_TE = None
        self.neff_TM = None
        self.propagation_constant_TE = None
        self.propagation_constant_TM = None
        self.attenuation_constant_TE = None
        self.attenuation_constant_TM = None
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

    def _bound_to_index(self, value):
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return int(round(float(value) / self.dx))
        raise ValueError("Region bounds must be int grid indices or float physical positions in metres.")

    def _region_slice(self, x_range):
        try:
            x0 = self._bound_to_index(x_range[0])
            x1 = self._bound_to_index(x_range[1])
        except (TypeError, IndexError):
            raise ValueError("x_range must be a (min, max) pair.")

        if not x1 > x0:
            raise ValueError("x_range must satisfy max > min.")
        if not 0 <= x0 < x1 <= self.Nx:
            raise ValueError("Region is out of bounds of the simulation grid.")
        return slice(x0, x1)

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

    def add_layer(self, epsilon, mu, x_range, *, average=True):
        """Add an isotropic or diagonal-anisotropic material region on the cell grid."""
        sl_x = self._region_slice(x_range)
        epsilon = self._normalise_three("epsilon", epsilon)
        mu = self._normalise_three("mu", mu)

        self.cell_eps_r_xx[sl_x] = epsilon[0]
        self.cell_eps_r_yy[sl_x] = epsilon[1]
        self.cell_eps_r_zz[sl_x] = epsilon[2]
        self.cell_mu_r_xx[sl_x] = mu[0]
        self.cell_mu_r_yy[sl_x] = mu[1]
        self.cell_mu_r_zz[sl_x] = mu[2]
        self.material_no_average_mask[sl_x] = not average
        self.update_component_materials()
        self._invalidate_solution()

    def add_pec(self, x_range, components=None):
        """Add a PEC cell region and expand it onto surrounding electric components."""
        sl_x = self._region_slice(x_range)
        cell_mask = np.zeros(self.shape_cell, dtype=bool)
        cell_mask[sl_x] = True
        xx_mask, yy_mask, zz_mask = self.component_masks_from_cell_mask(cell_mask, field="electric")
        selected = ("xx", "yy", "zz") if components is None else self._validate_components(components)
        for comp, mask in (("xx", xx_mask), ("yy", yy_mask), ("zz", zz_mask)):
            if comp in selected:
                self._component_mask("pec", comp)[:] |= mask
        self._effective_materials_and_masks()
        self._invalidate_solution()

    def add_pmc(self, x_range, components=None):
        """Add a PMC cell region and expand it onto surrounding magnetic components."""
        sl_x = self._region_slice(x_range)
        cell_mask = np.zeros(self.shape_cell, dtype=bool)
        cell_mask[sl_x] = True
        xx_mask, yy_mask, zz_mask = self.component_masks_from_cell_mask(cell_mask, field="magnetic")
        selected = ("xx", "yy", "zz") if components is None else self._validate_components(components)
        for comp, mask in (("xx", xx_mask), ("yy", yy_mask), ("zz", zz_mask)):
            if comp in selected:
                self._component_mask("pmc", comp)[:] |= mask
        self._effective_materials_and_masks()
        self._invalidate_solution()

    def component_masks_from_cell_mask(self, cell_mask, field="electric"):
        mask = np.asarray(cell_mask, dtype=bool)
        if mask.shape != self.shape_cell:
            raise ValueError(f"cell_mask must have shape {self.shape_cell}.")
        ii = np.nonzero(mask)[0]

        if field == "electric":
            xx_mask = mask.copy()
            yy_mask = np.zeros(self.shape_node, dtype=bool)
            zz_mask = np.zeros(self.shape_node, dtype=bool)
            yy_mask[ii] = True
            yy_mask[ii + 1] = True
            zz_mask[ii] = True
            zz_mask[ii + 1] = True
            return xx_mask, yy_mask, zz_mask

        if field == "magnetic":
            xx_mask = np.zeros(self.shape_node, dtype=bool)
            yy_mask = mask.copy()
            zz_mask = mask.copy()
            xx_mask[ii] = True
            xx_mask[ii + 1] = True
            return xx_mask, yy_mask, zz_mask

        raise ValueError("field must be 'electric' or 'magnetic'.")

    def add_pml(self, pml_width=50, n=3, sigma_max=25, direction="all"):
        """Add a simple uniaxial PML by stretching cell-centered epsilon and mu tensors."""
        pml_width = int(pml_width)
        if pml_width <= 0:
            raise ValueError("pml_width must be positive.")
        if direction not in ("x-", "x+", "x", "all"):
            raise ValueError("direction must be one of 'x-', 'x+', 'x', or 'all'.")

        sigma_x = np.zeros(self.Nx, dtype=float)
        if direction in ("x-", "x", "all"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[i] = sigma_max * ((pml_width - i) / pml_width) ** n
        if direction in ("x+", "x", "all"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[-i - 1] = sigma_max * ((pml_width - i) / pml_width) ** n

        omega = 2 * np.pi * self.frequency
        Sx = 1.0 + 1j * sigma_x / (self.epsilon0 * omega)

        self.cell_eps_r_xx *= 1 / Sx
        self.cell_eps_r_yy *= Sx
        self.cell_eps_r_zz *= Sx
        self.cell_mu_r_xx *= 1 / Sx
        self.cell_mu_r_yy *= Sx
        self.cell_mu_r_zz *= Sx
        self.update_component_materials()
        self._invalidate_solution()

    def add_impedance_surface(
            self,
            Zs: complex,
            position: float | int,
            *,
            thickness_cells: int = 1,
            eps_components=("xx", "yy", "zz"),
    ):
        thickness_cells = int(thickness_cells)
        if thickness_cells <= 0:
            raise ValueError("thickness_cells must be positive.")

        idx = self._bound_to_index(position)
        sl_x = self._region_slice((idx, idx + thickness_cells))
        thickness = thickness_cells * self.dx

        delta_eps = -1j / (2 * np.pi * self.frequency * self.epsilon0 * thickness * Zs)
        for comp in self._validate_components(eps_components):
            self._cell_material_array("eps", comp)[sl_x] += delta_eps
        self.update_component_materials()
        self._invalidate_solution()

    @staticmethod
    def _average_to_node(values, no_average_mask=None):
        out = np.zeros(values.size + 1, dtype=complex)
        counts = np.zeros(values.size + 1, dtype=float)
        out[:-1] += values
        counts[:-1] += 1
        out[1:] += values
        counts[1:] += 1
        out = out / counts
        if no_average_mask is not None:
            ii = np.nonzero(no_average_mask)[0]
            out[ii] = values[ii]
            out[ii + 1] = values[ii]
        return out

    def _material_on_fields(self, eps_r_xx, eps_r_yy, eps_r_zz, mu_r_xx, mu_r_yy, mu_r_zz, no_average_mask):
        return {
            "eps_xx": eps_r_xx.copy(),
            "eps_yy": self._average_to_node(eps_r_yy, no_average_mask),
            "eps_zz": self._average_to_node(eps_r_zz, no_average_mask),
            "mu_xx": self._average_to_node(mu_r_xx, no_average_mask),
            "mu_yy": mu_r_yy.copy(),
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

    def _yeeder1d(self):
        """Generate rectangular derivative matrices between node and cell locations."""
        rows = []
        cols = []
        data = []
        for i in range(self.Nx):
            rows.extend((i, i))
            cols.extend((i + 1, i))
            data.extend((1.0 / self.dx_normalized, -1.0 / self.dx_normalized))
        D_e_to_h = coo_matrix((data, (rows, cols)), shape=(self.Nx, self.Nx + 1)).tocsr()
        D_h_to_e = -D_e_to_h.conj().T
        self.DEX = D_e_to_h
        self.DHX = D_h_to_e
        return D_e_to_h, D_h_to_e

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
        return diags(np.asarray(values).ravel(), format="csr")

    def _inverse_diag_on_free(self, values, constrained_mask):
        inverse = np.zeros_like(values, dtype=complex)
        inverse[~constrained_mask] = 1.0 / values[~constrained_mask]
        return diags(inverse, format="csr")

    def _solve_reduced(self, Omega, free_mask, full_size, sigma):
        Omega = Omega[free_mask, :][:, free_mask]
        if Omega.shape[0] <= self.num_modes:
            raise ValueError(f"Not enough unconstrained DOFs ({Omega.shape[0]}) to solve {self.num_modes} modes.")
        eigenvalues, eigenvectors_reduced = eigs(Omega, k=self.num_modes, sigma=sigma)
        order = np.argsort(np.real(eigenvalues))
        eigenvalues = eigenvalues[order]
        eigenvectors_reduced = eigenvectors_reduced[:, order]
        eigenvectors = np.zeros((full_size, self.num_modes), dtype=complex)
        eigenvectors[free_mask, :] = eigenvectors_reduced
        return eigenvalues, eigenvectors

    def _zero_constrained_fields(self, pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask):
        self.Ex[pec_xx_mask, :] = 0.0
        self.Ey[pec_yy_mask, :] = 0.0
        self.Ez[pec_zz_mask, :] = 0.0
        self.Hx[pmc_xx_mask, :] = 0.0
        self.Hy[pmc_yy_mask, :] = 0.0
        self.Hz[pmc_zz_mask, :] = 0.0

    def solve(self, sigma=None):
        """Solve TE and TM slab modes and recover staggered field components."""
        sigma = self._resolve_eigs_guess(sigma)
        materials, pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask = (
            self._effective_materials_and_masks()
        )

        D_e_to_h, D_h_to_e = self._yeeder1d()
        eps_xx_diag = self._diag(materials["eps_xx"])
        eps_yy_diag = self._diag(materials["eps_yy"])
        mu_xx_diag = self._diag(materials["mu_xx"])
        mu_yy_diag = self._diag(materials["mu_yy"])
        eps_zz_inv = self._inverse_diag_on_free(materials["eps_zz"], pec_zz_mask)
        mu_zz_inv = self._inverse_diag_on_free(materials["mu_zz"], pmc_zz_mask)

        Omega_TE = -mu_xx_diag @ (D_h_to_e @ mu_zz_inv @ D_e_to_h + eps_yy_diag)
        Omega_TM = -eps_xx_diag @ (D_e_to_h @ eps_zz_inv @ D_h_to_e + mu_yy_diag)

        self.eigenvalues_TE, self.eigenvectors_TE = self._solve_reduced(
            Omega_TE, ~pec_yy_mask, self.Nx + 1, sigma
        )
        self.eigenvalues_TM, self.eigenvectors_TM = self._solve_reduced(
            Omega_TM, ~pmc_yy_mask, self.Nx, sigma
        )

        self.neff_TE = self._passive_positive_neff(-self.eigenvalues_TE)
        self.neff_TM = self._passive_positive_neff(-self.eigenvalues_TM)
        self.propagation_constant_TE = np.real(self.neff_TE)
        self.propagation_constant_TM = np.real(self.neff_TM)
        self.attenuation_constant_TE = np.imag(self.neff_TE)
        self.attenuation_constant_TM = np.imag(self.neff_TM)

        self.Ey = np.asarray(self.eigenvectors_TE, dtype=complex)
        self.Hy = np.asarray(self.eigenvectors_TM, dtype=complex)
        self.Hx = np.zeros_like(self.Ey)
        self.Hz = np.asarray(mu_zz_inv @ (D_e_to_h @ self.Ey), dtype=complex)
        self.Ex = np.zeros_like(self.Hy)
        self.Ez = np.asarray(eps_zz_inv @ (D_h_to_e @ self.Hy), dtype=complex)

        for mode in range(self.num_modes):
            self.Hx[:, mode] = self.neff_TE[mode] * (1.0 / materials["mu_xx"]) * self.Ey[:, mode]
            self.Ex[:, mode] = self.neff_TM[mode] * (1.0 / materials["eps_xx"]) * self.Hy[:, mode]

        self._zero_constrained_fields(
            pec_xx_mask,
            pec_yy_mask,
            pec_zz_mask,
            pmc_xx_mask,
            pmc_yy_mask,
            pmc_zz_mask,
        )

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

    def _field_x(self, field_name):
        if field_name in ("Ey", "Ez", "Hx"):
            return np.linspace(0, self.x_range * 1e3, self.Nx + 1)
        return (np.arange(self.Nx) + 0.5) * self.dx * 1e3

    def _component_fields_for_mode(self, mode):
        return {
            "ey": (self.Ey[:, mode], "Ey", "TE"),
            "hx": (self.Hx[:, mode], "Hx", "TE"),
            "hz": (self.Hz[:, mode], "Hz", "TE"),
            "hy": (self.Hy[:, mode], "Hy", "TM"),
            "ex": (self.Ex[:, mode], "Ex", "TM"),
            "ez": (self.Ez[:, mode], "Ez", "TM"),
        }

    def _field_to_cells(self, name, data):
        if name in ("ey", "ez", "hx"):
            return 0.5 * (data[:-1] + data[1:])
        return data

    def visualize(self, mode=1, **kwargs):
        """Visualize selected field components for a given one-based mode index."""
        if self.neff_TE is None:
            raise RuntimeError("solve() must be called before visualize().")
        mode -= 1
        if not (0 <= mode < self.num_modes):
            raise ValueError("mode is out of range.")

        import matplotlib.pyplot as plt

        fields = self._component_fields_for_mode(mode)
        e_abs = np.sqrt(sum(np.abs(self._field_to_cells(key, fields[key][0])) ** 2 for key in ("ey", "ex", "ez")))
        h_abs = np.sqrt(sum(np.abs(self._field_to_cells(key, fields[key][0])) ** 2 for key in ("hx", "hy", "hz")))
        fields["eabs"] = (e_abs, "|E| cell-centered", "E")
        fields["habs"] = (h_abs, "|H| cell-centered", "H")

        selected = [key for key in fields if kwargs.get(key)]
        if not selected:
            selected = ["ey", "hx", "hz", "hy", "ex", "ez"]

        ncols = min(3, len(selected))
        nrows = int(np.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), layout="compressed")
        axes = np.array(axes).reshape(-1)
        x_cell = (np.arange(self.Nx) + 0.5) * self.dx * 1e3
        material = np.real(self.cell_eps_r_zz)
        material_norm = material / np.max(np.abs(material)) if np.max(np.abs(material)) > 0 else material

        for i, field_name in enumerate(selected):
            field_data, title, pol = fields[field_name]
            norm = np.max(np.abs(field_data))
            if norm > 0:
                field_data = field_data / norm
            x = x_cell if field_name in ("eabs", "habs") else self._field_x(title)
            ax = axes[i]
            ax.plot(x, np.real(field_data), label=f"Re({title})")
            ax.plot(x, np.abs(field_data), "--", label=f"|{title}|")
            ax.plot(x_cell, material_norm, color="0.75", alpha=0.6, label="cell eps_r_zz")
            ax.set_title(f"{pol}: {title}")
            ax.set_xlabel("x (mm)")
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)

        for j in range(len(selected), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(
            rf"Mode {mode + 1}: TE $n_{{eff}}$ = {self.neff_TE[mode]:.4g}, "
            rf"TM $n_{{eff}}$ = {self.neff_TM[mode]:.4g}",
            fontsize=14,
        )
        plt.show()

    def visualize_with_gui(self):
        """Launch an interactive Tk GUI to inspect mode profiles."""
        if self.neff_TE is None:
            raise RuntimeError("solve() must be called before visualize_with_gui().")

        import matplotlib.pyplot as plt

        root = tk.Tk()
        root.title("FDFD 1D Mode Visualizer")

        fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=100)
        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        controls_frame = tk.Frame(root)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=10)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        mode_var = tk.IntVar(value=1)
        ttk.Label(controls_frame, text="Select mode:").grid(row=0, column=0, padx=10, sticky="w")
        mode_menu = ttk.Combobox(
            controls_frame,
            textvariable=mode_var,
            values=list(range(1, self.num_modes + 1)),
            state="readonly",
            width=5,
        )
        mode_menu.grid(row=0, column=1, padx=10, sticky="w")
        tk.Button(controls_frame, text="Quit", command=root.destroy).grid(row=0, column=2, padx=10, sticky="e")

        def update_plots(event=None):
            mode = int(mode_var.get()) - 1
            fields = self._component_fields_for_mode(mode)
            for ax in axes.flat:
                ax.clear()

            for ax, key in zip(axes.flat, ("ey", "hx", "hz", "hy", "ex", "ez")):
                data, title, pol = fields[key]
                norm = np.max(np.abs(data))
                data = data / norm if norm > 0 else data
                ax.plot(self._field_x(title), np.real(data), label=f"Re({title})")
                ax.plot(self._field_x(title), np.abs(data), "--", label=f"|{title}|")
                ax.set_title(f"{pol}: {title}")
                ax.set_xlabel("x (mm)")
                ax.grid(True)
                ax.legend(loc="best", fontsize=8)

            fig.suptitle(
                rf"Mode {mode + 1}: TE $n_{{eff}}$ = {self.neff_TE[mode]:.4g}, "
                rf"TM $n_{{eff}}$ = {self.neff_TM[mode]:.4g}",
                fontsize=14,
            )
            canvas.draw_idle()

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        controls_frame.columnconfigure(2, weight=1)
        mode_menu.bind("<<ComboboxSelected>>", update_plots)
        update_plots()
        root.mainloop()

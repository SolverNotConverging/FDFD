import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from scipy.sparse import bmat, diags, eye, kron
from scipy.sparse.linalg import eigs


class PeriodicModeSolver2D:
    """2D Bloch-periodic TE/TM mode solver on an (Nx, Nz) grid."""

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

        shape = (self.Nx, self.Nz)
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

    def _init_operators(self):
        def diff_operator(n, pbc=False):
            values = np.ones(n)
            D = diags([-values, values], [0, 1], shape=(n, n)).tolil()
            if pbc:
                D[-1, 0] = 1
            return D.tocsr()

        def add_operator(n, pbc=False):
            values = np.ones(n)
            D = 0.5 * diags([values, values], [0, 1], shape=(n, n)).tolil()
            if pbc:
                D[-1, 0] = 0.5
            return D.tocsr()

        Ix = eye(self.Nx, format="csr")
        Iz = eye(self.Nz, format="csr")
        self.DEX = kron(Iz, diff_operator(self.Nx), format="csr") / self.dx
        self.DEZ = kron(diff_operator(self.Nz, pbc=True), Ix, format="csr") / self.dz
        self.DHX = -self.DEX.conj().T
        self.DHZ = -self.DEZ.conj().T
        self.D3 = kron(add_operator(self.Nz, pbc=True), Ix, format="csr")
        if self.polarization == "TE":
            self.D3 = self.D3.conj().T
        self.D4 = self.D3.conj().T

    def add_rectangle(self, epsilon, mu, x_range, z_range):
        """Add a rectangular isotropic or diagonal-anisotropic material region."""
        sl_x, sl_z = self._region_slices(x_range, z_range)
        epsilon = self._normalise_three("epsilon", epsilon)
        mu = self._normalise_three("mu", mu)

        self.eps_r_xx[sl_x, sl_z] = epsilon[0]
        self.eps_r_yy[sl_x, sl_z] = epsilon[1]
        self.eps_r_zz[sl_x, sl_z] = epsilon[2]
        self.mu_r_xx[sl_x, sl_z] = mu[0]
        self.mu_r_yy[sl_x, sl_z] = mu[1]
        self.mu_r_zz[sl_x, sl_z] = mu[2]
        self._invalidate_solution()

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
            self._component_array("eps", comp)[sl_x, sl_z] = epsilon
        self._pec_regions.append((sl_x, sl_z))
        self._invalidate_solution()

    def add_pmc(self, x_range, z_range, components=None, mu=1e8):
        """Add a PMC-like region using a large permeability penalty."""
        sl_x, sl_z = self._region_slices(x_range, z_range)
        if components is None:
            components = ("xx", "yy", "zz")
        for comp in self._validate_components(components):
            self._component_array("mu", comp)[sl_x, sl_z] = mu
        self._pmc_regions.append((sl_x, sl_z))
        self._invalidate_solution()

    def add_PMC(self, x_range, z_range, components=None, mu=1e8):
        """Compatibility alias for add_pmc()."""
        self.add_pmc(x_range, z_range, components=components, mu=mu)

    def add_pml(self, pml_width=30, n=3, sigma_max=5.0, direction="all"):
        """Add a simple x-directed uniaxial PML."""
        pml_width = int(pml_width)
        if pml_width <= 0:
            raise ValueError("pml_width must be positive.")
        if direction not in ("x-", "x+", "x", "all"):
            raise ValueError("direction must be one of 'x-', 'x+', 'x', or 'all'.")

        sigma_x = np.zeros((self.Nx, self.Nz), dtype=float)

        if direction in ("x-", "x", "all"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[i, :] = sigma_max * ((pml_width - i) / pml_width) ** n
        if direction in ("x+", "x", "all"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[-i - 1, :] = sigma_max * ((pml_width - i) / pml_width) ** n

        Sx = 1.0 + 1j * sigma_x / (self.epsilon0 * self.omega)
        self.eps_r_xx *= 1 / Sx
        self.eps_r_yy *= Sx
        self.eps_r_zz *= Sx
        self.mu_r_xx *= 1 / Sx
        self.mu_r_yy *= Sx
        self.mu_r_zz *= Sx
        self._invalidate_solution()

    def add_UPML(self, pml_width=30, n=3, sigma_max=5.0, direction="all"):
        """Compatibility alias for add_pml()."""
        self.add_pml(pml_width=pml_width, n=n, sigma_max=sigma_max, direction=direction)


    def _flat(self, values):
        return values.ravel(order="F")

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

    def _build_tm_system(self, eps_r_xx, eps_r_zz, mu_r_yy):
        eps_r_xx_diag = diags(self._flat(eps_r_xx))
        eps_r_zz_diag = diags(self._flat(eps_r_zz))
        mu_r_yy_diag = diags(self._flat(mu_r_yy))
        zero = diags(np.zeros(self.N))

        D1 = 1j * self.omega * self.mu0 * mu_r_yy_diag
        D1 += 1j / self.omega * self.DEX @ (1 / self.epsilon0 * eps_r_zz_diag.power(-1) @ self.DHX)
        D2 = 1j * self.omega * self.epsilon0 * eps_r_xx_diag
        A = bmat([[self.DEZ, D1], [D2, self.DHZ]], format="csr")
        B = bmat([[self.D3, zero], [zero, self.D4]], format="csr")
        return A, B

    def _build_te_system(self, eps_r_yy, mu_r_xx, mu_r_zz):
        eps_r_yy_diag = diags(self._flat(eps_r_yy))
        mu_r_xx_diag = diags(self._flat(mu_r_xx))
        mu_r_zz_diag = diags(self._flat(mu_r_zz))
        zero = diags(np.zeros(self.N))

        D1 = -1j * self.omega * self.epsilon0 * eps_r_yy_diag
        D1 -= 1j / self.omega * self.DHX @ (1 / self.mu0 * mu_r_zz_diag.power(-1) @ self.DEX)
        D2 = -1j * self.omega * self.mu0 * mu_r_xx_diag
        A = bmat([[self.DHZ, D1], [D2, self.DEZ]], format="csr")
        B = bmat([[self.D3, zero], [zero, self.D4]], format="csr")
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

        eigenvectors = np.zeros((2 * self.N, self.num_modes), dtype=complex)
        eigenvectors[free, :] = eigenvectors_reduced

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.neff = eigenvalues / self.k0
        self.propagation_constant = np.imag(self.neff)
        self.attenuation_constant = np.real(self.neff)

        if self.polarization == "TM":
            self.Ex = eigenvectors[:self.N, :]
            self.Hy = eigenvectors[self.N:, :]
        else:
            self.Hx = eigenvectors[:self.N, :]
            self.Ey = eigenvectors[self.N:, :]

        self.spurious_scores = np.zeros(self.num_modes, dtype=float)
        self.accepted_candidate_indices = np.arange(self.num_modes)
        self.rejected_candidate_indices = np.array([], dtype=int)
        self.unselected_candidate_indices = np.array([], dtype=int)

    def _field_array(self, field, mode):
        return field[:, mode].reshape((self.Nx, self.Nz), order="F")

    def _material_plot_data(self):
        data = np.abs(self.eps_r_yy).astype(float)
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
                field_data = [self._material_plot_data(), self._field_array(self.Ex, mode), self._field_array(self.Hy, mode)]
                titles = [r"Structure (Abs($\epsilon$))", "Ex (norm.)", "Hy (norm.)"]
            else:
                field_data = [self._material_plot_data(), self._field_array(self.Hx, mode), self._field_array(self.Ey, mode)]
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

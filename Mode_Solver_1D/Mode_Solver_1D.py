import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.sparse import diags
from scipy.sparse.linalg import eigs


class ModeSolver1D:
    """1D FDFD eigen-mode solver for slab waveguides."""

    def __init__(self, frequency, x_range, Nx, num_modes, guess=-15):
        self.frequency = frequency
        self.x_range = x_range
        self.Nx = int(Nx)
        self.dx = x_range / self.Nx
        self.epsilon0 = 8.85e-12
        self.mu0 = 1.26e-6
        self.c = 1 / np.sqrt(self.epsilon0 * self.mu0)
        self.k_0 = 2 * np.pi * frequency / self.c
        self.dx_normalized = self.k_0 * self.dx

        shape = (self.Nx,)
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
        self.guess = guess
        self._invalidate_solution()

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

        # Backward-compatible aliases for older example scripts.
        self.beta_TE = None
        self.beta_TM = None
        self.alpha_TE = None
        self.alpha_TM = None
        self.fields = {}

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

    def add_object(self, epsilon, mu, x_range):
        """Add an isotropic or diagonal-anisotropic material region."""
        sl_x = self._region_slice(x_range)
        epsilon = self._normalise_three("epsilon", epsilon)
        mu = self._normalise_three("mu", mu)

        self.eps_r_xx[sl_x] = epsilon[0]
        self.eps_r_yy[sl_x] = epsilon[1]
        self.eps_r_zz[sl_x] = epsilon[2]
        self.mu_r_xx[sl_x] = mu[0]
        self.mu_r_yy[sl_x] = mu[1]
        self.mu_r_zz[sl_x] = mu[2]
        self._invalidate_solution()

    def add_pec(self, x_range, components=None):
        """Add a PEC region for selected electric-field components."""
        sl_x = self._region_slice(x_range)
        if components is None:
            components = ("xx", "yy", "zz")
        for comp in self._validate_components(components):
            self._component_array("pec", comp)[sl_x] = True
        self._invalidate_solution()

    def add_pmc(self, x_range, components=None):
        """Add a PMC region for selected magnetic-field components."""
        sl_x = self._region_slice(x_range)
        if components is None:
            components = ("xx", "yy", "zz")
        for comp in self._validate_components(components):
            self._component_array("pmc", comp)[sl_x] = True
        self._invalidate_solution()

    def add_pml(self, pml_width=50, n=3, sigma_max=25, direction="both"):
        """Add a simple uniaxial PML by stretching epsilon and mu tensors."""
        pml_width = int(pml_width)
        if pml_width <= 0:
            raise ValueError("pml_width must be positive.")
        if direction not in ("x-", "x+", "x", "top", "bottom", "both"):
            raise ValueError("direction must be one of 'x-', 'x+', 'x', 'top', 'bottom', or 'both'.")

        sigma_x = np.zeros(self.Nx, dtype=float)
        if direction in ("x-", "x", "top", "both"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[i] = sigma_max * ((pml_width - i) / pml_width) ** n
        if direction in ("x+", "x", "bottom", "both"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[-i - 1] = sigma_max * ((pml_width - i) / pml_width) ** n

        eps0 = 8.854187817e-12
        omega = 2 * np.pi * self.frequency
        Sx = 1.0 + 1j * sigma_x / (eps0 * omega)

        self.eps_r_xx *= 1 / Sx
        self.eps_r_yy *= Sx
        self.eps_r_zz *= Sx
        self.mu_r_xx *= 1 / Sx
        self.mu_r_yy *= Sx
        self.mu_r_zz *= Sx
        self._invalidate_solution()

    def add_UPML(self, pml_width=50, n=3, sigma_max=25, direction="both"):
        """Backward-compatible alias for add_pml()."""
        self.add_pml(pml_width=pml_width, n=n, sigma_max=sigma_max, direction=direction)

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
        x_range = (idx, idx + thickness_cells)
        sl_x = self._region_slice(x_range)
        thickness = thickness_cells * self.dx

        eps0 = 8.854187817e-12
        delta_eps = -1j / (2 * np.pi * self.frequency * eps0 * thickness * Zs)
        for comp in self._validate_components(eps_components):
            self._component_array("eps", comp)[sl_x] += delta_eps
        self._invalidate_solution()

    def _yeeder1d(self):
        """Generate derivative matrices on a 1D Yee-style grid."""
        values = np.ones(self.Nx)
        D = diags([-values, values], [0, 1], shape=(self.Nx, self.Nx), format="csr")
        DEX = D / self.dx_normalized
        DHX = -DEX.conj().T
        return DEX, DHX

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
        inverse = np.zeros_like(values, dtype=complex)
        inverse[~constrained_mask] = 1.0 / values[~constrained_mask]
        return diags(inverse)

    def _solve_reduced(self, Omega, free_mask, sigma):
        Omega = Omega[free_mask, :][:, free_mask]
        if Omega.shape[0] <= self.num_modes:
            raise ValueError(
                f"Not enough unconstrained DOFs ({Omega.shape[0]}) to solve {self.num_modes} modes."
            )
        eigenvalues, eigenvectors_reduced = eigs(Omega, k=self.num_modes, sigma=sigma)
        order = np.argsort(np.real(eigenvalues))
        eigenvalues = eigenvalues[order]
        eigenvectors_reduced = eigenvectors_reduced[:, order]

        eigenvectors = np.zeros((self.Nx, self.num_modes), dtype=complex)
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
        """Solve TE and TM slab modes and recover field components."""
        if sigma is None:
            sigma = self.guess

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

        DEX, DHX = self._yeeder1d()
        eps_r_xx_diag = diags(eps_r_xx)
        eps_r_yy_diag = diags(eps_r_yy)
        mu_r_xx_diag = diags(mu_r_xx)
        mu_r_yy_diag = diags(mu_r_yy)
        eps_r_zz_inv = self._inverse_diag_on_free(eps_r_zz, pec_zz_mask)
        mu_r_zz_inv = self._inverse_diag_on_free(mu_r_zz, pmc_zz_mask)

        Omega_TE = -mu_r_xx_diag @ (DHX @ mu_r_zz_inv @ DEX + eps_r_yy_diag)
        Omega_TM = -eps_r_xx_diag @ (DEX @ eps_r_zz_inv @ DHX + mu_r_yy_diag)

        free_te = ~pec_yy_mask
        free_tm = ~pmc_yy_mask
        self.eigenvalues_TE, self.eigenvectors_TE = self._solve_reduced(Omega_TE, free_te, sigma)
        self.eigenvalues_TM, self.eigenvectors_TM = self._solve_reduced(Omega_TM, free_tm, sigma)

        self.neff_TE = self._passive_positive_neff(-self.eigenvalues_TE)
        self.neff_TM = self._passive_positive_neff(-self.eigenvalues_TM)
        self.propagation_constant_TE = np.real(self.neff_TE)
        self.propagation_constant_TM = np.real(self.neff_TM)
        self.attenuation_constant_TE = np.imag(self.neff_TE)
        self.attenuation_constant_TM = np.imag(self.neff_TM)

        self.Ey = np.asarray(self.eigenvectors_TE, dtype=complex)
        self.Hy = np.asarray(self.eigenvectors_TM, dtype=complex)
        self.Ex = np.zeros_like(self.Hy)
        self.Ez = np.zeros_like(self.Hy)
        self.Hx = np.zeros_like(self.Ey)
        self.Hz = np.zeros_like(self.Ey)

        for mode in range(self.num_modes):
            self.Hx[:, mode] = self.neff_TE[mode] * (1.0 / mu_r_xx) * self.Ey[:, mode]
            self.Hz[:, mode] = np.asarray(mu_r_zz_inv @ (DEX @ self.Ey[:, mode])).ravel()
            self.Ex[:, mode] = self.neff_TM[mode] * (1.0 / eps_r_xx) * self.Hy[:, mode]
            self.Ez[:, mode] = np.asarray(eps_r_zz_inv @ (DHX @ self.Hy[:, mode])).ravel()

        self._zero_constrained_fields(
            pec_xx_mask,
            pec_yy_mask,
            pec_zz_mask,
            pmc_xx_mask,
            pmc_yy_mask,
            pmc_zz_mask,
        )

        self.fields = {
            "TE": {"Ey": self.Ey, "Hx": self.Hx, "Hz": self.Hz},
            "TM": {"Hy": self.Hy, "Ex": self.Ex, "Ez": self.Ez},
        }

        # Legacy names now expose the dimensionless effective-index pieces.
        self.beta_TE = self.propagation_constant_TE
        self.beta_TM = self.propagation_constant_TM
        self.alpha_TE = self.attenuation_constant_TE
        self.alpha_TM = self.attenuation_constant_TM

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

    def visualize(self, mode=1, **kwargs):
        """Visualize selected field components for a given one-based mode index."""
        if not self.fields:
            raise RuntimeError("solve() must be called before visualize().")
        mode -= 1
        if not (0 <= mode < self.num_modes):
            raise ValueError("mode is out of range.")

        import matplotlib.pyplot as plt

        fields = {
            "ey": (self.Ey[:, mode], "Ey", "TE"),
            "hx": (self.Hx[:, mode], "Hx", "TE"),
            "hz": (self.Hz[:, mode], "Hz", "TE"),
            "hy": (self.Hy[:, mode], "Hy", "TM"),
            "ex": (self.Ex[:, mode], "Ex", "TM"),
            "ez": (self.Ez[:, mode], "Ez", "TM"),
        }
        e_abs = np.sqrt(np.abs(self.Ey[:, mode]) ** 2 + np.abs(self.Ex[:, mode]) ** 2 + np.abs(self.Ez[:, mode]) ** 2)
        h_abs = np.sqrt(np.abs(self.Hx[:, mode]) ** 2 + np.abs(self.Hy[:, mode]) ** 2 + np.abs(self.Hz[:, mode]) ** 2)
        fields["eabs"] = (e_abs, "|E|", "E")
        fields["habs"] = (h_abs, "|H|", "H")

        selected = [key for key in fields if kwargs.get(key)]
        if not selected:
            selected = ["ey", "hx", "hz", "hy", "ex", "ez"]

        x = np.linspace(0, self.x_range * 1e3, self.Nx)
        ncols = min(3, len(selected))
        nrows = int(np.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), layout="compressed")
        axes = np.array(axes).reshape(-1)

        material = np.real(self.eps_r_zz)
        material_norm = material / np.max(np.abs(material)) if np.max(np.abs(material)) > 0 else material
        for i, field_name in enumerate(selected):
            field_data, title, pol = fields[field_name]
            norm = np.max(np.abs(field_data))
            if norm > 0:
                field_data = field_data / norm
            ax = axes[i]
            ax.plot(x, np.real(field_data), label=f"Re({title})")
            ax.plot(x, np.abs(field_data), "--", label=f"|{title}|")
            ax.plot(x, material_norm, color="0.75", alpha=0.6, label="eps_r_zz")
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
        if not self.fields:
            raise RuntimeError("solve() must be called before visualize_with_gui().")

        import matplotlib.pyplot as plt
        import sys

        root = tk.Tk()
        root.title("FDFD 1D Mode Visualizer")
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

        win_w, win_h = _configure_window()
        dpi = 110
        fig_w = max(8.0, (win_w / dpi) * 0.95)
        fig_h = max(5.5, (win_h / dpi) * 0.7)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        gs = fig.add_gridspec(4, 3, height_ratios=[0.08, 1.0, 0.08, 1.0], hspace=0.5, wspace=0.3)
        header_te_ax = fig.add_subplot(gs[0, :])
        header_te_ax.axis("off")
        header_tm_ax = fig.add_subplot(gs[2, :])
        header_tm_ax.axis("off")

        axes = np.empty((2, 3), dtype=object)
        for c in range(3):
            axes[0, c] = fig.add_subplot(gs[1, c])
            axes[1, c] = fig.add_subplot(gs[3, c])
        field_map = [("TE", ["Ey", "Hx", "Hz"]), ("TM", ["Hy", "Ex", "Ez"])]
        lines = []

        for r, (pol, comps) in enumerate(field_map):
            for c, comp in enumerate(comps):
                ax = axes[r, c]
                ax.set_ylabel(comp)
                ax.set_title(f"{pol}: {comp}", pad=2)
                ax.grid(True)
                line, = ax.plot([], [], lw=2)
                lines.append(line)

        for ax in axes[-1, :]:
            ax.set_xlabel("x (mm)")

        te_info = header_te_ax.text(0.0, 0.5, "", fontsize=10, fontweight="bold", ha="left", va="center")
        tm_info = header_tm_ax.text(0.0, 0.5, "", fontsize=10, fontweight="bold", ha="left", va="center")

        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        controls_frame = tk.Frame(root)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=10)

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

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        controls_frame.columnconfigure(0, weight=0)
        controls_frame.columnconfigure(1, weight=0)
        controls_frame.columnconfigure(2, weight=1)

        def update_plots(event=None):
            idx = int(mode_var.get()) - 1
            x = np.linspace(0, self.x_range * 1e3, self.Nx)

            Ey = self.fields["TE"]["Ey"][:, idx].copy()
            Hx = self.fields["TE"]["Hx"][:, idx].copy()
            Hz = self.fields["TE"]["Hz"][:, idx].copy()
            Ey /= np.max(np.abs(Ey)) + 1e-12
            Hmag = np.sqrt(np.abs(Hx) ** 2 + np.abs(Hz) ** 2)
            norm_H = np.max(Hmag) + 1e-12
            Hx /= norm_H
            Hz /= norm_H

            Hy = self.fields["TM"]["Hy"][:, idx].copy()
            Ex = self.fields["TM"]["Ex"][:, idx].copy()
            Ez = self.fields["TM"]["Ez"][:, idx].copy()
            Hy /= np.max(np.abs(Hy)) + 1e-12
            Emag = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ez) ** 2)
            norm_E = np.max(Emag) + 1e-12
            Ex /= norm_E
            Ez /= norm_E

            values = [Ey.real, Hx.real, Hz.real, Hy.real, Ex.real, Ez.real]
            for line, y in zip(lines, values):
                line.set_data(x, y)
                ax = line.axes
                ax.relim()
                ax.autoscale_view()

            te_info.set_text(f"Mode {idx + 1}  |  TE n_eff = {self.neff_TE[idx]:.4g}")
            tm_info.set_text(f"Mode {idx + 1}  |  TM n_eff = {self.neff_TM[idx]:.4g}")
            canvas.draw_idle()

        update_plots()
        mode_menu.bind("<<ComboboxSelected>>", update_plots)
        root.mainloop()

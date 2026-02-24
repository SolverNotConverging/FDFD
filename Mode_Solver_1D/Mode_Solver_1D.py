import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

from yee_derivative import yeeder2d


class ModeSolver1D:
    """1‑D FDFD eigen‑mode solver for slab waveguides."""

    def __init__(self, frequency: float, x_range: float, Nx: int, num_modes: int, guess=-15):
        self.frequency = float(frequency)
        self.omega = 2 * np.pi * self.frequency
        self.c0 = 3.0e8  # speed of light [m/s]
        self.k0 = self.omega / self.c0  # free‑space wavenumber

        self.x_range = float(x_range)
        self.Nx = int(Nx)
        self.dx = self.x_range / self.Nx

        self.num_modes = int(num_modes)
        self.guess = guess

        # Material tensors (diagonal, anisotropic allowed)
        self.epsilon = {comp: np.ones(self.Nx, dtype=complex) for comp in ("xx", "yy", "zz")}
        self.mu = {comp: np.ones(self.Nx, dtype=complex) for comp in ("xx", "yy", "zz")}

        # Finite‑difference operators
        self._init_operators()
        self._clear_results()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _clear_results(self) -> None:
        self.beta_TE = self.alpha_TE = None
        self.beta_TM = self.alpha_TM = None
        self.fields: dict[str, dict[str, np.ndarray]] = {}

    def _init_operators(self) -> None:
        """Central‑difference first‑derivative matrices D and −Dᵀ."""

        NS = [self.Nx, 1]
        dx_normalised = self.k0 * self.dx
        RES = [dx_normalised, 1]
        BC = [0, 0]
        [D, _, DT, _] = yeeder2d(NS, RES, BC)
        self.D = D
        self.DT = DT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_object(self, epsilon, mu, x_range):
        # ---- helpers ----
        def _norm_three(name, val):
            # scalar -> [s, s, s]; length-3 1D -> as-is
            if np.isscalar(val):
                return np.full(3, val, dtype=complex)
            arr = np.asarray(val, dtype=complex)
            if arr.ndim == 1 and arr.size == 3:
                return arr
            raise ValueError(f"{name} must be a scalar or a length-3 1D array (xx, yy, zz).")

        # ---- ranges ----
        try:
            x0, x1 = int(x_range[0]), int(x_range[1])
        except Exception:
            raise ValueError("x_range must be (min, max) integer-like pairs.")

        if not x1 > x0:
            x1, x0 = x0, x1

        Nx = self.Nx
        if not 0 <= x0 < x1 <= Nx:
            raise ValueError("Region is out of bounds of the simulation grid.")

        # ---- materials ----
        epsilon = _norm_three("epsilon", epsilon)
        mu = _norm_three("mu", mu)

        # ---- assign ----
        self.epsilon['xx'][x0:x1] = epsilon[0]
        self.epsilon['yy'][x0:x1] = epsilon[1]
        self.epsilon['zz'][x0:x1] = epsilon[2]

        self.mu['xx'][x0:x1] = mu[0]
        self.mu['yy'][x0:x1] = mu[1]
        self.mu['zz'][x0:x1] = mu[2]

    # ------------------------------------------------------------------
    # Balanced surface‑impedance sheet (electric + magnetic)
    # ------------------------------------------------------------------
    def add_impedance_surface(
            self,
            Zs: complex,
            x: float | int,
            *,
            thickness_cells: int = 1,
            eps_components: tuple[str, ...] = ("xx", "yy", "zz"),
            mu_components: tuple[str, ...] = ("xx", "yy", "zz"),
    ):
        """
        Insert an impedance sheet whose admittance is split equally
        between ε‑ and µ‑perturbations so that TE and TM modes load
        identically.

        Parameters
        ----------
        Zs : complex
            Desired sheet impedance in ohms (may be complex).
        x : float | int
            Position of the sheet.  *float* → physical coordinate [m];
            *int* → grid cell index.
        thickness_cells : int, optional
            How many grid cells represent the sheet.  Default 1.
        eps_components, mu_components : tuple[str], optional
            Tensor components to perturb.  Leaving the defaults makes
            the sheet isotropic; supply e.g. ("yy",) if you need
            anisotropy.
        """
        # ------------------------------------------------------------------
        # 1) map the requested location to a grid index
        # ------------------------------------------------------------------
        if isinstance(x, float):
            idx_start = int(round(x / self.dx))
        else:
            idx_start = int(x)

        if not (0 <= idx_start < self.Nx):
            raise ValueError("Impedance surface is outside the simulation domain.")

        idx_stop = idx_start + thickness_cells
        if idx_stop > self.Nx:
            raise ValueError("thickness_cells runs beyond the right boundary.")

        sl = slice(idx_start, idx_stop)  # slice of cells holding the sheet
        t = thickness_cells * self.dx  # physical thickness [m]

        # ------------------------------------------------------------------
        # 2) convert sheet impedance → Δε  and  Δµ  (balanced split)
        # ------------------------------------------------------------------
        eps0 = 8.854187817e-12  # F/m

        delta_eps = -1j / (self.omega * eps0 * t * Zs)

        # ------------------------------------------------------------------
        # 3) write the perturbations into the chosen tensor components
        # ------------------------------------------------------------------
        for comp in eps_components:
            self.epsilon[comp][sl] += delta_eps

        # results are no longer valid
        self._clear_results()

    def add_UPML(self, pml_width: int = 50, n: int = 3, sigma_max: float = 25,
                 direction: str = "both", ):
        Nx = self.Nx
        sigma_x = np.zeros(Nx)

        # --- build σ profiles --------------------------------------------------
        if direction in ("top", "both"):
            for i in range(pml_width):
                prof = sigma_max * ((pml_width - i) / pml_width) ** n
                sigma_x[i] = prof  # top

        if direction in ("bottom", "both"):
            for i in range(pml_width):
                prof = sigma_max * ((pml_width - i) / pml_width) ** n
                sigma_x[-i - 1] = prof  # bottom

        eps0 = 8.854187817e-12  # F m⁻¹
        omega = 2 * np.pi * self.frequency

        self.Sx = 1.0 + 1j * sigma_x / (eps0 * omega)
        self.epsilon["xx"] *= 1 / self.Sx
        self.epsilon["yy"] *= self.Sx
        self.epsilon["zz"] *= self.Sx
        self.mu["xx"] *= 1 / self.Sx
        self.mu["yy"] *= self.Sx
        self.mu["zz"] *= self.Sx

    # ------------------------------------------------------------------

    def solve(self):
        """Compute the lowest‑order eigen‑modes (TE & TM)."""

        # Build diagonal sparse matrices for ε and μ
        eps = {k: diags(v) for k, v in self.epsilon.items()}
        mu = {k: diags(v) for k, v in self.mu.items()}

        # TE operator (Ey field)
        A_TE = self.DT @ mu["zz"].power(-1) @ self.D + eps["yy"]
        Omega_TE = -mu["xx"] @ A_TE

        # TM operator (Hy field)
        A_TM = self.D @ eps["zz"].power(-1) @ self.DT + mu["yy"]
        Omega_TM = -eps["xx"] @ A_TM

        # Solve eigen‑systems (shift‑invert for faster convergence)
        eigvals_TE, eigvecs_TE = eigs(Omega_TE, k=self.num_modes, sigma=self.guess)
        eigvals_TM, eigvecs_TM = eigs(Omega_TM, k=self.num_modes, sigma=self.guess)

        # Propagation constants: gamma = alpha + i*beta = sqrt(lambda)
        gamma_TE = np.sqrt(eigvals_TE)
        gamma_TM = np.sqrt(eigvals_TM)

        self.alpha_TE, self.beta_TE = abs(gamma_TE.real), abs(gamma_TE.imag)
        self.alpha_TM, self.beta_TM = abs(gamma_TM.real), abs(gamma_TM.imag)

        # Field components -------------------------------------------------
        Ey = eigvecs_TE  # TE primary field
        Hy = eigvecs_TM  # TM primary field

        # Allocate containers (each shape: (Nx, num_modes))
        TE = {"Ey": Ey, "Hx": np.zeros_like(Ey), "Hz": np.zeros_like(Ey)}
        TM = {"Hy": Hy, "Ex": np.zeros_like(Hy), "Ez": np.zeros_like(Hy)}

        mu_zz = self.mu["zz"]
        mu_xx = self.mu["xx"]
        eps_zz = self.epsilon["zz"]
        eps_xx = self.epsilon["xx"]

        # Reconstruct remaining components mode‑by‑mode
        for m in range(self.num_modes):
            TE["Hx"][:, m] = gamma_TE[m] * (1 / mu_xx) * Ey[:, m]
            TE["Hz"][:, m] = (1 / mu_zz) * (self.D @ Ey[:, m])

            TM["Ex"][:, m] = gamma_TM[m] * (1 / eps_xx) * Hy[:, m]
            TM["Ez"][:, m] = (1 / eps_zz) * (self.DT @ Hy[:, m])

        self.fields = {"TE": TE, "TM": TM}

    def visualize_with_gui(self):
        """Launch an interactive Tk GUI to inspect mode profiles."""
        if not self.fields:
            raise RuntimeError("Run solve_modes() before visualization.")

        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt
        import numpy as np

        root = tk.Tk()
        root.title("FDFD 1D Mode Visualizer")

        # Mode selector
        mode_var = tk.IntVar(value=1)
        mode_menu = ttk.Combobox(
            root,
            textvariable=mode_var,
            values=list(range(1, self.num_modes + 1)),
            state="readonly",
            width=5,
        )
        mode_menu.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(root, text="Select mode:").grid(row=0, column=0, padx=5, pady=5, sticky="e")

        # Info label for α and β
        info_label = ttk.Label(root, text="", font=("Segoe UI", 10, "bold"))
        info_label.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        # Set up figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=110, sharex=True)
        field_map = [("TE", ["Ey", "Hx", "Hz"]), ("TM", ["Hy", "Ex", "Ez"])]
        lines = []

        for r, (pol, comps) in enumerate(field_map):
            for c, comp in enumerate(comps):
                ax = axes[r, c]
                ax.set_ylabel(comp)
                ax.set_title(f"{pol}: {comp}")
                ax.grid(True)
                line, = ax.plot([], [], lw=2)
                lines.append(line)

        for ax in axes[-1, :]:
            ax.set_xlabel("x (mm)")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        # Allow dynamic resizing
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        root.rowconfigure(2, weight=1)

        def update_plots(event=None):
            idx = int(mode_var.get()) - 1
            x = np.linspace(0, self.x_range * 1e3, self.Nx)  # mm

            # Normalize fields per mode
            Ey = self.fields["TE"]["Ey"][:, idx]
            Hx = self.fields["TE"]["Hx"][:, idx]
            Hz = self.fields["TE"]["Hz"][:, idx]
            Ey /= np.max(np.abs(Ey)) + 1e-12
            Hmag = np.sqrt(np.abs(Hx) ** 2 + np.abs(Hz) ** 2)
            norm_H = np.max(Hmag) + 1e-12
            Hx /= norm_H
            Hz /= norm_H

            Hy = self.fields["TM"]["Hy"][:, idx]
            Ex = self.fields["TM"]["Ex"][:, idx]
            Ez = self.fields["TM"]["Ez"][:, idx]
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

            # Update info label with β, α values
            info_label.config(text=(
                f"Mode {idx + 1}:    "
                f"TE → β = {self.beta_TE[idx]:.4g}, α = {self.alpha_TE[idx]:.4g}| "
                f"TM → β = {self.beta_TM[idx]:.4g}, α = {self.alpha_TM[idx]:.4g}"
            ))

            canvas.draw_idle()

        update_plots()
        mode_menu.bind("<<ComboboxSelected>>", update_plots)

        root.mainloop()

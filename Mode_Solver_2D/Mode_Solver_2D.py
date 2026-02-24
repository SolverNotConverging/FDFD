import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.sparse import diags, bmat
from scipy.sparse.linalg import eigs

from yee_derivative import yeeder2d


class ModeSolver2D:
    def __init__(self, frequency, x_range, y_range, Nx, Ny, num_modes):
        self.frequency = frequency
        self.x_range = x_range
        self.y_range = y_range
        self.Nx = Nx
        self.Ny = Ny
        self.dx = x_range / Nx
        self.dy = y_range / Ny
        self.k_0 = 2 * np.pi * frequency / 3e8
        self.dx_normalized = self.k_0 * self.dx
        self.dy_normalized = self.k_0 * self.dy
        self.epsilon = {comp: np.ones((Ny, Nx), dtype=complex) for comp in ['xx', 'yy', 'zz']}
        self.mu = {comp: np.ones((Ny, Nx), dtype=complex) for comp in ['xx', 'yy', 'zz']}
        self.num_modes = num_modes
        self.propagation_constant = None
        self.attenuation_constant = None

    def add_object(self, epsilon, mu, x_range, y_range):
        """
        Add a rectangular region with isotropic or diagonal-anisotropic material.

        Parameters
        ----------
        epsilon : scalar (isotropic) or length-3 1D (xx, yy, zz)
        mu      : scalar (isotropic) or length-3 1D (xx, yy, zz)
        x_range : (x_min, x_max) indices [inclusive, exclusive)
        y_range : (y_min, y_max) indices [inclusive, exclusive)
        """

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
            y0, y1 = int(y_range[0]), int(y_range[1])
        except Exception:
            raise ValueError("x_range and y_range must be (min, max) integer-like pairs.")

        if not (x1 > x0 and y1 > y0):
            raise ValueError("x_range and y_range must satisfy max > min.")

        Ny, Nx = self.Ny, self.Nx
        if not (0 <= x0 < x1 <= Nx and 0 <= y0 < y1 <= Ny):
            raise ValueError("Region is out of bounds of the simulation grid.")

        # ---- materials ----
        epsilon = _norm_three("epsilon", epsilon)
        mu = _norm_three("mu", mu)

        # ---- assign ----
        self.epsilon['xx'][y0:y1, x0:x1] = epsilon[0]
        self.epsilon['yy'][y0:y1, x0:x1] = epsilon[1]
        self.epsilon['zz'][y0:y1, x0:x1] = epsilon[2]

        self.mu['xx'][y0:y1, x0:x1] = mu[0]
        self.mu['yy'][y0:y1, x0:x1] = mu[1]
        self.mu['zz'][y0:y1, x0:x1] = mu[2]

    def add_UPML(self, pml_width: int = 50, n: int = 3, sigma_max: float = 5, direction: str = "both", ):
        """
        Add uniaxial PML using polynomial conductivity profiles.

        Parameters
        ----------
        pml_width : int
            Thickness of the PML in grid points.
        n : int
            Polynomial order of the conductivity profile.
        sigma_max : float
            Maximum normalised conductivity (σ / (ε0 ω) at the inner PML edge).
        direction : {'x', 'y', 'both'}
            Which boundaries receive the PML.
        """

        Nx, Ny = self.Nx, self.Ny
        sigma_x = np.zeros((Ny, Nx))
        sigma_y = np.zeros((Ny, Nx))

        # --- build σ profiles --------------------------------------------------
        if direction in ("x-", "x", "both"):
            for i in range(pml_width):
                prof = sigma_max * ((pml_width - i) / pml_width) ** n
                sigma_x[:, i] = prof  # left

        if direction in ("x+", "x", "both"):
            for i in range(pml_width):
                prof = sigma_max * ((pml_width - i) / pml_width) ** n
                sigma_x[:, -i - 1] = prof  # right

        if direction in ("y-", "y", "both"):
            for i in range(pml_width):
                prof = sigma_max * ((pml_width - i) / pml_width) ** n
                sigma_y[i, :] = prof  # bottom

        if direction in ("y+", "y", "both"):
            for i in range(pml_width):
                prof = sigma_max * ((pml_width - i) / pml_width) ** n
                sigma_y[-i - 1, :] = prof  # top

        # --- stretch variables (κ = 1 everywhere) ------------------------------
        eps0 = 8.854187817e-12  # F m⁻¹
        omega = 2 * np.pi * self.frequency

        self.Sx = 1.0 + 1j * sigma_x / (eps0 * omega)
        self.Sy = 1.0 + 1j * sigma_y / (eps0 * omega)

        self.epsilon["xx"] *= self.Sy / self.Sx
        self.epsilon["yy"] *= self.Sx / self.Sy
        self.epsilon["zz"] *= self.Sx * self.Sy
        self.mu["xx"] *= self.Sy / self.Sx
        self.mu["yy"] *= self.Sx / self.Sy
        self.mu["zz"] *= self.Sx * self.Sy

    def add_impedance_surface(
            self,
            Zs: complex,
            position: float | int,
            *,
            orientation: str = "x",
            thickness_cells: int = 1,
            eps_components: tuple[str, ...] = ("xx", "yy", "zz"),
    ):
        """
        Parameters
        ----------
        Zs : complex
            Desired sheet impedance in ohms (may be complex).
        position : float | int
            • If *orientation* == 'x': x‑coordinate (m) or x‑index (int)
              of the *leftmost* cell that will hold the sheet.
            • If *orientation* == 'y': y‑coordinate (m) or y‑index (int)
              of the *bottom* cell that will hold the sheet.
        orientation : {'x', 'y'}, optional
            'x' → vertical sheet normal to x (constant‑x line)
            'y' → horizontal sheet normal to y (constant‑y line)
        thickness_cells : int, optional
            Number of grid cells that represent the sheet.  Default 1.
        eps_components, mu_components : tuple[str], optional
            Tensor components to perturb.  Leave defaults for isotropy,
            or supply e.g. ("yy",) to limit to one component.
        """
        # ──────────────────────────────────────────────────────────────────
        # 1) figure out the slice of cells that will receive the update
        # ──────────────────────────────────────────────────────────────────
        if orientation not in ("x", "y"):
            raise ValueError("orientation must be 'x' or 'y'.")

        if orientation == "x":
            if isinstance(position, float):
                idx = int(round(position / self.dx))
            else:
                idx = int(position)
            if not (0 <= idx < self.Nx):
                raise ValueError("Impedance surface lies outside the x‑domain.")
            sl_x = slice(idx, idx + thickness_cells)
            sl_y = slice(None)
            t = thickness_cells * self.dx  # physical thickness [m]

        else:  # orientation == "y"
            if isinstance(position, float):
                idy = int(round(position / self.dy))
            else:
                idy = int(position)
            if not (0 <= idy < self.Ny):
                raise ValueError("Impedance surface lies outside the y‑domain.")
            sl_x = slice(None)
            sl_y = slice(idy, idy + thickness_cells)
            t = thickness_cells * self.dy  # physical thickness [m]

        # guard against overrunning the grid
        if sl_x.stop is not None and sl_x.stop > self.Nx:
            raise ValueError("thickness_cells runs beyond the right boundary.")
        if sl_y.stop is not None and sl_y.stop > self.Ny:
            raise ValueError("thickness_cells runs beyond the top boundary.")

        # ──────────────────────────────────────────────────────────────────
        # 2) balanced conversion  (same formula as 1‑D version)
        # ──────────────────────────────────────────────────────────────────
        eps0 = 8.854187817e-12
        delta_eps = -1j / (2 * np.pi * self.frequency * eps0 * t * Zs)

        # ──────────────────────────────────────────────────────────────────
        # 3) write perturbations into ε and µ
        # ──────────────────────────────────────────────────────────────────
        for comp in eps_components:
            self.epsilon[comp][sl_y, sl_x] += delta_eps

        # ──────────────────────────────────────────────────────────────────
        # 4) invalidate previous eigen‑solution (if any)
        # ──────────────────────────────────────────────────────────────────
        self.propagation_constant = None
        self.attenuation_constant = None

    def solve(self):
        """Solve for the modes and calculate field components."""
        epsilon_diag = {comp: diags(self.epsilon[comp].flatten()) for comp in self.epsilon}
        mu_diag = {comp: diags(self.mu[comp].flatten()) for comp in self.mu}

        Dx_e, Dy_e, Dx_h, Dy_h = self._yeeder2d()
        epsilon_zz_inv = epsilon_diag['zz'].power(-1)
        P11 = Dx_e @ epsilon_zz_inv @ Dy_h
        P12 = -(Dx_e @ epsilon_zz_inv @ Dx_h + mu_diag['yy'])
        P21 = Dy_e @ epsilon_zz_inv @ Dy_h + mu_diag['xx']
        P22 = -Dy_e @ epsilon_zz_inv @ Dx_h
        P = bmat([[P11, P12], [P21, P22]])

        mu_zz_inv = mu_diag['zz'].power(-1)
        Q11 = Dx_h @ mu_zz_inv @ Dy_e
        Q12 = -(Dx_h @ mu_zz_inv @ Dx_e + epsilon_diag['yy'])
        Q21 = Dy_h @ mu_zz_inv @ Dy_e + epsilon_diag['xx']
        Q22 = -Dy_h @ mu_zz_inv @ Dx_e
        Q = bmat([[Q11, Q12], [Q21, Q22]])

        Omega = P @ Q
        self.eigenvalues, self.eigenvectors = eigs(Omega, k=self.num_modes, sigma=-13)

        self.gamma_tilda = -1j * self._sqrt_positive_real(self.eigenvalues)
        self.propagation_constant = np.real(self.gamma_tilda)
        self.attenuation_constant = np.imag(self.gamma_tilda)

        # Calculate fields
        Exy = self.eigenvectors
        Nx, Ny = self.Nx, self.Ny
        eigenvalues_inv = diags(np.sqrt(self.eigenvalues)).power(-1)
        self.Ex = Exy[: Nx * Ny, :]
        self.Ey = Exy[Nx * Ny:, :]
        Hxy = Q @ Exy @ eigenvalues_inv
        self.Hx = Hxy[: Nx * Ny, :]
        self.Hy = Hxy[Nx * Ny:, :]
        self.Ez = epsilon_diag['zz'].power(-1) @ (Dx_h @ self.Hy - Dy_h @ self.Hx)
        self.Hz = mu_diag['zz'].power(-1) @ (Dx_e @ self.Ey - Dy_e @ self.Ex)

    def visualize(self, mode=1, **kwargs):
        """Visualize selected field components for a given mode.

        Keyword arguments:
        Use any of the following to selectively plot components:
        ex=True, ey=True, ez=True, eabs=True, hx=True, hy=True, hz=True, habs=True
        """
        Nx, Ny = self.Nx, self.Ny
        mode -= 1

        # Reshape fields for visualization
        ex = self.Ex[:, mode].reshape(Ny, Nx)
        ey = self.Ey[:, mode].reshape(Ny, Nx)
        ez = self.Ez[:, mode].reshape(Ny, Nx)
        hx = self.Hx[:, mode].reshape(Ny, Nx)
        hy = self.Hy[:, mode].reshape(Ny, Nx)
        hz = self.Hz[:, mode].reshape(Ny, Nx)

        # Normalize fields
        e_abs = np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)
        h_abs = np.sqrt(np.abs(hx) ** 2 + np.abs(hy) ** 2 + np.abs(hz) ** 2)

        ex /= np.max(e_abs)
        ey /= np.max(e_abs)
        ez /= np.max(e_abs)
        hx /= np.max(h_abs)
        hy /= np.max(h_abs)
        hz /= np.max(h_abs)

        # Prepare field dictionary
        field_map = {
            'ex': (ex, 'Ex'),
            'ey': (ey, 'Ey'),
            'ez': (ez, 'Ez'),
            'hx': (hx, 'Hx'),
            'hy': (hy, 'Hy'),
            'hz': (hz, 'Hz'),
            'eabs': (e_abs / np.max(e_abs), '|E|'),
            'habs': (h_abs / np.max(h_abs), '|H|'),
        }

        # Determine which components to plot
        selected_fields = [key for key in field_map if kwargs.get(key)]
        if not selected_fields:
            selected_fields = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']

        n_fields = len(selected_fields)
        ncols = 3
        nrows = (n_fields + 2) // 3

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), layout='compressed')
        axes = np.array(axes).reshape(-1)  # Flatten in case of 1D array

        for i, field_name in enumerate(selected_fields):
            field_data, title = field_map[field_name]
            ax = axes[i]
            im = ax.imshow(np.abs(field_data), cmap='viridis', origin='lower',
                           extent=[0, self.x_range * 1e3, 0, self.y_range * 1e3])
            ax.imshow(np.abs(self.epsilon['xx']), cmap='inferno', origin='lower',
                      extent=[0, self.x_range * 1e3, 0, self.y_range * 1e3],
                      vmax=20, alpha=0.2)
            ax.set_title(title)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])  # Remove unused subplots

        fig.suptitle(
            rf'Mode {mode + 1}: $\hat{{\beta}}$ = {self.propagation_constant[mode]:.4f}, '
            rf'$\hat{{\alpha}}$ = {self.attenuation_constant[mode]:.4f}',
            fontsize=16
        )
        fig.colorbar(im, ax=axes[:i + 1], location='right', shrink=1, pad=0.02, label='Normalized Magnitude')
        plt.show()

    def visualize_with_gui(self):
        """Visualize field components with a dropdown menu for mode selection."""
        Nx, Ny = self.Nx, self.Ny
        colorbar = [None]  # To track and remove old colorbar
        import sys

        def plot_mode(selected_mode):
            mode = int(selected_mode) - 1

            # Reshape and normalize
            ex = self.Ex[:, mode].reshape(Ny, Nx)
            ey = self.Ey[:, mode].reshape(Ny, Nx)
            ez = self.Ez[:, mode].reshape(Ny, Nx)
            hx = self.Hx[:, mode].reshape(Ny, Nx)
            hy = self.Hy[:, mode].reshape(Ny, Nx)
            hz = self.Hz[:, mode].reshape(Ny, Nx)

            e_abs = np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)
            h_abs = np.sqrt(np.abs(hx) ** 2 + np.abs(hy) ** 2 + np.abs(hz) ** 2)

            ex /= np.max(e_abs)
            ey /= np.max(e_abs)
            ez /= np.max(e_abs)
            hx /= np.max(h_abs)
            hy /= np.max(h_abs)
            hz /= np.max(h_abs)

            fields = [ex, ey, ez, hx, hy, hz]
            titles = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']

            # Clear previous axes
            for ax in axes.flat:
                ax.clear()

            if colorbar[0] is not None:
                colorbar[0].remove()
                colorbar[0] = None

            # Plot and keep last image for colorbar
            for i, ax in enumerate(axes.flat):
                im = ax.imshow(np.abs(fields[i]), cmap='viridis', origin='lower',
                               extent=[0, self.x_range * 1e3, 0, self.y_range * 1e3],
                               vmin=0, vmax=1)
                ax.imshow(np.abs(self.epsilon['zz']), cmap='inferno', origin='lower',
                          extent=[0, self.x_range * 1e3, 0, self.y_range * 1e3],
                          vmax=20, alpha=0.2)
                ax.set_title(titles[i])
                ax.set_xlabel('x (mm)')
                ax.set_ylabel('y (mm)')

            # Adjust layout and add colorbar
            fig.subplots_adjust(right=0.86)  # Leave space for colorbar
            cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # position [left, bottom, width, height]
            norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            colorbar[0] = fig.colorbar(sm, cax=cbar_ax)
            colorbar[0].set_label("Normalized Magnitude")

            fig.suptitle(
                rf'Mode {mode + 1}: $\hat{{\beta}}$= {self.propagation_constant[mode]:.4f}, '
                rf'$\hat{{\alpha}}$ = {self.attenuation_constant[mode]:.4f}',
                fontsize=16
            )
            canvas.draw()

        # Tkinter GUI setup
        root = tk.Tk()
        root.title("Field Visualization")
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

        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky="nsew")

        controls_frame = tk.Frame(root)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=10)

        win_w, win_h = _configure_window()
        dpi = 100
        fig_w = max(8.0, (win_w / dpi) * 0.95)
        fig_h = max(5.0, (win_h / dpi) * 0.7)
        fig, axes = plt.subplots(2, 3, figsize=(fig_w, fig_h), dpi=dpi)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        mode_var = tk.StringVar(value="1")
        mode_menu = ttk.Combobox(controls_frame, textvariable=mode_var, values=list(range(1, self.num_modes + 1)))
        mode_menu.grid(row=0, column=0, padx=10, sticky="w")
        mode_menu.bind("<<ComboboxSelected>>", lambda event: plot_mode(mode_var.get()))

        quit_button = tk.Button(controls_frame, text="Quit", command=root.destroy)
        quit_button.grid(row=0, column=1, padx=10, sticky="e")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        controls_frame.columnconfigure(0, weight=0)
        controls_frame.columnconfigure(1, weight=1)

        plot_mode(mode_var.get())
        root.mainloop()

    def _yeeder2d(self):
        """Generate derivative matrices on a 2D Yee grid."""
        dx_normalised = self.k_0 * self.dx
        dy_normalised = self.k_0 * self.dy
        DEX, DEY, DHX, DHY = yeeder2d([self.Nx, self.Ny], [dx_normalised, dy_normalised], [0, 0])

        return DEX, DEY, DHX, DHY

    def _sqrt_positive_real(self, x):
        """Calculate the square root with positive real part."""
        sqrt = np.sqrt(x)
        return np.where(np.imag(sqrt) < 0, -sqrt, sqrt)

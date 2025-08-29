import tkinter as tk
from tkinter import ttk, filedialog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.sparse import diags, kron, eye, bmat
from scipy.sparse.linalg import eigs


class TM_Mode_Solver:
    def __init__(self, freq=24e9, x_range=20e-3, z_range=5e-3, Nx=200, Nz=50, num_modes=6, guess=0):
        self.freq = freq
        self.Nx, self.Nz = Nx, Nz
        self.dx, self.dz = x_range / Nx, z_range / Nz
        self.N = Nx * Nz
        self.num_modes = num_modes

        self.epsilon0 = 8.85e-12
        self.mu0 = 1.26e-6
        self.c = 1 / np.sqrt(self.epsilon0 * self.mu0)
        self.omega = 2 * np.pi * freq
        self.k0 = self.omega / self.c

        self.guess = guess
        self.eigenvectors = None
        self.gammas = None

        self._init_operators()
        self._init_materials()

    def _init_operators(self):
        def diff_operator(n, pbc=False):
            e = np.ones(n)
            data = np.array([-e, e])
            offsets = np.array([0, 1])
            D = diags(data, offsets, shape=(n, n)).tolil()
            if pbc:
                D[-1, 0] = 1
            return D.tocsr()

        def add_operator(n, pbc=False):
            e = np.ones(n)
            data = np.array([e, e])
            offsets = np.array([0, 1])
            D = 0.5 * diags(data, offsets, shape=(n, n)).tolil()
            if pbc:
                D[-1, 0] = 0.5
            return D.tocsr()

        Ix, Iz = eye(self.Nx), eye(self.Nz)

        self.DEX = kron(Iz, diff_operator(self.Nx)) / self.dx
        self.DEZ = kron(diff_operator(self.Nz, pbc=True), Ix) / self.dz
        self.DHX = -self.DEX.conj().T
        self.DHZ = -self.DEZ.conj().T
        self.D3 = kron(add_operator(self.Nz, pbc=True), Ix)
        self.D4 = self.D3.conj().T

    def _init_materials(self):
        shape = (self.Nz, self.Nx)
        self.Erxx = np.ones(shape, dtype=complex)
        self.Eryy = np.ones(shape, dtype=complex)
        self.Erzz = np.ones(shape, dtype=complex)
        self.Mrxx = np.ones(shape, dtype=complex)
        self.Mryy = np.ones(shape, dtype=complex)
        self.Mrzz = np.ones(shape, dtype=complex)

    def add_object(self, epsilon, mu, x_indices, z_indices):
        if np.isscalar(epsilon):
            self.Erxx[np.ix_(z_indices, x_indices)] = epsilon
            self.Eryy[np.ix_(z_indices, x_indices)] = epsilon
            self.Erzz[np.ix_(z_indices, x_indices)] = epsilon
        else:
            self.Erxx[np.ix_(z_indices, x_indices)] = epsilon[0]
            self.Eryy[np.ix_(z_indices, x_indices)] = epsilon[1]
            self.Erzz[np.ix_(z_indices, x_indices)] = epsilon[2]
        if np.isscalar(mu):
            self.Mrxx[np.ix_(z_indices, x_indices)] = mu
            self.Mryy[np.ix_(z_indices, x_indices)] = mu
            self.Mrzz[np.ix_(z_indices, x_indices)] = mu
        else:
            self.Mrxx[np.ix_(z_indices, x_indices)] = mu[0]
            self.Mryy[np.ix_(z_indices, x_indices)] = mu[1]
            self.Mrzz[np.ix_(z_indices, x_indices)] = mu[2]

    def add_UPML(self, pml_width=20, n=3, sigma_max=5.0, direction="top"):
        Nx, Nz = self.Nx, self.Nz
        sigma_x = np.zeros((Nz, Nx))

        def profile(i):  # polynomial grading
            return sigma_max * ((pml_width - i) / pml_width) ** n

        if direction in ("t", "top", "both"):
            for i in range(pml_width):
                s = profile(i)
                sigma_x[:, i] = s  # top
        if direction in ("b", "bottom", "both"):
            for i in range(pml_width):
                s = profile(i)
                sigma_x[:, -i - 1] = s  # bottom

        Sx = 1.0 + 1j * sigma_x / (self.epsilon0 * self.omega)

        # uniaxial scaling
        self.Erxx *= 1 / Sx
        self.Eryy *= Sx
        self.Erzz *= Sx
        self.Mrxx *= 1 / Sx
        self.Mryy *= Sx
        self.Mrzz *= Sx

    def solve(self):
        N = self.N
        Erxx_diag = diags(self.Erxx.ravel())
        Erzz_diag = diags(self.Erzz.ravel())
        Mryy_diag = diags(self.Mryy.ravel())

        D1 = 1j * self.omega * self.mu0 * Mryy_diag + 1j / self.omega * self.DEX @ (
                1 / self.epsilon0 * Erzz_diag.power(-1) @ self.DHX)
        D2 = 1j * self.omega * self.epsilon0 * Erxx_diag
        Zero = diags(np.zeros(N))

        A = bmat([[self.DEZ, D1], [D2, self.DHZ]]).tocsr()
        B = bmat([[self.D3, Zero], [Zero, self.D4]]).tocsr()

        eigenvalues, eigenvectors = eigs(A, M=B, k=self.num_modes, sigma=self.guess)
        self.eigenvectors = eigenvectors
        self.gammas = eigenvalues / self.k0

    def visualize_with_gui(self):
        if self.eigenvectors is None:
            raise RuntimeError("You need to run solve_modes() first.")

        def plot_mode(selected_mode):
            mode = int(selected_mode) - 1
            field_vector = self.eigenvectors[:, mode]
            Ex_field = np.abs(field_vector[:self.N].reshape((self.Nz, self.Nx)))
            Hy_field = np.abs(field_vector[self.N:].reshape((self.Nz, self.Nx)))

            # Normalize
            Ex_field /= np.max(Ex_field)
            Hy_field /= np.max(Hy_field)

            fields = [self.Eryy, Ex_field, Hy_field]
            titles = [r'Structure (Abs($\epsilon$))', 'ex (norm.)', 'hy (norm.)']

            # Clear previous images and colorbars
            for cbar in colorbars:
                cbar.remove()
            colorbars.clear()

            for ax in axes:
                ax.clear()

            for i, ax in enumerate(axes):
                if i == 0:
                    im = ax.imshow(np.abs(fields[i].T), cmap='viridis',
                                   extent=[0, self.Nz * self.dz * 1e3, 0, self.Nx * self.dx * 1e3],
                                   vmin=1, vmax=20)
                else:
                    im = ax.imshow(np.abs(fields[i].T), cmap='viridis',
                                   extent=[0, self.Nz * self.dz * 1e3, 0, self.Nx * self.dx * 1e3])
                ax.set_title(titles[i])
                ax.set_xlabel('z (mm)')
                ax.set_ylabel('x (mm)')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                colorbars.append(cbar)

            fig.suptitle(
                f'{self.freq / 1e9:.2f} GHz, Mode {mode + 1}: Beta = {self.gammas[mode].imag:.4f}, Alpha = {-self.gammas[mode].real:.4f}',
                fontsize=14)
            canvas.draw()

        root = tk.Tk()
        root.title("TM Mode Viewer")

        fig, axs = plt.subplots(1, 3, figsize=(12, 5))
        axes = axs
        colorbars = []

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        mode_var = tk.StringVar(value="1")
        mode_menu = ttk.Combobox(root, textvariable=mode_var, values=list(range(1, self.num_modes + 1)))
        mode_menu.pack(side=tk.LEFT, padx=10, pady=10)
        mode_menu.bind("<<ComboboxSelected>>", lambda event: plot_mode(mode_var.get()))

        quit_button = tk.Button(root, text="Quit", command=root.destroy)
        quit_button.pack(side=tk.RIGHT, padx=10, pady=10)

        plot_mode(mode_var.get())

        root.mainloop()


class TE_Mode_Solver:
    def __init__(self, freq=30e9, x_range=20e-3, z_range=5e-3, Nx=200, Nz=50, num_modes=4, guess=0):
        self.freq = freq
        self.Nx, self.Nz = Nx, Nz
        self.dx, self.dz = x_range / Nx, z_range / Nz
        self.N = Nx * Nz
        self.num_modes = num_modes

        self.epsilon0 = 8.85e-12
        self.mu0 = 1.26e-6
        self.c = 1 / np.sqrt(self.epsilon0 * self.mu0)
        self.omega = 2 * np.pi * freq
        self.k0 = self.omega / self.c

        self.guess = guess
        self.eigenvectors = None
        self.gammas = None

        self._init_operators()
        self._init_materials()

    def _init_operators(self):
        def diff_operator(n, pbc=False):
            e = np.ones(n)
            data = np.array([-e, e])
            offsets = np.array([0, 1])
            D = diags(data, offsets, shape=(n, n)).tolil()
            if pbc:
                D[-1, 0] = 1
            return D.tocsr()

        def add_operator(n, pbc=False):
            e = np.ones(n)
            data = np.array([e, e])
            offsets = np.array([0, 1])
            D = 0.5 * diags(data, offsets, shape=(n, n)).tolil()
            if pbc:
                D[-1, 0] = 0.5
            return D.tocsr()

        Ix, Iz = eye(self.Nx), eye(self.Nz)

        self.DEX = kron(Iz, diff_operator(self.Nx)) / self.dx
        self.DEZ = kron(diff_operator(self.Nz, pbc=True), Ix) / self.dz
        self.DHX = -self.DEX.conj().T
        self.DHZ = -self.DEZ.conj().T
        self.D3 = kron(add_operator(self.Nz, pbc=True), Ix).conj().T
        self.D4 = self.D3.conj().T

    def _init_materials(self):
        shape = (self.Nz, self.Nx)
        self.Erxx = np.ones(shape, dtype=complex)
        self.Eryy = np.ones(shape, dtype=complex)
        self.Erzz = np.ones(shape, dtype=complex)
        self.Mrxx = np.ones(shape, dtype=complex)
        self.Mryy = np.ones(shape, dtype=complex)
        self.Mrzz = np.ones(shape, dtype=complex)

    def add_object(self, epsilon, mu, x_indices, z_indices):
        if np.isscalar(epsilon):
            self.Erxx[np.ix_(z_indices, x_indices)] = epsilon
            self.Eryy[np.ix_(z_indices, x_indices)] = epsilon
            self.Erzz[np.ix_(z_indices, x_indices)] = epsilon
        else:
            self.Erxx[np.ix_(z_indices, x_indices)] = epsilon[0]
            self.Eryy[np.ix_(z_indices, x_indices)] = epsilon[1]
            self.Erzz[np.ix_(z_indices, x_indices)] = epsilon[2]
        if np.isscalar(mu):
            self.Mrxx[np.ix_(z_indices, x_indices)] = mu
            self.Mryy[np.ix_(z_indices, x_indices)] = mu
            self.Mrzz[np.ix_(z_indices, x_indices)] = mu
        else:
            self.Mrxx[np.ix_(z_indices, x_indices)] = mu[0]
            self.Mryy[np.ix_(z_indices, x_indices)] = mu[1]
            self.Mrzz[np.ix_(z_indices, x_indices)] = mu[2]

    def add_UPML(self, pml_width=20, n=3, sigma_max=5.0, direction="top"):
        Nx, Nz = self.Nx, self.Nz
        sigma_x = np.zeros((Nz, Nx))

        def profile(i):  # polynomial grading
            return sigma_max * ((pml_width - i) / pml_width) ** n

        if direction in ("t", "top", "both"):
            for i in range(pml_width):
                s = profile(i)
                sigma_x[:, i] = s  # top
        if direction in ("b", "bottom", "both"):
            for i in range(pml_width):
                s = profile(i)
                sigma_x[:, -i - 1] = s  # bottom

        Sx = 1.0 + 1j * sigma_x / (self.epsilon0 * self.omega)

        # uniaxial scaling
        self.Erxx *= 1 / Sx
        self.Eryy *= Sx
        self.Erzz *= Sx
        self.Mrxx *= 1 / Sx
        self.Mryy *= Sx
        self.Mrzz *= Sx

    def solve(self):
        N = self.N

        Eryy_diag = diags(self.Eryy.ravel())
        Mrxx_diag = diags(self.Mrxx.ravel())
        Mrzz_diag = diags(self.Mrzz.ravel())

        D1 = -1j * self.omega * self.epsilon0 * Eryy_diag - 1j / self.omega * self.DHX @ (
                1 / self.mu0 * Mrzz_diag.power(-1) @ self.DEX)
        D2 = -1j * self.omega * self.mu0 * Mrxx_diag
        Zero = diags(np.zeros(N))

        A = bmat([[self.DHZ, D1], [D2, self.DEZ]]).tocsr()
        B = bmat([[self.D3, Zero], [Zero, self.D4]]).tocsr()

        eigenvalues, eigenvectors = eigs(A, M=B, k=self.num_modes, sigma=self.guess)
        self.eigenvectors = eigenvectors
        self.gammas = eigenvalues / self.k0

    def visualize_with_gui(self):
        if self.eigenvectors is None:
            raise RuntimeError("You need to run solve_modes() first.")

        def plot_mode(selected_mode):
            mode = int(selected_mode) - 1
            field_vector = self.eigenvectors[:, mode]
            Hx_field = np.abs(field_vector[:self.N].reshape((self.Nz, self.Nx)))
            Ey_field = np.abs(field_vector[self.N:].reshape((self.Nz, self.Nx)))

            Hx_field /= np.max(Hx_field)
            Ey_field /= np.max(Ey_field)

            fields = [self.Eryy, Hx_field, Ey_field]
            titles = ['Structure (Abs($\epsilon$))', 'hx (norm.)', 'ey (norm.)']

            for cbar in colorbars:
                cbar.remove()
            colorbars.clear()

            for ax in axes:
                ax.clear()

            for i, ax in enumerate(axes):
                if i == 0:
                    im = ax.imshow(np.abs(fields[i].T), cmap='viridis',
                                   extent=[0, self.Nz * self.dz * 1e3, 0, self.Nx * self.dx * 1e3],
                                   vmin=1, vmax=20)
                else:
                    im = ax.imshow(np.abs(fields[i].T), cmap='viridis',
                                   extent=[0, self.Nz * self.dz * 1e3, 0, self.Nx * self.dx * 1e3])
                ax.set_title(titles[i])
                ax.set_xlabel('z (mm)')
                ax.set_ylabel('x (mm)')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                colorbars.append(cbar)

            fig.suptitle(
                f'{self.freq / 1e9:.2f} GHz, Mode {mode + 1}: Beta = {self.gammas[mode].imag:.4f}, Alpha = {-self.gammas[mode].real:.4f}',
                fontsize=14)
            canvas.draw()

        def save_figure():
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                fig.savefig(file_path, dpi=300)
                print(f"Figure saved to {file_path}")

        root = tk.Tk()
        root.title("TE Mode Viewer")

        fig, axs = plt.subplots(1, 3, figsize=(12, 5))
        axes = axs
        colorbars = []

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        controls_frame = tk.Frame(root)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        mode_var = tk.StringVar(value="1")
        mode_menu = ttk.Combobox(controls_frame, textvariable=mode_var, values=list(range(1, self.num_modes + 1)))
        mode_menu.pack(side=tk.LEFT, padx=10)
        mode_menu.bind("<<ComboboxSelected>>", lambda event: plot_mode(mode_var.get()))

        save_button = tk.Button(controls_frame, text="Save Figure", command=save_figure)
        save_button.pack(side=tk.LEFT, padx=10)

        quit_button = tk.Button(controls_frame, text="Quit", command=root.destroy)
        quit_button.pack(side=tk.RIGHT, padx=10)

        plot_mode(mode_var.get())

        root.mainloop()

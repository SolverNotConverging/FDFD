import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.sparse import diags, kron, eye, bmat
from scipy.sparse.linalg import eigs


class Periodic_3D_Mode_Solver:
    def __init__(self, Nx, Ny, Nz, x_range, y_range, z_range, freq, num_modes, sigma_guess_func=None, tol=0):
        # Store parameters
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dx = x_range / Nx
        self.dy = y_range / Ny
        self.dz = z_range / Nz
        self.N = Nx * Ny * Nz

        self.freq = freq
        self.omega = 2 * np.pi * freq
        self.k0 = self.omega / 3e8
        self.epsilon0 = 8.85e-12
        self.mu0 = 1.26e-6
        self.num_modes = num_modes

        # Frequency-dependent guess
        if sigma_guess_func:
            self.sigma_guess = sigma_guess_func(freq)
        else:
            self.sigma_guess = 0

        self.tol = tol

        # Grids
        self.Ix, self.Iy, self.Iz = eye(Nx), eye(Ny), eye(Nz)
        self.DEX = kron(kron(self.Iz, self.Iy), self.diff_operator(Nx)) / self.dx
        self.DEY = kron(kron(self.Iz, self.diff_operator(Ny)), self.Ix) / self.dy
        self.DEZ = kron(kron(self.diff_operator_pbc(Nz), self.Iy), self.Ix) / self.dz
        self.DHX = -self.DEX.conj().T
        self.DHY = -self.DEY.conj().T
        self.DHZ = -self.DEZ.conj().T

        # Material arrays
        self.Erxx_3D = np.ones((Nz, Ny, Nx), dtype=complex)
        self.Eryy_3D = np.ones((Nz, Ny, Nx), dtype=complex)
        self.Erzz_3D = np.ones((Nz, Ny, Nx), dtype=complex)
        self.Mrxx_3D = np.ones((Nz, Ny, Nx), dtype=complex)
        self.Mryy_3D = np.ones((Nz, Ny, Nx), dtype=complex)
        self.Mrzz_3D = np.ones((Nz, Ny, Nx), dtype=complex)

        # Storage
        self.fields = {}
        self.eigenvalues = None
        self.eigenvectors = None

    # --- Differentiation operators
    def diff_operator(self, n):
        e = np.ones(n)
        return diags([-e, e], [0, 1], shape=(n, n)).tolil().tocsr()

    def diff_operator_pbc(self, n):
        e = np.ones(n)
        D = diags([-e, e], [0, 1], shape=(n, n)).tolil()
        D[-1, 0] = 1
        return D.tocsr()

    def add_operator_pbc(self, n):
        e = np.ones(n)
        D = 0.5 * diags([e, e], [0, 1], shape=(n, n)).tolil()
        D[-1, 0] = 0.5
        return D.tocsr()

    # --- Helper functions for modeling
    def add_object(self, er, mr, x_slice, y_slice, z_slice):
        if np.isscalar(er):
            self.Erxx_3D[z_slice, y_slice, x_slice] = er
            self.Eryy_3D[z_slice, y_slice, x_slice] = er
            self.Erzz_3D[z_slice, y_slice, x_slice] = er
        else:
            self.Erxx_3D[z_slice, y_slice, x_slice] = er[0]
            self.Eryy_3D[z_slice, y_slice, x_slice] = er[1]
            self.Erzz_3D[z_slice, y_slice, x_slice] = er[2]

        if np.isscalar(mr):
            self.Mrxx_3D[z_slice, y_slice, x_slice] = mr
            self.Mryy_3D[z_slice, y_slice, x_slice] = mr
            self.Mrzz_3D[z_slice, y_slice, x_slice] = mr
        else:
            self.Mrxx_3D[z_slice, y_slice, x_slice] = mr[0]
            self.Mryy_3D[z_slice, y_slice, x_slice] = mr[1]
            self.Mrzz_3D[z_slice, y_slice, x_slice] = mr[2]

    def add_absorbing_boundary(self, sides=('-x', '+x', '-y', '+y'), width=10, max_loss=5, n=3):
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
                sl = np.s_[:, :, i]
                self.Erxx_3D[sl] /= S
                self.Eryy_3D[sl] *= S
                self.Erzz_3D[sl] *= S
                self.Mrxx_3D[sl] /= S
                self.Mryy_3D[sl] *= S
                self.Mrzz_3D[sl] *= S

            if '+x' in sides:
                sl = np.s_[:, :, -1 - i]
                self.Erxx_3D[sl] /= S
                self.Eryy_3D[sl] *= S
                self.Erzz_3D[sl] *= S
                self.Mrxx_3D[sl] /= S
                self.Mryy_3D[sl] *= S
                self.Mrzz_3D[sl] *= S

            if '-y' in sides:
                sl = np.s_[:, i, :]
                self.Erxx_3D[sl] *= S
                self.Eryy_3D[sl] /= S
                self.Erzz_3D[sl] *= S
                self.Mrxx_3D[sl] *= S
                self.Mryy_3D[sl] /= S
                self.Mrzz_3D[sl] *= S

            if '+y' in sides:
                sl = np.s_[:, -1 - i, :]
                self.Erxx_3D[sl] *= S
                self.Eryy_3D[sl] /= S
                self.Erzz_3D[sl] *= S
                self.Mrxx_3D[sl] *= S
                self.Mryy_3D[sl] /= S
                self.Mrzz_3D[sl] *= S

    # --- Solver
    def solve(self):
        N = self.N
        omega, epsilon0, mu0 = self.omega, self.epsilon0, self.mu0

        # Build diagonal sparse matrices
        Erxx = diags(self.Erxx_3D.ravel())
        Eryy = diags(self.Eryy_3D.ravel())
        Erzz = diags(self.Erzz_3D.ravel())
        Mrxx = diags(self.Mrxx_3D.ravel())
        Mryy = diags(self.Mryy_3D.ravel())
        Mrzz = diags(self.Mrzz_3D.ravel())
        Zero = diags(np.zeros(N))

        # Build system matrices
        A = bmat([
            [self.DEZ, Zero, self.DEX @ (-1j / (omega * epsilon0) * Erzz.power(-1) @ self.DHY),
             self.DEX @ (1j / (omega * epsilon0) * Erzz.power(-1) @ self.DHX) + 1j * omega * mu0 * Mryy],
            [Zero, self.DEZ,
             self.DEY @ (-1j / (omega * epsilon0) * Erzz.power(-1) @ self.DHY) - 1j * omega * mu0 * Mrxx,
             self.DEY @ (1j / (omega * epsilon0) * Erzz.power(-1) @ self.DHX)],
            [self.DHX @ (1j / (omega * mu0) * Mrzz.power(-1) @ self.DEY),
             self.DHX @ (-1j / (omega * mu0) * Mrzz.power(-1) @ self.DEX) - 1j * omega * epsilon0 * Eryy,
             self.DHZ, Zero],
            [self.DHY @ (1j / (omega * mu0) * Mrzz.power(-1) @ self.DEY) + 1j * omega * epsilon0 * Erxx,
             self.DHY @ (-1j / (omega * mu0) * Mrzz.power(-1) @ self.DEX),
             Zero, self.DHZ]
        ]).tocsr()

        B_diag = kron(kron(self.add_operator_pbc(self.Nz), self.Iy), self.Ix)
        B = bmat([[B_diag, Zero, Zero, Zero], [Zero, B_diag, Zero, Zero],
                  [Zero, Zero, B_diag.conj().T, Zero], [Zero, Zero, Zero, B_diag.conj().T]]).tocsr()

        # Solve
        self.eigenvalues, self.eigenvectors = eigs(A, M=B, k=self.num_modes, sigma=self.sigma_guess, tol=self.tol)
        self.gammas = self.eigenvalues / self.k0
        self.store_fields()

    def store_fields(self):
        N = self.N
        self.fields['Ex'] = np.array(
            [self.eigenvectors[:, i][0 * N:1 * N].reshape((self.Nz, self.Ny, self.Nx)) for i in range(self.num_modes)])
        self.fields['Ey'] = np.array(
            [self.eigenvectors[:, i][1 * N:2 * N].reshape((self.Nz, self.Ny, self.Nx)) for i in range(self.num_modes)])
        self.fields['Hx'] = np.array(
            [self.eigenvectors[:, i][2 * N:3 * N].reshape((self.Nz, self.Ny, self.Nx)) for i in range(self.num_modes)])
        self.fields['Hy'] = np.array(
            [self.eigenvectors[:, i][3 * N:4 * N].reshape((self.Nz, self.Ny, self.Nx)) for i in range(self.num_modes)])

    # --- Plotting
    def plot_field_plane(self, axis, index, mode_index=0, field='Ex'):
        field_data = np.abs(self.fields[field][mode_index])
        if axis == 'x':
            plt.imshow(field_data[:, :, index], cmap='hot')
        elif axis == 'y':
            plt.imshow(field_data[:, index, :], cmap='hot')
        elif axis == 'z':
            plt.imshow(field_data[index, :, :], cmap='hot')
        plt.colorbar()
        plt.title(f'{field} at {axis}={index} | Mode {mode_index}')
        plt.show()

    def visualize_with_gui(self):
        root = tk.Tk()
        root.title("Mode Viewer")

        frame = ttk.Frame(root)
        frame.pack(padx=10, pady=10)

        ttk.Label(frame, text="Select Mode:").grid(row=0, column=0, padx=5)
        mode_var = tk.IntVar(value=0)
        mode_dropdown = ttk.Combobox(frame, textvariable=mode_var, values=list(range(self.num_modes)), width=5)
        mode_dropdown.grid(row=0, column=1, padx=5)

        plot_frame = ttk.Frame(root)
        plot_frame.pack()

        canvas = None

        def update_plot(*args):
            nonlocal canvas
            if canvas:
                canvas.get_tk_widget().destroy()

            fig = self._create_mode_figure(mode_var.get())
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

        mode_dropdown.bind("<<ComboboxSelected>>", update_plot)

        # Initial plot
        update_plot()

        root.mainloop()

    def _create_mode_figure(self, mode_index):
        # This replicates your original subplot figure, adapted to self.fields
        Ex_field = np.abs(self.fields['Ex'][mode_index])
        Ey_field = np.abs(self.fields['Ey'][mode_index])
        Hx_field = np.abs(self.fields['Hx'][mode_index])
        Hy_field = np.abs(self.fields['Hy'][mode_index])

        Ex_field /= np.max(Ex_field)
        Ey_field /= np.max(Ey_field)
        Hx_field /= np.max(Hx_field)
        Hy_field /= np.max(Hy_field)

        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz

        fig, axs = plt.subplots(2, 5, figsize=(18, 10))
        x_slice = Nx // 2
        z_slice = Nz // 2

        im0 = axs[0, 0].imshow(np.abs(self.Erzz_3D[:, :, x_slice]).T, cmap='viridis',
                               extent=[0, Nz * dz * 1e3, 0, Ny * dy * 1e3], vmax=20)
        axs[0, 0].set_title('Unit Cell')
        plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

        for ax, data, title in zip(
                axs[0, 1:],
                [Ex_field[:, :, x_slice], Ey_field[:, :, x_slice], Hx_field[:, :, x_slice], Hy_field[:, :, x_slice]],
                ['Ex', 'Ey', 'Hx', 'Hy']):
            im = ax.imshow(data.T, cmap='viridis', extent=[0, Nz * dz * 1e3, 0, Ny * dy * 1e3])
            ax.set_xlabel('z (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        im0 = axs[1, 0].imshow(np.abs(self.Erzz_3D[z_slice, :, :]), cmap='viridis',
                               extent=[0, Nx * dx * 1e3, 0, Ny * dy * 1e3], vmax=20)
        axs[1, 0].set_title('Unit Cell')
        plt.colorbar(im0, ax=axs[1, 0], fraction=0.046, pad=0.04)

        for ax, data, title in zip(
                axs[1, 1:],
                [Ex_field[z_slice, :, :], Ey_field[z_slice, :, :], Hx_field[z_slice, :, :], Hy_field[z_slice, :, :]],
                ['Ex', 'Ey', 'Hx', 'Hy']):
            im = ax.imshow(data, cmap='viridis', extent=[0, Nx * dx * 1e3, 0, Ny * dy * 1e3])
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(
            f'Mode {mode_index} | Beta = {self.gammas[mode_index].imag:.4f}, Alpha = {self.gammas[mode_index].real:.4f}',
            fontsize=16)
        plt.tight_layout()
        return fig

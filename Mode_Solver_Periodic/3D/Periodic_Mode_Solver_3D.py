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

            if '+y' in sides:
                sl = np.s_[:, i, :]
                self.Erxx_3D[sl] *= S
                self.Eryy_3D[sl] /= S
                self.Erzz_3D[sl] *= S
                self.Mrxx_3D[sl] *= S
                self.Mryy_3D[sl] /= S
                self.Mrzz_3D[sl] *= S

            if '-y' in sides:
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

        # ==== Controls ====
        ctrl = ttk.Frame(root)
        ctrl.pack(padx=10, pady=10, fill='x')

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
        plot_frame.pack(fill='both', expand=True)
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

        # Clamp index to valid range
        if axis == 'x':
            index = int(np.clip(index, 0, self.Nx - 1))
            slicer = (slice(None), slice(None), index)  # (z, y, x=index) -> 2D: (z, y)
            orient_T = True  # transpose to show y vertical, z horizontal
        elif axis == 'y':
            index = int(np.clip(index, 0, self.Ny - 1))
            slicer = (slice(None), index, slice(None))  # (z, y=index, x) -> 2D: (z, x)
            orient_T = True  # transpose to show x vertical, z horizontal
        else:  # 'z'
            index = int(np.clip(index, 0, self.Nz - 1))
            slicer = (index, slice(None), slice(None))  # (z=index, y, x) -> 2D: (y, x)
            orient_T = False  # already (y, x)

        # Extract slices
        Ex2 = Ex[slicer]
        Ey2 = Ey[slicer]
        Hx2 = Hx[slicer]
        Hy2 = Hy[slicer]

        # Decide plotting extents and labels
        extent, xlabel, ylabel = self._slice_extent_labels(axis)

        # Figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.25, hspace=0.25)

        panels = [
            ('Ex', Ex2, axes[0, 0]),
            ('Ey', Ey2, axes[0, 1]),
            ('Hx', Hx2, axes[1, 0]),
            ('Hy', Hy2, axes[1, 1]),
        ]

        for title, data2d, ax in panels:
            imdata = data2d.T if orient_T else data2d
            im = ax.imshow(imdata, cmap='viridis', extent=extent, aspect='auto')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Mode info
        beta = float(np.imag(self.gammas[mode_index]))
        alpha = float(np.real(self.gammas[mode_index]))
        fig.suptitle(f"Mode {mode_index} | Slice {axis}={index} | Beta={beta:.4f}, Alpha={alpha:.4f}", fontsize=12)

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
            Erxx_3D=self.Erxx_3D,
            Eryy_3D=self.Eryy_3D,
            Erzz_3D=self.Erzz_3D,
            Mrxx_3D=self.Mrxx_3D,
            Mryy_3D=self.Mryy_3D,
            Mrzz_3D=self.Mrzz_3D,

            # modal results
            eigenvalues=self.eigenvalues,
            gammas=self.gammas,
            Ex=self.fields['Ex'],  # shape: (num_modes, Nz, Ny, Nx)
            Ey=self.fields['Ey'],
            Hx=self.fields['Hx'],
            Hy=self.fields['Hy'],
        )

        if include_eigenvectors and self.eigenvectors is not None:
            out['eigenvectors'] = self.eigenvectors  # shape: (4*N, num_modes)

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
            Nx = int(d['Nx']);
            Ny = int(d['Ny']);
            Nz = int(d['Nz'])
            x_range = float(d['x_range']);
            y_range = float(d['y_range']);
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
            inst.Erxx_3D = d['Erxx_3D']
            inst.Eryy_3D = d['Eryy_3D']
            inst.Erzz_3D = d['Erzz_3D']
            inst.Mrxx_3D = d['Mrxx_3D']
            inst.Mryy_3D = d['Mryy_3D']
            inst.Mrzz_3D = d['Mrzz_3D']

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

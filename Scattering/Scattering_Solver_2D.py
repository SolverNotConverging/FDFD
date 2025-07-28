import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.special  # for Hankel-functions (cylindrical waves)
from scipy.sparse import diags, linalg as spla

from yee_derivative import yeeder2d  # ← your own helper remains untouched


class FDFD2DScatteringSolver:
    """
    2-D frequency–domain FDFD solver (TEz / TMz) on a Yee grid.
    Geometry is defined on the cell centres,   E/H - derivatives on staggered edges
    via the helper `yeeder2d`.

    Parameters
    ----------
    frequency : float      – source frequency [Hz]
    x_range   : float      – physical size of the domain in x  [m]
    y_range   : float      – physical size of the domain in y  [m]
    Nx, Ny    : int        – number of Yee cells in x and y
    """

    c0 = 299_792_458.0  # exact value
    eps0 = 8.854187817e-12

    # ------------- initialisation & helper data ---------------------------------
    def __init__(self, frequency, x_range, y_range, Nx, Ny):
        self.frequency = float(frequency)
        self.omega = 2 * np.pi * self.frequency
        self.k0 = self.omega / self.c0

        self.Nx, self.Ny = int(Nx), int(Ny)
        self.x_range, self.y_range = float(x_range), float(y_range)
        self.dx, self.dy = self.x_range / self.Nx, self.y_range / self.Ny

        self.N = self.Nx * self.Ny
        self._prepare_coordinate_grids()

        # material tensors (relative)
        shape = (self.Ny, self.Nx)
        self.ERxx = np.ones(shape, dtype=np.complex128)
        self.ERyy = np.ones(shape, dtype=np.complex128)
        self.ERzz = np.ones(shape, dtype=np.complex128)
        self.MRxx = np.ones(shape, dtype=np.complex128)
        self.MRyy = np.ones(shape, dtype=np.complex128)
        self.MRzz = np.ones(shape, dtype=np.complex128)

        # placeholders
        self.source = np.zeros(self.N, complex)
        self.Q = sp.identity(self.N, format='csr')

        # cache derivative operators so they are built **once**
        self._DEX = self._DEY = self._DHX = self._DHY = None  # filled lazily

    # --- utility ----------------------------------------------------------------
    def _prepare_coordinate_grids(self):
        """ Create 2-D arrays of the cell-centre coordinates, reused for sources. """
        x = (np.arange(self.Nx) + 0.5) * self.dx - self.x_range / 2  # centre at 0
        y = (np.arange(self.Ny) + 0.5) * self.dy - self.y_range / 2
        self.X, self.Y = np.meshgrid(x, y, indexing='xy')  # Ny×Nx each

    def _yeeder2d(self):
        if self._DEX is None:  # build once and cache
            dx_norm = self.k0 * self.dx
            dy_norm = self.k0 * self.dy
            self._DEX, self._DEY, self._DHX, self._DHY = yeeder2d(
                [self.Nx, self.Ny], [dx_norm, dy_norm], [0, 0])
        return self._DEX, self._DEY, self._DHX, self._DHY

    # ============ 1.  Geometry / material definition ============================
    def add_object(self, er_tensor, mr_tensor, region_mask):
        """
        region_mask : boolean Ny×Nx array – cells where the object lives
        er_tensor   : scalar or len-3 list/array  (ε_xx, ε_yy, ε_zz)
        mr_tensor   : same for μ
        """
        if np.isscalar(er_tensor): er_tensor = (er_tensor,) * 3
        if np.isscalar(mr_tensor): mr_tensor = (mr_tensor,) * 3

        self.ERxx[region_mask] = er_tensor[0]
        self.ERyy[region_mask] = er_tensor[1]
        self.ERzz[region_mask] = er_tensor[2]
        self.MRxx[region_mask] = mr_tensor[0]
        self.MRyy[region_mask] = mr_tensor[1]
        self.MRzz[region_mask] = mr_tensor[2]

    # ============ 2.  Source definition =========================================
    def add_source(self,
                   src_type: str = "plane_wave",
                   angle_deg: float = 0.0,
                   polarization: str = "TE",  # TE means Ez, TM means Hz solved
                   location: tuple | None = None,
                   amplitude: float | complex = 1.0):
        """
        Populate self.source (flattened Ny×Nx) with the incident field.

        plane_wave : exp(-i k · r),  k=(k0 cosθ, k0 sinθ) with θ measured *from +x*
        point      : 2-D scalar Green’s function H₀⁽¹⁾(k r)
        """
        self.source[:] = 0.0
        kx = self.k0 * np.cos(np.deg2rad(angle_deg))
        ky = self.k0 * np.sin(np.deg2rad(angle_deg))

        if src_type == "plane_wave":
            ph = np.exp(-1j * (kx * self.X + ky * self.Y))
            self.source = (amplitude * ph).ravel()

        elif src_type == "point":
            if location is None:
                raise ValueError("For a point source you must supply location=(x0,y0)")
            x0, y0 = location
            r = np.hypot(self.X - x0, self.Y - y0)
            r[r == 0] = self.dx / 50  # avoid singularity at the source cell
            if polarization.upper() == "TE":
                field = scipy.special.hankel1(0, self.k0 * r)  # Ez for 2-D TE
            else:  # TM (magnetic out of plane)
                field = 1j / 4 * scipy.special.hankel1(0, self.k0 * r)  # convention
            self.source = (amplitude * field).ravel()
        else:
            raise ValueError(f"Unknown src_type {src_type}")

    # ============ 3.  Uniaxial PML  (UPML) ======================================
    def add_UPML(self, pml_width=20, n=3, sigma_max=5.0, direction="both"):
        Nx, Ny = self.Nx, self.Ny
        sigma_x = np.zeros((Ny, Nx));
        sigma_y = np.zeros_like(sigma_x)

        # --- build σ(x), σ(y) ---------------------------------------------------
        def profile(i):  # polynomial grading
            return sigma_max * ((pml_width - i) / pml_width) ** n

        if direction in ("x", "both"):
            for i in range(pml_width):
                s = profile(i)
                sigma_x[:, i] = s  # left
                sigma_x[:, -i - 1] = s  # right
        if direction in ("y", "both"):
            for i in range(pml_width):
                s = profile(i)
                sigma_y[i, :] = s  # bottom
                sigma_y[-i - 1, :] = s  # top

        Sx = 1.0 + 1j * sigma_x / (self.eps0 * self.omega)
        Sy = 1.0 + 1j * sigma_y / (self.eps0 * self.omega)

        # uniaxial scaling
        self.ERxx *= Sy / Sx
        self.ERyy *= Sx / Sy
        self.ERzz *= Sx * Sy
        self.MRxx *= Sy / Sx
        self.MRyy *= Sx / Sy
        self.MRzz *= Sx * Sy

    # ============ 4.  Mask operator (total-field / scattered-field) =============
    def add_mask(self, value: int | float | np.ndarray | sp.spmatrix = 30):
        Ny, Nx = self.Ny, self.Nx
        if np.isscalar(value):
            v = int(value)
            q = np.ones((Ny, Nx))
            if 2 * v < min(Nx, Ny): q[v:-v, v:-v] = 0
            diag = q.ravel()
            self.Q = diags(diag, format='csr')
        elif isinstance(value, (np.ndarray, list)):
            arr = np.asarray(value)
            if arr.shape != (Ny, Nx):
                raise ValueError(f"mask must be shape {(Ny, Nx)}")
            self.Q = diags(arr.ravel(), format='csr')
        elif sp.issparse(value):
            self.Q = value.tocsr()
        else:
            raise TypeError("Unsupported mask type")
        return self.Q

    # ============ 5.  Build system matrices ====================================
    def _build_A_TE(self):
        MRXX = diags(self.MRxx.ravel())
        MRYY = diags(self.MRyy.ravel())
        ERZZ = diags(self.ERzz.ravel())
        DEX, DEY, DHX, DHY = self._yeeder2d()
        return DHX @ MRYY.power(-1) @ DEX + DHY @ MRXX.power(-1) @ DEY + ERZZ

    def _build_A_TM(self):
        ERXX = diags(self.ERxx.ravel())
        ERYY = diags(self.ERyy.ravel())
        MRZZ = diags(self.MRzz.ravel())
        DEX, DEY, DHX, DHY = self._yeeder2d()
        return DEX @ ERYY.power(-1) @ DHX + DEY @ ERXX.power(-1) @ DHY + MRZZ

    # ============ 6.  Solvers ====================================================
    def solve_total_field_TE(self, reuse_factorisation: bool = True):
        if not hasattr(self, "_A_TE"):
            self._A_TE = self._build_A_TE()
            self._LU_TE = spla.factorized(self._A_TE) if reuse_factorisation else None

        A = self._A_TE
        Q = self.Q
        b = (Q @ A - A @ Q) @ self.source[:, None]
        if self._LU_TE is not None:
            ez = self._LU_TE(b)
        else:
            ez = spla.spsolve(A, b)
        self.Ez = ez.reshape(self.Ny, self.Nx)
        return self.Ez

    def solve_total_field_TM(self, reuse_factorisation: bool = True):
        if not hasattr(self, "_A_TM"):
            self._A_TM = self._build_A_TM()
            self._LU_TM = spla.factorized(self._A_TM) if reuse_factorisation else None

        A = self._A_TM
        Q = self.Q
        b = (Q @ A - A @ Q) @ self.source[:, None]
        if self._LU_TM is not None:
            hz = self._LU_TM(b)
        else:
            hz = spla.spsolve(A, b)
        self.Hz = hz.reshape(self.Ny, self.Nx)
        return self.Hz

    # ============ 7.  Quick-look visualisations =================================
    def _quick_imshow(self, data, ax, title, cmap="viridis"):
        im = ax.imshow(np.real_if_close(data), origin='lower', cmap=cmap,
                       extent=[-self.x_range / 2, self.x_range / 2,
                               -self.y_range / 2, self.y_range / 2])
        ax.set_title(title);
        ax.set_xlabel('x [m]');
        ax.set_ylabel('y [m]')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def TE_Visualization(self):
        """
        4-panel figure:
        1 – |ε_r|   (structure)
        2 – Re(incident Ez)
        3 – Q mask
        4 – Re(total Ez)
        """
        if not hasattr(self, "Ez"):
            raise RuntimeError("Run solve_total_field_TE() first")
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        self._quick_imshow(np.abs(self.ERzz), axs[0, 0], r"|ε_r|")
        self._quick_imshow(self.source.reshape(self.Ny, self.Nx).real,
                           axs[0, 1], "Incident Ez (real)")
        self._quick_imshow(self.Q.diagonal().reshape(self.Ny, self.Nx),
                           axs[1, 0], "Mask Q")
        self._quick_imshow(abs(self.Ez.real), axs[1, 1], "Total Ez (abs)")
        plt.show()

    def TM_Visualization(self):
        """
        4-panel figure analogous to TE_Visualization, but for TM (Hz).
        """
        if not hasattr(self, "Hz"):
            raise RuntimeError("Run solve_total_field_TM() first")
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        self._quick_imshow(np.abs(self.ERzz), axs[0, 0], r"|ε_r|")
        self._quick_imshow(self.source.reshape(self.Ny, self.Nx).real,
                           axs[0, 1], "Incident Hz (real)")
        self._quick_imshow(self.Q.diagonal().reshape(self.Ny, self.Nx),
                           axs[1, 0], "Mask Q")
        self._quick_imshow(abs(self.Hz.real), axs[1, 1], "Total Hz (abs)")
        plt.show()

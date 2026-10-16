"""Photonic band structure of a 2D periodic cell."""

import numpy as np

from ..core import eigs, materials, shapes
from ..core.constants import C0
from ..core.errors import ConfigurationError
from ..core.grid import GridSolver, check_guess, check_num_modes, check_polarization, check_tol
from ..core.results import BandResult, SolveInfo


def bloch_path(points, *, num_points=40):
    """Sample a polyline of Bloch vectors.

    Parameters
    ----------
    points : sequence of (kx, ky)
        Vertices in rad/m, e.g. Gamma, X, M, Gamma.
    num_points : int
        Total samples, distributed in proportion to segment length. Every
        vertex is included.

    Returns
    -------
    ndarray
        Array of shape ``(2, num_points)``.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2 or not np.isfinite(pts).all():
        raise ConfigurationError("points must contain at least two finite (kx, ky) vectors.")
    if isinstance(num_points, bool) or int(num_points) != num_points or num_points < len(pts):
        raise ConfigurationError("num_points must be an integer at least the number of vertices.")
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if np.any(lengths == 0):
        raise ConfigurationError("Consecutive Bloch vertices must differ.")
    counts = np.ones(len(lengths), dtype=int)
    for _ in range(num_points - 1 - len(lengths)):
        counts[np.argmax(lengths / counts)] += 1
    segments = (
        np.linspace(a, b, n, endpoint=False) for a, b, n in zip(pts[:-1], pts[1:], counts, strict=True)
    )
    return np.vstack([*segments, pts[-1:]]).T


class BandStructureSolver2D(GridSolver):
    """Bloch eigenfrequencies of a rectangular cell periodic in x and y.

    Parameters
    ----------
    x_range, y_range : (float, float)
        One period along each axis, in metres.
    background_material : Material
        Fills the cell before objects are added.

    Notes
    -----
    Materials must be isotropic (scalar epsilon and mu). Conductors,
    surface impedances and PML are not supported.
    """

    family = "bands"
    _physical_axes = ("x", "y")
    _periodic_axes = ("x", "y")
    _supports_pec = False
    _supports_pmc = False
    _supports_pml = False
    _material_form = "scalar"

    def __init__(self, *, x_range, y_range, background_material=materials.vacuum):
        self._init_grid(ranges=(x_range, y_range), background_material=background_material)

    def _make_kernel(self, resolution):
        from ._kernel import BandKernel

        eps, mu = materials.bulk_values(self.background_material, form="scalar")
        (x0, x1), (y0, y1) = self._ranges
        return BandKernel(x1 - x0, y1 - y0, *resolution, background_epsilon=eps, background_mu=mu)

    def _populate(self, kernel, grid):
        cx, cy = ((lo + hi) / 2 for lo, hi in self._ranges)
        for record in self._objects.values():
            eps, mu = materials.bulk_values(record.material, form="scalar")
            kernel.apply_material(
                lambda x, y, shape=record.shape: shape.contains(x + cx, y + cy), eps, mu, grid.subpixels
            )

    def add_rectangle(self, *, x_range, y_range, material, name=None, clip=False):
        """Add an axis-aligned rectangle (metres); see :meth:`add_geometry`."""
        return self.add_geometry(
            shape=shapes.Rectangle(bounds=(x_range, y_range)), material=material, name=name, clip=clip
        )

    def add_circle(self, *, center, radius, material, name=None, clip=False):
        """Add a circle (metres); see :meth:`add_geometry`."""
        return self.add_geometry(
            shape=shapes.Circle(center=center, radius=radius), material=material, name=name, clip=clip
        )

    def add_polygon(self, *, points, material, name=None, clip=False):
        """Add a polygon with vertices in metres; see :meth:`add_geometry`."""
        return self.add_geometry(shape=shapes.Polygon(points=points), material=material, name=name, clip=clip)

    def solve(
        self,
        *,
        beta_path,
        num_modes=4,
        polarization="both",
        guess=None,
        tol=0.0,
        device="cpu",
        method="arpack",
        progress=True,
    ):
        """Solve the band structure along a Bloch path.

        Parameters
        ----------
        beta_path : array_like
            Bloch vectors in rad/m, shape ``(2, points)``; see :func:`bloch_path`.
        num_modes : int
            Number of bands per point, lowest ``Re(f)`` first.
        polarization : {"TE", "TM", "both"}
            TM has (Ez, Hx, Hy), TE has (Hz, Ex, Ey).
        guess : float, optional
            Frequency in hertz near which bands are sought; ``None`` returns
            the lowest bands.
        tol : float
            Eigensolver tolerance; 0 means machine precision.
        device : {"auto", "cpu", "cuda"}
        method : {"arpack", "refined"}
        progress : bool
            Show a progress bar over the path.

        Returns
        -------
        BandResult
            ``frequencies[pol]`` with shape ``(num_modes, points)`` in hertz.
        """
        num_modes = check_num_modes(num_modes)
        polarization = check_polarization(polarization, ("TE", "TM", "both"))
        guess = check_guess(guess)
        tol = check_tol(tol)
        device = eigs.check_eigen_device(device)
        eigs.check_method(method)
        path = np.asarray(beta_path, dtype=float)
        if path.ndim != 2 or path.shape[0] != 2 or not path.shape[1] or not np.isfinite(path).all():
            raise ConfigurationError("beta_path must be a finite array of shape (2, points) in rad/m.")
        self._ensure_mesh()
        self._result = None
        if guess is None:
            # Below every band, so the lowest bands are nearest. A shift on
            # the static Gamma-point mode (sigma = 0) would make the
            # shift-invert spectrum so wide that other bands lose accuracy.
            period = max(hi - lo for lo, hi in self._ranges)
            sigma = -((np.pi / period) ** 2)
        else:
            sigma = (2 * np.pi * guess / C0) ** 2
        polarizations = ("TE", "TM") if polarization == "both" else (polarization,)
        frequencies, residuals = self._kernel.solve(
            path,
            num_modes,
            polarizations=polarizations,
            sigma=sigma,
            tol=tol,
            device=device,
            method=method,
            progress=progress,
        )
        info = SolveInfo(
            device=device,
            method=method,
            tol=tol,
            guess=guess,
            residuals=np.stack([residuals[p] for p in polarizations]),
        )
        self._result = BandResult(
            family=self.family,
            grid=self.grid,
            info=info,
            beta_path=path,
            frequencies=frequencies,
            metadata={"context": self._scene_context()},
        )
        return self._result

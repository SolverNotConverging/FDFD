"""Periodic (Bloch) mode solvers: complex propagation constants of a unit cell."""

import numpy as np

from ..core import eigs, materials, shapes
from ..core.errors import ConfigurationError
from ..core.grid import GridSolver, check_guess, check_num_modes, check_polarization, check_tol
from ..core.results import ModeResult, SolveInfo

_SOLVE_DOC = """
        Parameters
        ----------
        num_modes : int
            Number of Bloch modes returned, nearest to ``guess`` first.{polarization}
        guess : complex, optional
            Effective index near which modes are sought (shift
            ``gamma = i*k0*guess``). ``None`` uses 1.01 times the largest
            material index. Leaky modes usually need ``guess=0``.
        tol : float
            Eigensolver tolerance; 0 means machine precision (the refined
            method uses at least 1e-12).
        device : {{"auto", "cpu", "hybrid", "gpu", "cuda"}}
            Auto selects hybrid GPU Arnoldi/CPU Ritz; gpu/cuda require resident CUDA Ritz.
        method : {{"arpack", "refined"}}
            Krylov method; "refined" is more robust for difficult 3D cells.
        ncv : int, optional
            Krylov subspace size.
        max_restarts, random_seed : int
            Restart limit and start-vector seed of the refined method.

        Returns
        -------
        ModeResult
            Complex ``neff`` (``Im(neff) <= 0`` for passive forward modes) and
            the periodic-envelope fields.

        Raises
        ------
        SolverError
            If a returned eigenpair's relative residual exceeds
            ``max(1e-8, 100 * tol)``.
"""


class _PeriodicSolver(GridSolver):
    family = "periodic"
    _periodic_axes = ("z",)

    def _start(self, num_modes, guess, tol, device, method):
        num_modes = check_num_modes(num_modes)
        guess = check_guess(guess)
        tol = check_tol(tol)
        device = eigs.check_eigen_device(device)
        eigs.check_method(method)
        self._ensure_mesh()
        self._result = None
        if guess is None:
            media = [self.background_material]
            media += [
                r.material for r in self._objects.values() if isinstance(r.material, materials.Material)
            ]
            guess = 1.01 * materials.max_index(media)
        return num_modes, complex(guess), tol, device

    def _finish(self, gamma, fields, residuals, *, polarizations, guess, tol, device, method):
        kernel = self._kernel
        info = SolveInfo(device=device, method=method, tol=tol, guess=guess, residuals=np.asarray(residuals))
        metadata = {
            "polarizations": tuple(polarizations),
            "field_representation": "periodic envelope; Bloch factor exp(-gamma*z)",
            "field_normalization": "eigenvector (SI E and H)",
            "material_background": self._material_background(),
            "context": self._scene_context(),
            "pml": tuple(dict(p) for p in self._pmls),
        }
        self._result = ModeResult(
            family=self.family,
            grid=self.grid,
            info=info,
            metadata=metadata,
            frequency=self.frequency,
            fields=fields,
            field_coordinates=self._field_coordinates(fields, periodic_nodes=("Ex", "Ey", "Hz")),
            neff=-1j * np.asarray(gamma) / kernel.k0,
        )
        return self._result


class PeriodicModeSolver2D(_PeriodicSolver):
    """Bloch modes of an (x, z) unit cell, periodic in z and invariant in y.

    Parameters
    ----------
    frequency : float
        Frequency in hertz.
    x_range, z_range : (float, float)
        Cell extent in metres; ``z_range`` spans one period.
    background_material : Material
        Fills the cell before objects are added.
    boundary : materials.PEC or materials.PMC
        Closure of the two x ends of the domain.
    """

    _physical_axes = ("x", "z")

    def __init__(
        self, *, frequency, x_range, z_range, background_material=materials.vacuum, boundary=materials.PEC
    ):
        if boundary not in (materials.PEC, materials.PMC):
            raise ConfigurationError("boundary must be materials.PEC or materials.PMC.")
        self.boundary = boundary
        self._init_grid(
            ranges=(x_range, z_range), background_material=background_material, frequency=frequency
        )

    def _make_kernel(self, resolution):
        from ._kernel_2d import PeriodicKernel2D

        (x0, x1), (z0, z1) = self._ranges
        return PeriodicKernel2D(self.frequency, x1 - x0, z1 - z0, *resolution, boundary=self.boundary.kind)

    def add_rectangle(self, *, x_range, z_range, material, name=None, clip=False):
        """Add an axis-aligned rectangle (metres); see :meth:`add_geometry`."""
        return self.add_geometry(
            shape=shapes.Rectangle(bounds=(x_range, z_range)), material=material, name=name, clip=clip
        )

    def add_circle(self, *, center, radius, material, name=None, clip=False):
        """Add a circle centred at ``(x, z)`` (metres); see :meth:`add_geometry`."""
        return self.add_geometry(
            shape=shapes.Circle(center=center, radius=radius), material=material, name=name, clip=clip
        )

    def add_polygon(self, *, points, material, name=None, clip=False):
        """Add a polygon with ``(x, z)`` vertices in metres; see :meth:`add_geometry`."""
        return self.add_geometry(shape=shapes.Polygon(points=points), material=material, name=name, clip=clip)

    def solve(
        self,
        *,
        num_modes=4,
        polarization="both",
        guess=None,
        tol=0.0,
        device="cpu",
        method="arpack",
        ncv=None,
        max_restarts=12,
        random_seed=0,
    ):
        polarization = check_polarization(polarization, ("TE", "TM", "both"))
        num_modes, guess, tol, device = self._start(num_modes, guess, tol, device, method)
        sigma = 1j * self._kernel.k0 * guess
        options = dict(
            tol=tol, device=device, method=method, ncv=ncv, max_restarts=max_restarts, random_seed=random_seed
        )
        candidates = []
        for pol in ("TE", "TM") if polarization == "both" else (polarization,):
            gamma, fields, residuals = self._kernel.solve(num_modes, sigma, polarization=pol, **options)
            candidates += [
                (abs(gamma[i] - sigma), pol, gamma[i], fields, residuals[i], i) for i in range(num_modes)
            ]
        candidates.sort(key=lambda c: c[0])
        selected = candidates[:num_modes]
        fields = {
            name: np.stack([c[3][name][..., c[5]] for c in selected], axis=-1) for name in selected[0][3]
        }
        return self._finish(
            [c[2] for c in selected],
            fields,
            [c[4] for c in selected],
            polarizations=[c[1] for c in selected],
            guess=guess,
            tol=tol,
            device=device,
            method=method,
        )

    solve.__doc__ = "Solve for Bloch modes.\n" + _SOLVE_DOC.format(
        polarization="""
        polarization : {"TE", "TM", "both"}
            TE has (Hx, Ey, Hz), TM has (Ex, Hy, Ez). With "both", each
            polarization is solved and the nearest ``num_modes`` are kept."""
    )


class PeriodicModeSolver3D(_PeriodicSolver):
    """Full-vector Bloch modes of an (x, y, z) unit cell, periodic in z.

    Parameters
    ----------
    frequency : float
        Frequency in hertz.
    x_range, y_range, z_range : (float, float)
        Cell extent in metres; ``z_range`` spans one period.
    background_material : Material
        Fills the cell before objects are added.

    Notes
    -----
    The transverse outer boundary is the natural Yee closure (PMC).
    """

    _physical_axes = ("x", "y", "z")

    def __init__(self, *, frequency, x_range, y_range, z_range, background_material=materials.vacuum):
        self._init_grid(
            ranges=(x_range, y_range, z_range), background_material=background_material, frequency=frequency
        )

    def _make_kernel(self, resolution):
        from ._kernel_3d import PeriodicKernel3D

        return PeriodicKernel3D(self.frequency, *(hi - lo for lo, hi in self._ranges), *resolution)

    def add_box(self, *, x_range, y_range, z_range, material, name=None, clip=False):
        """Add an axis-aligned box (metres); see :meth:`add_geometry`."""
        return self.add_geometry(
            shape=shapes.Box(bounds=(x_range, y_range, z_range)), material=material, name=name, clip=clip
        )

    def add_sphere(self, *, center, radius, material, name=None, clip=False):
        """Add a sphere (metres); see :meth:`add_geometry`."""
        return self.add_geometry(
            shape=shapes.Sphere(center=center, radius=radius), material=material, name=name, clip=clip
        )

    def add_cylinder(self, *, center, radius, z_range, material, name=None, clip=False):
        """Add a z-directed cylinder with ``(x, y)`` centre (metres); see :meth:`add_geometry`."""
        return self.add_geometry(
            shape=shapes.Cylinder(center=center, radius=radius, z_range=z_range),
            material=material,
            name=name,
            clip=clip,
        )

    def solve(
        self,
        *,
        num_modes=4,
        guess=None,
        tol=0.0,
        device="cpu",
        method="arpack",
        ncv=None,
        max_restarts=12,
        random_seed=0,
    ):
        num_modes, guess, tol, device = self._start(num_modes, guess, tol, device, method)
        gamma, fields, residuals = self._kernel.solve(
            num_modes,
            1j * self._kernel.k0 * guess,
            tol=tol,
            device=device,
            method=method,
            ncv=ncv,
            max_restarts=max_restarts,
            random_seed=random_seed,
        )
        return self._finish(
            gamma,
            fields,
            residuals,
            polarizations=("vector",) * num_modes,
            guess=guess,
            tol=tol,
            device=device,
            method=method,
        )

    solve.__doc__ = "Solve for full-vector Bloch modes.\n" + _SOLVE_DOC.format(polarization="")

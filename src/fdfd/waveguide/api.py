"""Waveguide mode solvers: guided modes of a z-invariant cross-section."""

import numpy as np

from ..core import eigs, materials, shapes
from ..core.constants import ETA_0, SPATIAL_CONVENTION
from ..core.grid import GridSolver, check_guess, check_num_modes, check_polarization, check_tol
from ..core.results import ModeResult, SolveInfo

_SOLVE_DOC = """
        Parameters
        ----------
        num_modes : int
            Number of modes returned, ordered by descending ``Re(neff)``.{polarization}
        guess : complex, optional
            Effective index near which modes are sought. ``None`` searches just
            above the largest material index, which returns the fundamental
            modes first.
        tol : float
            Eigensolver tolerance; 0 means machine precision.
        device : {{"auto", "cpu", "hybrid", "gpu", "cuda"}}
            Auto selects hybrid GPU Arnoldi/CPU Ritz; gpu/cuda require resident CUDA Ritz.
        method : {{"arpack", "refined"}}
            Krylov method, see :func:`fdfd.core.eigs.solve`.

        Returns
        -------
        ModeResult
            ``neff`` and the six staggered field components. Stored magnetic
            fields are ``-i*eta0*H`` (same units as E).
"""


def _boundary_provenance(kernel):
    """Which outer edges are closed by conductor cells."""
    closed = kernel._pec_cell_mask | kernel._pmc_cell_mask | (kernel._surface_impedance_owner >= 0)
    edges = {}
    for axis, label in enumerate(("x", "y")[: closed.ndim]):
        for index, side in ((0, "-"), (-1, "+")):
            edges[label + side] = bool(np.all(np.take(closed, index, axis=axis)))
    return {"conductor_edges": edges, "closed_conductor_boundary": all(edges.values())}


class _WaveguideSolver(GridSolver):
    family = "waveguide"
    _supports_sibc = True

    def _search_guess(self, guess):
        media = [self.background_material]
        media += [r.material for r in self._objects.values() if isinstance(r.material, materials.Material)]
        bound = 1.01 * materials.max_index(media)
        return bound if guess is None else guess

    def _solve(
        self,
        *,
        num_modes,
        guess,
        tol,
        device,
        method,
        polarizations,
        wide_search=False,
        cutoff_tolerance=1e-8,
    ):
        num_modes = check_num_modes(num_modes)
        guess = check_guess(guess)
        tol = check_tol(tol)
        device = eigs.check_eigen_device(device)
        eigs.check_method(method)
        self._ensure_mesh()
        self._result = None
        search = self._search_guess(guess)
        options = {"tol": tol, "device": device, "method": method}
        if len(self._physical_axes) == 1:
            options["polarizations"] = polarizations
        else:
            options.update(wide_search=wide_search, cutoff_tolerance=cutoff_tolerance)
        solutions = self._kernel.solve(num_modes, -(complex(search) ** 2), **options)
        candidates = [(s, i) for s in solutions for i in range(num_modes)]
        candidates.sort(key=lambda c: (-c[0].neff[c[1]].real, abs(c[0].neff[c[1]].imag)))
        selected = candidates[:num_modes]
        fields = {
            name: np.stack([s.fields[name][..., i] for s, i in selected], axis=-1)
            for name in solutions[0].fields
        }
        neff = np.array([s.neff[i] for s, i in selected])
        info = SolveInfo(
            device=device,
            method=method,
            tol=tol,
            guess=complex(search),
            residuals=np.array([s.residuals[i] for s, i in selected]),
            field_residuals=np.array([s.field_residuals[i] for s, i in selected]),
        )
        metadata = {
            "polarizations": tuple(s.polarization for s, _ in selected),
            "field_normalization": "eigenvector; stored H is -i*eta0*H",
            "spatial_convention": SPATIAL_CONVENTION,
            "eta0": ETA_0,
            "material_background": self._material_background(),
            "context": self._scene_context(),
            "pml": tuple(dict(p) for p in self._pmls),
            "boundaries": _boundary_provenance(self._kernel),
        }
        self._result = ModeResult(
            family=self.family,
            grid=self.grid,
            info=info,
            metadata=metadata,
            frequency=self.frequency,
            fields=fields,
            field_coordinates=self._field_coordinates(fields),
            neff=neff,
        )
        return self._result


class ModeSolver1D(_WaveguideSolver):
    """TE/TM modes of a planar (slab) waveguide layered along x.

    Parameters
    ----------
    frequency : float
        Frequency in hertz.
    x_range : (float, float)
        Domain extent in metres.
    background_material : Material
        Fills the domain before layers are added.

    Notes
    -----
    The outer ends of the domain are the natural Yee closure (tangential H
    = 0, i.e. PMC). Add PEC/PMC layers for explicit walls, or a PML.
    """

    _physical_axes = ("x",)

    def __init__(self, *, frequency, x_range, background_material=materials.vacuum):
        self._init_grid(ranges=(x_range,), background_material=background_material, frequency=frequency)

    def _make_kernel(self, resolution):
        from ._kernel_1d import WaveguideKernel1D

        ((lo, hi),) = self._ranges
        return WaveguideKernel1D(self.frequency, hi - lo, resolution[0])

    def add_layer(self, *, x_range, material, name=None, clip=False):
        """Fill ``x_range`` (metres) with ``material``; see :meth:`add_geometry`."""
        return self.add_geometry(
            shape=shapes.Interval(bounds=x_range), material=material, name=name, clip=clip
        )

    def solve(self, *, num_modes=4, polarization="both", guess=None, tol=0.0, device="cpu", method="arpack"):
        polarization = check_polarization(polarization, ("TE", "TM", "both"))
        return self._solve(
            num_modes=num_modes,
            guess=guess,
            tol=tol,
            device=device,
            method=method,
            polarizations=("TE", "TM") if polarization == "both" else (polarization,),
        )

    solve.__doc__ = "Solve for slab modes.\n" + _SOLVE_DOC.format(
        polarization="""
        polarization : {"TE", "TM", "both"}
            TE has (Ey, Hx, Hz), TM has (Hy, Ex, Ez). With "both", the
            ``num_modes`` modes with largest ``Re(neff)`` are returned."""
    )


class ModeSolver2D(_WaveguideSolver):
    """Full-vector modes of a waveguide with an (x, y) cross-section.

    Parameters
    ----------
    frequency : float
        Frequency in hertz.
    x_range, y_range : (float, float)
        Domain extent in metres.
    background_material : Material
        Fills the domain before objects are added.

    Notes
    -----
    The outer boundary is the natural Yee closure (PMC). Use conductor
    objects for physical walls or :meth:`add_pml` for open boundaries.
    """

    _physical_axes = ("x", "y")

    def __init__(self, *, frequency, x_range, y_range, background_material=materials.vacuum):
        self._init_grid(
            ranges=(x_range, y_range), background_material=background_material, frequency=frequency
        )

    def _make_kernel(self, resolution):
        from ._kernel_2d import WaveguideKernel2D

        (x0, x1), (y0, y1) = self._ranges
        return WaveguideKernel2D(self.frequency, x1 - x0, y1 - y0, *resolution)

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
        """Add a simple polygon with vertices in metres; see :meth:`add_geometry`."""
        return self.add_geometry(shape=shapes.Polygon(points=points), material=material, name=name, clip=clip)

    def solve(self, *, num_modes=4, guess=None, tol=0.0, device="cpu", method="arpack"):
        return self._solve(
            num_modes=num_modes, guess=guess, tol=tol, device=device, method=method, polarizations=("vector",)
        )

    solve.__doc__ = "Solve for full-vector modes.\n" + _SOLVE_DOC.format(polarization="")

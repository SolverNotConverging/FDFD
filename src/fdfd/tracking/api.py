"""Frequency sweeps of waveguide modes with tracked branch identities."""

from dataclasses import replace

import numpy as np

from ..core import eigs, materials, shapes
from ..core.errors import ConfigurationError
from ..core.grid import GridSolver, check_guess, check_num_modes, check_polarization, check_tol
from ..waveguide import ModeSolver1D, ModeSolver2D
from .contracts import PortSpec, TrackingConfig
from .sweep import track_modes

DEFAULT_PORT = PortSpec()


class _ModeTracker(GridSolver):
    """Geometry and mesh are shared by every sweep frequency; no PML is allowed."""

    family = "tracking"
    _supports_sibc = True
    _supports_pml = False
    _solver_type = None

    def _init_tracker(self, *, frequencies, ranges, background_material, port):
        values = np.asarray(frequencies, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ConfigurationError(
                "frequencies must be a nonempty sequence of positive finite hertz values."
            )
        if len(np.unique(values)) != len(values):
            raise ConfigurationError("frequencies must be unique.")
        if not isinstance(port, PortSpec):
            raise ConfigurationError("port must be a PortSpec.")
        self._init_grid(ranges=ranges, background_material=background_material)
        self.frequencies = tuple(float(value) for value in np.sort(values))
        self.port = port

    def mesh(self, *, resolution=None, max_element_size=None, subpixels=8):
        """Set the base Yee grid used at every sweep frequency.

        Parameters are those of :meth:`fdfd.ModeSolver2D.mesh`. Verification
        solves refine this grid or pad it at the same spacing.
        """
        self._invalidate()
        self._mesh_settings = self.grid = self._resolve_mesh(resolution, max_element_size, subpixels)
        return self.grid

    def _ensure_mesh(self):
        # The grid does not depend on the geometry, so edits keep it.
        if self.grid is None:
            self.grid = self._mesh_settings if self._mesh_settings is not None else self.mesh()

    def _factory(self, frequency, verification):
        self._ensure_mesh()
        padded_ranges, resolution = [], []
        for (lo, hi), cells in zip(self._ranges, self.grid.resolution, strict=True):
            step = (hi - lo) / cells
            padding = round(cells * verification.padding_fraction)
            padded_ranges.append((lo - padding * step, hi + padding * step))
            resolution.append((cells + 2 * padding) * verification.mesh_factor)
        arguments = {"frequency": frequency, "background_material": self.background_material}
        arguments.update(
            {axis + "_range": span for axis, span in zip(self._physical_axes, padded_ranges, strict=True)}
        )
        solver = self._solver_type(**arguments)
        for record in self._objects.values():
            solver.add_geometry(
                shape=record.shape, material=record.material, name=record.name, clip=record.clip
            )
        solver.mesh(resolution=tuple(resolution), subpixels=self.grid.subpixels)
        return solver

    def _track(self, *, num_modes, polarization, guess, tol, device, reference_frequency, config, progress):
        num_modes = check_num_modes(num_modes)
        guess = check_guess(guess)
        tol = check_tol(tol)
        device = eigs.check_eigen_device(device)
        if config is not None and not isinstance(config, TrackingConfig):
            raise ConfigurationError("config must be a TrackingConfig.")
        self._ensure_mesh()
        config = replace(
            config or TrackingConfig(),
            num_candidates=num_modes,
            max_candidates=num_modes,
            guess=guess,
            polarization=polarization,
            tol=tol,
        )
        self._result = track_modes(
            self._factory,
            self.frequencies,
            port=self.port,
            config=config,
            seed_modes=None,
            reference_frequency=reference_frequency,
            progress=progress,
            device=device,
        )
        return self._result

    _SOLVE_DOC = """
        Every candidate returned at each frequency is tracked; new branches
        are added when they appear, and each base solve is verified by an
        independent refined-mesh (and, for open ports, padded) solve.

        Parameters
        ----------
        num_modes : int
            Candidates solved at each frequency.{polarization}
        guess : complex, optional
            Effective index to search near; ``None`` uses the largest material
            index (fundamental modes first).
        tol : float
            Eigensolver tolerance; 0 means machine precision.
        device : {{"auto", "cpu", "hybrid", "gpu", "cuda"}}
        reference_frequency : float, optional
            Requested frequency at which tracking starts (default: the lowest).
        config : TrackingConfig, optional
            Assignment and verification thresholds.
        progress : bool
            Show a progress bar.

        Returns
        -------
        TrackedSweep
"""


class ModeTracker1D(_ModeTracker):
    """Track slab modes (layered along x) across a frequency sweep.

    Parameters
    ----------
    frequencies : sequence of float
        Requested frequencies in hertz (sorted internally).
    x_range : (float, float)
        Domain extent in metres.
    background_material : Material
    port : PortSpec
        ``boundary="enclosed"`` when the outer conductors are physical walls,
        ``"open"`` when the domain edges are artificial.
    """

    _physical_axes = ("x",)
    _solver_type = ModeSolver1D

    def __init__(self, *, frequencies, x_range, background_material=materials.vacuum, port=DEFAULT_PORT):
        self._init_tracker(
            frequencies=frequencies, ranges=(x_range,), background_material=background_material, port=port
        )

    def add_layer(self, *, x_range, material, name=None, clip=False):
        """Fill ``x_range`` (metres) with ``material``; see :meth:`add_geometry`."""
        return self.add_geometry(
            shape=shapes.Interval(bounds=x_range), material=material, name=name, clip=clip
        )

    def solve(
        self,
        *,
        num_modes=4,
        polarization="both",
        guess=None,
        tol=0.0,
        device="cpu",
        reference_frequency=None,
        config=None,
        progress=True,
    ):
        polarization = check_polarization(polarization, ("TE", "TM", "both"))
        return self._track(
            num_modes=num_modes,
            polarization=polarization,
            guess=guess,
            tol=tol,
            device=device,
            reference_frequency=reference_frequency,
            config=config,
            progress=progress,
        )

    solve.__doc__ = "Solve and track slab modes over the sweep.\n" + _ModeTracker._SOLVE_DOC.format(
        polarization="""
        polarization : {"TE", "TM", "both"}
            Polarizations solved at each frequency."""
    )


class ModeTracker2D(_ModeTracker):
    """Track full-vector modes of an (x, y) cross-section across a frequency sweep.

    Parameters
    ----------
    frequencies : sequence of float
        Requested frequencies in hertz (sorted internally).
    x_range, y_range : (float, float)
        Domain extent in metres.
    background_material : Material
    port : PortSpec
        ``boundary="enclosed"`` when the outer conductors are physical walls,
        ``"open"`` when the domain edges are artificial.
    """

    _physical_axes = ("x", "y")
    _solver_type = ModeSolver2D

    def __init__(
        self, *, frequencies, x_range, y_range, background_material=materials.vacuum, port=DEFAULT_PORT
    ):
        self._init_tracker(
            frequencies=frequencies,
            ranges=(x_range, y_range),
            background_material=background_material,
            port=port,
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
        num_modes=4,
        guess=None,
        tol=0.0,
        device="cpu",
        reference_frequency=None,
        config=None,
        progress=True,
    ):
        return self._track(
            num_modes=num_modes,
            polarization="both",
            guess=guess,
            tol=tol,
            device=device,
            reference_frequency=reference_frequency,
            config=config,
            progress=progress,
        )

    solve.__doc__ = "Solve and track full-vector modes over the sweep.\n" + _ModeTracker._SOLVE_DOC.format(
        polarization=""
    )

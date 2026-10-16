"""2D plane-wave and line-source scattering with a total-field/scattered-field split."""

import numpy as np

from ..core import eigs, materials, shapes
from ..core.errors import ConfigurationError
from ..core.grid import GridSolver, check_polarization
from ..core.materials import _positive
from ..core.results import ScatteringResult, SolveInfo


class ScatteringSolver2D(GridSolver):
    """Scattering by objects in an (x, y) domain, invariant along z.

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
    The outer boundary is PEC; add a PML (:meth:`add_pml`) to model open
    space. PEC objects are supported; PMC and surface impedances are not.
    """

    family = "scattering"
    _physical_axes = ("x", "y")
    _supports_pmc = False

    def __init__(self, *, frequency, x_range, y_range, background_material=materials.vacuum):
        self._source = None
        self._init_grid(
            ranges=(x_range, y_range), background_material=background_material, frequency=frequency
        )

    def _make_kernel(self, resolution):
        from ._kernel import ScatteringKernel

        (x0, x1), (y0, y1) = self._ranges
        return ScatteringKernel(self.frequency, x1 - x0, y1 - y0, *resolution)

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

    def set_source(self, *, inset, kind="plane_wave", angle=0.0, location=None, amplitude=1.0):
        """Define the incident field and the total-field region.

        Parameters
        ----------
        inset : float
            Distance in metres from the domain boundary to the rectangular
            total-field region. Scatterers must lie inside that region and
            the PML outside it.
        kind : {"plane_wave", "point"}
            A plane wave, or an outgoing line source (Hankel function).
        angle : float
            Plane-wave direction in degrees from +x.
        location : (float, float)
            Line-source position in metres (required for ``kind="point"``).
        amplitude : complex
            Incident amplitude of the primary field (Ez for TE, Hz for TM)
            at the domain origin.
        """
        if kind not in ("plane_wave", "point"):
            raise ConfigurationError("kind must be 'plane_wave' or 'point'.")
        if not np.isfinite(angle) or not np.isfinite(amplitude):
            raise ConfigurationError("angle and amplitude must be finite.")
        if kind == "point":
            if location is None or len(location) != 2 or not np.isfinite(location).all():
                raise ConfigurationError("A point source needs a finite (x, y) location in metres.")
            location = (float(location[0]), float(location[1]))
        elif location is not None:
            raise ConfigurationError("location applies only to point sources.")
        self._source = dict(
            kind=kind,
            angle=float(angle),
            location=location,
            amplitude=complex(amplitude),
            inset=_positive(inset, "inset"),
        )
        self._result = None

    def solve(self, *, polarization="TE", device="cpu"):
        """Solve for the total field.

        Parameters
        ----------
        polarization : {"TE", "TM"}
            TE has (Ez, Hx, Hy), TM has (Hz, Ex, Ey).
        device : {"auto", "cpu", "hybrid", "gpu", "cuda"}
            Where the sparse system is factorized. Factorizations are
            reused between solves until the geometry or mesh changes.

        Returns
        -------
        ScatteringResult
            The three field components (total field inside the total-field
            region, scattered field outside), in V/m and A/m.
        """
        polarization = check_polarization(polarization, ("TE", "TM"))
        device = eigs.check_device(device)
        if self._source is None:
            raise ConfigurationError("Call set_source() before solve().")
        self._ensure_mesh()
        self._result = None
        kernel = self._kernel
        source = dict(self._source)
        centre = np.array([0.5 * (lo + hi) for lo, hi in self._ranges])
        if source["kind"] == "point":
            location = tuple(np.asarray(source["location"]) - centre)
            amplitude = source["amplitude"]
        else:
            # The kernel measures phase from the domain centre; keep the
            # requested amplitude at the physical origin.
            location = None
            theta = np.deg2rad(source["angle"])
            amplitude = source["amplitude"] * np.exp(
                -1j * kernel.k0 * np.dot(centre, (np.cos(theta), np.sin(theta)))
            )
        incident = kernel.incident(
            polarization, kind=source["kind"], angle=source["angle"], location=location, amplitude=amplitude
        )
        masks = kernel.total_field_masks(source["inset"])
        primary = "Ez" if polarization == "TE" else "Hz"
        outside = masks[primary].diagonal() != 0
        if outside.all():
            raise ConfigurationError("The source inset leaves no total-field region.")
        if np.any(kernel.pec_field_masks()[primary].ravel(order="F") & outside):
            raise ConfigurationError("The total-field region must enclose every PEC object.")
        fields, residual = kernel.solve(polarization, incident, masks, device=device)
        fields = {name: np.array(values[..., None], copy=True) for name, values in fields.items()}
        info = SolveInfo(device=device, method="direct", tol=0.0, guess=None, residuals=np.array([residual]))
        metadata = {
            "field_representation": "total field inside the TF/SF region, scattered field outside",
            "source": source,
            "material_background": self._material_background(),
            "context": self._scene_context(),
            "pml": tuple(dict(p) for p in self._pmls),
        }
        self._result = ScatteringResult(
            family=self.family,
            grid=self.grid,
            info=info,
            metadata=metadata,
            frequency=self.frequency,
            fields=fields,
            field_coordinates=self._field_coordinates(fields),
            polarization=polarization,
        )
        return self._result

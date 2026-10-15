"""Shared machinery for every Yee-grid solver.

:class:`GridSolver` owns the domain, geometry, mesh settings and PML requests
of a public solver; each family adds a *kernel* that assembles and solves
its operators. Kernels store cell-centred material tensors through
:class:`CellMaterials`, so meshing, subpixel smoothing, conductors and the PML
are applied the same way in every family.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from . import materials, pml, shapes
from .errors import BackendCapabilityError, ConfigurationError, GeometryError, NoResultError
from .materials import _positive
from .scene import Scene


def bounds(value, name="range"):
    """Validate a ``(min, max)`` pair in metres."""
    try:
        pair = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a (min, max) pair in metres.") from exc
    if len(pair) != 2 or not np.isfinite(pair).all() or pair[1] <= pair[0]:
        raise ConfigurationError(f"{name} must be a finite increasing (min, max) pair in metres.")
    return pair


@dataclass(frozen=True)
class GridData:
    """A uniform Yee grid.

    Attributes
    ----------
    axes : tuple of str
        Physical axis names, e.g. ``("x", "y")``.
    bounds : tuple of (float, float)
        Domain extent along each axis, in metres.
    resolution : tuple of int
        Number of cells along each axis.
    subpixels : int
        Subpixel samples per cell edge used to smooth material interfaces.
    """

    axes: tuple
    bounds: tuple
    resolution: tuple
    subpixels: int = 1

    @property
    def spacing(self):
        """Cell size along each axis in metres."""
        return tuple((hi - lo) / n for (lo, hi), n in zip(self.bounds, self.resolution, strict=True))

    @property
    def coordinates(self):
        """Cell-centre coordinates along each axis; fields carry their own staggered axes."""
        return tuple(
            lo + (np.arange(n) + 0.5) * (hi - lo) / n
            for (lo, hi), n in zip(self.bounds, self.resolution, strict=True)
        )


def fractions(shape, spans, resolution, subpixels):
    """Fraction of each cell inside ``shape`` by midpoint subpixel sampling.

    Returns ``(fraction, slices)``: fractions for the cells inside the shape's
    bounding box and the slices locating that box in the full grid.
    """
    b = shape.bounds
    starts = [
        max(0, int(np.floor((lo - span[0]) * n / (span[1] - span[0]))))
        for lo, span, n in zip(b[::2], spans, resolution, strict=True)
    ]
    stops = [
        min(n, int(np.ceil((hi - span[0]) * n / (span[1] - span[0]))))
        for hi, span, n in zip(b[1::2], spans, resolution, strict=True)
    ]
    slices = tuple(slice(a, max(a, z)) for a, z in zip(starts, stops, strict=True))
    result = np.zeros(tuple(max(0, z - a) for a, z in zip(starts, stops, strict=True)))
    if not result.size:
        return result, slices
    base = [
        span[0] + np.arange(a, z) * (span[1] - span[0]) / n
        for a, z, span, n in zip(starts, stops, spans, resolution, strict=True)
    ]
    steps = [(hi - lo) / n for (lo, hi), n in zip(spans, resolution, strict=True)]
    for offset in product(range(subpixels), repeat=len(spans)):
        points = np.meshgrid(
            *(axis + (i + 0.5) * step / subpixels for axis, i, step in zip(base, offset, steps, strict=True)),
            indexing="ij",
            sparse=True,
        )
        result += shape.contains(*points)
    return result / subpixels ** len(spans), slices


class CellMaterials:
    """Cell-centred relative ``epsilon``/``mu`` tensors held by a kernel.

    Kernels call :meth:`_init_materials` and implement
    ``update_component_materials()``, which interpolates these cell values to
    the staggered Yee components.
    """

    def _init_materials(self, shape):
        self.shape_cell = tuple(shape)
        for quantity in ("eps", "mu"):
            for component in ("xx", "yy", "zz"):
                setattr(self, f"cell_{quantity}_r_{component}", np.ones(self.shape_cell, dtype=complex))
        self._pml_cell_mask = np.zeros(self.shape_cell, dtype=bool)

    def cell_tensors(self):
        """``((eps_xx, eps_yy, eps_zz), (mu_xx, mu_yy, mu_zz))`` cell arrays (writable)."""
        return tuple(
            tuple(getattr(self, f"cell_{q}_r_{c}") for c in ("xx", "yy", "zz")) for q in ("eps", "mu")
        )

    def apply_fractional_material(self, epsilon, mu, fraction, *slices):
        """Blend a diagonal material into the cells in ``slices`` by ``fraction``."""
        from .yee import diagonal3

        fraction = np.asarray(fraction, dtype=float)
        covered = fraction > 0.0
        if not np.any(covered):
            return
        for tensor, values in zip(
            self.cell_tensors(), (diagonal3("epsilon", epsilon), diagonal3("mu", mu)), strict=True
        ):
            for target, value in zip(tensor, values, strict=True):
                local = target[slices]
                local[covered] = local[covered] * (1.0 - fraction[covered]) + value * fraction[covered]

    def set_pml_cells(self, mask):
        """Record which cells contain PML loss (kernels may validate against it)."""
        self._pml_cell_mask = np.asarray(mask, dtype=bool)


class GridSolver(Scene):
    """Base class for the public solvers: domain, geometry, mesh and PML.

    Subclasses define ``family``, ``_physical_axes``, ``_make_kernel`` and the
    capability flags below.
    """

    family = None
    _physical_axes = ()
    _periodic_axes = ()
    _supports_pec = True
    _supports_pmc = True
    _supports_sibc = False
    _supports_pml = True
    _material_form = "diagonal"

    def _init_grid(self, *, ranges, background_material, frequency=None):
        self._init_scene(background_material=background_material)
        self._ranges = tuple(
            bounds(r, axis + "_range") for r, axis in zip(ranges, self._physical_axes, strict=True)
        )
        for axis, span in zip(self._physical_axes, self._ranges, strict=True):
            setattr(self, axis + "_range", span)
        self._frequency = None if frequency is None else _positive(frequency, "frequency")
        self._pmls = []
        self._mesh_settings = None
        self.grid = self._kernel = self._result = None

    @property
    def frequency(self):
        """Operating frequency in hertz (fixed for the lifetime of the solver)."""
        return self._frequency

    @property
    def wavelength(self):
        """Free-space wavelength in metres."""
        from .constants import C0

        return C0 / self._frequency

    @property
    def result(self):
        """The result of the latest :meth:`solve`, or ``None``."""
        return self._result

    def _invalidate(self):
        self.grid = self._kernel = self._result = None

    def _validate_record(self, record):
        if not isinstance(record.shape, shapes.Shape) or record.shape.dimension != len(self._physical_axes):
            raise GeometryError("Shape must match the solver physical axes.")
        outside = any(
            a < lo or b > hi
            for a, b, (lo, hi) in zip(
                record.shape.bounds[::2], record.shape.bounds[1::2], self._ranges, strict=True
            )
        )
        if not record.clip and outside:
            raise GeometryError(f"{record.name!r} lies outside the domain; pass clip=True to allow it.")
        material = record.material
        if isinstance(material, materials.Material):
            materials.bulk_values(material, form=self._material_form)
        elif material == materials.PEC and not self._supports_pec:
            raise BackendCapabilityError(f"{type(self).__name__} does not support PEC objects.")
        elif material == materials.PMC and not self._supports_pmc:
            raise BackendCapabilityError(f"{type(self).__name__} does not support PMC objects.")
        elif (
            isinstance(material, (materials.GoodConductor, materials.SurfaceImpedance))
            and not self._supports_sibc
        ):
            raise BackendCapabilityError(
                f"{type(self).__name__} does not support surface-impedance materials."
            )

    def _resolve_mesh(self, resolution, max_element_size, subpixels):
        dim = len(self._physical_axes)
        if resolution is not None and max_element_size is not None:
            raise ConfigurationError("Specify resolution or max_element_size, not both.")
        if resolution is None:
            if max_element_size is None:
                resolution = (40,) * dim
            else:
                size = _positive(max_element_size, "max_element_size")
                resolution = tuple(max(2, int(np.ceil((hi - lo) / size - 1e-9))) for lo, hi in self._ranges)
        raw = (resolution,) if np.isscalar(resolution) else tuple(resolution)
        if len(raw) != dim or any(isinstance(n, (bool, np.bool_)) or int(n) != n or n < 2 for n in raw):
            raise ConfigurationError(
                f"resolution must give at least two integer cells for each of {self._physical_axes}."
            )
        if isinstance(subpixels, (bool, np.bool_)) or int(subpixels) != subpixels or subpixels < 1:
            raise ConfigurationError("subpixels must be a positive integer.")
        return GridData(self._physical_axes, self._ranges, tuple(int(n) for n in raw), int(subpixels))

    def mesh(self, *, resolution=None, max_element_size=None, subpixels=8):
        """Discretize the domain on a uniform Yee grid.

        Parameters
        ----------
        resolution : int or tuple of int, optional
            Number of cells along each physical axis (an int is accepted for
            one-dimensional solvers). Defaults to 40 per axis.
        max_element_size : float, optional
            Largest cell size in metres; the cell count per axis is rounded
            up. Mutually exclusive with ``resolution``.
        subpixels : int
            Midpoint samples per cell edge used to compute the fraction of
            each cell covered by a dielectric shape. Conductors always use
            the cell centre. Default 8.

        Returns
        -------
        GridData
            The grid, also available as :attr:`grid`.

        Raises
        ------
        ConfigurationError
            For invalid arguments.
        GeometryError
            When a conductor does not cover any cell centre.
        """
        grid = self._resolve_mesh(resolution, max_element_size, subpixels)
        self._invalidate()
        self._mesh_settings = grid
        kernel = self._make_kernel(grid.resolution)
        self._populate(kernel, grid)
        if self._pmls:
            self._apply_pml(kernel, grid)
        self._kernel = kernel
        self.grid = grid
        return grid

    def _ensure_mesh(self):
        if self.grid is None:
            settings = self._mesh_settings
            if settings is None:
                self.mesh()
            else:
                self.mesh(resolution=settings.resolution, subpixels=settings.subpixels)

    def add_pml(self, *, thickness, sides="all", strength=5.0, order=3):
        """Add a perfectly matched layer inside the domain boundary.

        Parameters
        ----------
        thickness : float
            Layer thickness in metres, measured inward from each selected
            side. It is not rounded to whole cells.
        sides : str or sequence of str
            ``"all"`` (every non-periodic side), an axis such as ``"x"`` (both
            ends), or a single side such as ``"x+"``.
        strength : float
            Peak value of ``-Im(s)`` in the stretch
            ``s = 1 - i*strength*(depth/thickness)**order``. Dimensionless;
            see :func:`fdfd.suggest_pml_strength`.
        order : int
            Polynomial grading order.
        """
        if not self._supports_pml:
            raise BackendCapabilityError(f"{type(self).__name__} does not support a PML.")
        axes = tuple(a for a in self._physical_axes if a not in self._periodic_axes)
        self._pmls.append(
            pml.validate(
                thickness=thickness,
                sides=sides,
                strength=strength,
                order=order,
                axes=axes,
                ranges=self._ranges,
                all_axes=self._physical_axes,
            )
        )
        self._invalidate()

    def _apply_pml(self, kernel, grid):
        stretch, mask = pml.stretches(
            self._pmls, axes=self._physical_axes, ranges=self._ranges, resolution=grid.resolution
        )
        factors = pml.tensor_factors(stretch, grid.resolution)
        for tensor in kernel.cell_tensors():
            for values, factor in zip(tensor, factors, strict=True):
                values *= factor
        kernel.set_pml_cells(mask)
        kernel.update_component_materials()

    def _populate(self, kernel, grid):
        """Write the background, smoothed dielectrics and conductors into ``kernel``."""
        resolution = grid.resolution
        everywhere = tuple(slice(0, n) for n in resolution)
        kernel.apply_fractional_material(
            *materials.bulk_values(self.background_material), np.ones(resolution), *everywhere
        )
        records = list(self._objects.values())
        for record in records:
            if isinstance(record.material, materials.Material):
                occupancy, slices = fractions(record.shape, self._ranges, resolution, grid.subpixels)
                if occupancy.size:
                    kernel.apply_fractional_material(
                        *materials.bulk_values(record.material), occupancy, *slices
                    )
        kernel.update_component_materials()
        owner = np.zeros(resolution, dtype=bool)
        for record in records:
            material = record.material
            if isinstance(material, materials.Material):
                continue
            # Conductors select whole cells by their centres; their footprint
            # is then constrained on the actual Yee samples by the kernel.
            occupancy, slices = fractions(record.shape, self._ranges, resolution, 1)
            mask = np.zeros(resolution, dtype=bool)
            mask[slices] = occupancy.astype(bool)
            if not np.any(mask):
                raise GeometryError(f"Conductor {record.name!r} contains no grid cells; refine the mesh.")
            if np.any(mask & owner):
                raise GeometryError("Conductor objects overlap on the grid; combine them with shapes.Union.")
            owner |= mask
            if material == materials.PEC:
                kernel.add_pec(mask)
            elif material == materials.PMC:
                kernel.add_pmc(mask)
            else:
                kernel.add_impedance_surface(mask, material.at_frequency(frequency=self.frequency))

    def _conductor_mask(self):
        """Cells whose centre lies inside a conductor object."""
        mask = np.zeros(self.grid.resolution, dtype=bool)
        centres = np.meshgrid(*self.grid.coordinates, indexing="ij", sparse=True)
        for record in self._objects.values():
            if not isinstance(record.material, materials.Material):
                mask |= record.shape.contains(*centres)
        return mask

    def _material_background(self):
        """Cell materials and conductor footprints, kept with results for plotting."""
        eps, mu = self._kernel.cell_tensors()
        return {
            "epsilon": np.stack([np.array(v, copy=True) for v in eps]),
            "mu": np.stack([np.array(v, copy=True) for v in mu]),
            "conductor": self._conductor_mask(),
        }

    def _field_coordinates(self, fields, periodic_nodes=()):
        """Staggered sample coordinates of each field component.

        Along a non-periodic axis, ``N + 1`` samples are nodes and ``N`` are
        cell centres. On a periodic axis both have ``N`` samples, so the
        components listed in ``periodic_nodes`` are placed on nodes.
        """
        result = {}
        for name, values in fields.items():
            axes = []
            for index, (n, cells, (lo, hi)) in enumerate(
                zip(values.shape[:-1], self.grid.resolution, self._ranges, strict=True)
            ):
                step = (hi - lo) / cells
                if n == cells + 1:
                    axes.append(np.linspace(lo, hi, n))
                elif (
                    n == cells
                    and self._physical_axes[index] in self._periodic_axes
                    and name in periodic_nodes
                ):
                    axes.append(lo + np.arange(n) * step)
                elif n == cells:
                    axes.append(lo + (np.arange(n) + 0.5) * step)
                else:
                    raise ConfigurationError(f"Unexpected staggered shape for {name}: {values.shape}.")
            result[name] = tuple(axes)
        return result

    def show(self, *, block=True):
        """Open the interactive viewer for the latest result."""
        if self.result is None:
            raise NoResultError("Call solve() before show(); there is no current result.")
        return self.result.show(block=block)


def check_num_modes(num_modes):
    if isinstance(num_modes, (bool, np.bool_)) or int(num_modes) != num_modes or num_modes < 1:
        raise ConfigurationError("num_modes must be a positive integer.")
    return int(num_modes)


def check_guess(guess, name="guess"):
    if guess is None:
        return None
    try:
        value = complex(guess)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a finite number or None.") from exc
    if not np.isfinite(value):
        raise ConfigurationError(f"{name} must be a finite number or None.")
    return value


def check_tol(tol):
    if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or tol < 0:
        raise ConfigurationError("tol must be finite and nonnegative.")
    return float(tol)


def check_polarization(polarization, allowed):
    if polarization not in allowed:
        raise ConfigurationError(f"polarization must be one of {allowed}; received {polarization!r}.")
    return polarization

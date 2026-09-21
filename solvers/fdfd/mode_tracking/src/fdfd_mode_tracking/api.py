"""Material-first sweep API mirroring the single-frequency FDFD mode solvers."""
from dataclasses import replace
import numpy as np
from cem_common import materials, shapes
from cem_common.contracts import bounds
from cem_common.errors import ConfigurationError, NoResultError
from cem_common.grid import GridData, GridSceneMixin
from fdfd_waveguide_modes import ModeSolver1D, ModeSolver2D
from .contracts import PortSpec, TrackingConfig, VerificationSpec
from .sweep import track_modes


class _ModeTrackerAPI(GridSceneMixin):
    _supports_conductors = True
    _supports_sibc = True
    _periodic = False
    _solver_type = None

    def _init_tracker(self, *, frequencies, ranges, background_material, port):
        values = np.asarray(frequencies, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ConfigurationError('frequencies must be a nonempty sequence of positive finite hertz values.')
        if len(np.unique(values)) != len(values):
            raise ConfigurationError('frequencies must be unique.')
        if not isinstance(port, PortSpec):
            raise ConfigurationError('port must be a PortSpec.')
        self._init_scene(background_material=background_material)
        self.frequencies = tuple(float(value) for value in np.sort(values))
        self._ranges = tuple(bounds(value, axis+'_range') for value, axis in zip(ranges, self._physical_axes))
        for axis, span in zip(self._physical_axes, self._ranges):
            setattr(self, axis+'_range', span)
        self.port = port
        self.mesh_data = self._result = self._backend = None
        self._mesh_settings = None
        self._pmls = []

    def _invalidate(self):
        self.mesh_data = self._result = self._backend = None

    @property
    def result(self):
        return self._result

    def mesh(self, *, resolution=None, max_element_size=None, subpixels=None):
        """Set the base Yee grid shared by every requested sweep frequency."""
        dim = len(self._physical_axes)
        if resolution is not None and max_element_size is not None:
            raise ConfigurationError('Specify resolution or max_element_size, not both.')
        if resolution is None:
            resolution = (40,)*dim if max_element_size is None else tuple(
                max(2, int(np.ceil((hi-lo)/materials._positive(max_element_size, 'max_element_size'))))
                for lo, hi in self._ranges)
        raw = (resolution,) if dim == 1 and np.isscalar(resolution) else tuple(resolution)
        if len(raw) != dim or any(isinstance(n, (bool, np.bool_)) or int(n) != n or n < 2 for n in raw):
            raise ConfigurationError('resolution must give at least two integer cells per physical axis.')
        subpixels = (100 if dim == 1 else 8) if subpixels is None else subpixels
        if isinstance(subpixels, bool) or int(subpixels) != subpixels or subpixels < 1:
            raise ConfigurationError('subpixels must be a positive integer.')
        selected = tuple(int(n) for n in raw)
        self._mesh_settings = {'resolution': selected, 'subpixels': int(subpixels)}
        self.mesh_data = GridData(self._physical_axes, self._ranges, selected,
                                  {'subpixels': int(subpixels), 'context': self._scene_context()})
        self._result = None
        return self.mesh_data

    def _factory(self, frequency, verification):
        if self._mesh_settings is None:
            self.mesh()
        base_resolution = self._mesh_settings['resolution']
        padded_ranges, resolution = [], []
        for (lo, hi), cells in zip(self._ranges, base_resolution):
            step = (hi-lo)/cells
            padding_cells = int(round(cells*verification.padding_fraction))
            padded_ranges.append((lo-padding_cells*step, hi+padding_cells*step))
            resolution.append((cells+2*padding_cells)*verification.mesh_factor)
        arguments = {'frequency': frequency, 'background_material': self.background_material}
        arguments.update({axis+'_range': span for axis, span in zip(self._physical_axes, padded_ranges)})
        solver = self._solver_type(**arguments)
        for record, _ in self._objects.values():
            solver.add_geometry(shape=record.shape, material=record.material,
                                name=record.name, clip=record.clip)
        solver.mesh(resolution=tuple(resolution) if len(resolution) > 1 else resolution[0],
                    subpixels=self._mesh_settings['subpixels'])
        return solver

    def solve(self, *, num_modes=4, neff_guess=None, polarization='both',
              eigensolver_tolerance=1e-10,
              reference_frequency=None, tracking_config=None, scorer=None, progress=True):
        """Solve and automatically track num_modes candidates per frequency."""
        if isinstance(num_modes, bool) or int(num_modes) != num_modes or num_modes < 1:
            raise ConfigurationError('num_modes must be a positive integer.')
        if tracking_config is not None and not isinstance(tracking_config, TrackingConfig):
            raise ConfigurationError('tracking_config must be a TrackingConfig.')
        if self._mesh_settings is None:
            self.mesh()
        candidate_count = int(num_modes)
        base = tracking_config or TrackingConfig()
        config = replace(base, num_candidates=candidate_count,
                         max_candidates=candidate_count,
                         neff_guess=neff_guess, polarization=polarization,
                         eigensolver_tolerance=eigensolver_tolerance)
        self._result = track_modes(self._factory, self.frequencies, port=self.port,
                                   config=config, seed_modes=None,
                                   reference_frequency=reference_frequency, scorer=scorer,
                                   progress=progress)
        return self.result

    def show(self, *, component='E', quantity='magnitude', block=True):
        if self.result is None:
            raise NoResultError('Call solve() before show(); there is no tracked sweep.')
        return self.result.show(component=component, quantity=quantity, block=block)


class ModeTracker1D(_ModeTrackerAPI):
    """Frequency-swept material-first 1D FDFD mode tracker."""
    _physical_axes = ('x',)
    _solver_type = ModeSolver1D

    def __init__(self, *, frequencies, x_range, background_material=materials.vacuum,
                 port=PortSpec()):
        self._init_tracker(frequencies=frequencies, ranges=(x_range,),
                           background_material=background_material, port=port)

    def add_layer(self, *, x_range, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Interval(bounds=x_range), material=material,
                                 name=name, clip=clip)


class ModeTracker2D(_ModeTrackerAPI):
    """Frequency-swept material-first full-vector 2D FDFD mode tracker."""
    _physical_axes = ('x', 'y')
    _solver_type = ModeSolver2D

    def __init__(self, *, frequencies, x_range, y_range,
                 background_material=materials.vacuum, port=PortSpec()):
        self._init_tracker(frequencies=frequencies, ranges=(x_range, y_range),
                           background_material=background_material, port=port)

    def add_rectangle(self, *, x_range, y_range, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Rectangle(bounds=(x_range, y_range)),
                                 material=material, name=name, clip=clip)

    def add_circle(self, *, center, radius, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Circle(center=center, radius=radius),
                                 material=material, name=name, clip=clip)

    def add_polygon(self, *, points, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Polygon(points=points), material=material,
                                 name=name, clip=clip)

"""Public material-first FDFD waveguide APIs; numerical kernels stay private."""
import numpy as np
from cem_common import materials, shapes
from cem_common.grid import GridSceneMixin, GridResult, load_grid_result
from cem_common._yee_scene import populate, apply_pml, field_coordinates, validate_solve
from cem_common.errors import ConfigurationError


class ModeSet(GridResult):
    """Returned staggered waveguide fields and complex effective indices."""


def load_result(path):
    return load_grid_result(path, family='fdfd_waveguide_modes', result_type=ModeSet)


class _WaveguideAPI(GridSceneMixin):
    _supports_conductors = True
    _supports_sibc = True
    _periodic = False
    def _populate_backend(self, backend, resolution, subpixels):
        populate(self, backend, resolution, subpixels)
    def _apply_pml(self, backend, resolution, spec):
        apply_pml(self, backend, resolution, spec)
    def add_pml(self, *, thickness, direction='all', order=3, sigma_max=5.):
        self._record_pml(thickness=thickness, direction=direction, order=order, sigma_max=sigma_max)
    def _result_from_fields(self, fields, neff, polarizations):
        metadata = {'k0': self._backend.k_0, 'field_representation': 'staggered-fields; exp(-i*beta*z)',
                    'field_normalization': 'native eigenvector normalization; H_num=-i*eta0*H',
                    'polarizations': tuple(polarizations), 'context': self._scene_context(),
                    'solve_info': {'eigensolver_tolerance': self._backend._eigensolver_tolerance}}
        self._result = ModeSet('fdfd_waveguide_modes', self.mesh_data, self.frequency,
                               fields, field_coordinates(self, fields), np.array(neff), metadata)
        return self.result


class ModeSolver1D(_WaveguideAPI):
    _physical_axes = ('x',)
    def __init__(self, *, frequency, x_range, background_material=materials.vacuum):
        self._init_grid(ranges=(x_range,), background_material=background_material, frequency=frequency)
    def _make_backend(self, resolution):
        from .solver_1d import _ModeSolver1D
        return _ModeSolver1D(self.frequency, self.x_range[1]-self.x_range[0], resolution[0], 1)
    def add_layer(self, *, x_range, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Interval(bounds=x_range), material=material, name=name, clip=clip)
    def mesh(self, *, resolution=None, max_element_size=None, subpixels=100):
        return self._mesh_grid(resolution=resolution, max_element_size=max_element_size, subpixels=subpixels)
    def solve(self, *, num_modes=4, neff_guess=None, polarization='both', eigensolver_tolerance=0.):
        """Return num_modes total modes; select TE, TM, or both polarizations."""
        validate_solve(num_modes, neff_guess, eigensolver_tolerance)
        if polarization not in ('TE', 'TM', 'both'):
            raise ConfigurationError('polarization must be TE, TM, or both.')
        if self.mesh_data is None:
            self.mesh(**(self._mesh_settings or {}))
        backend = self._backend
        backend.num_modes = int(num_modes)
        backend._eigensolver_tolerance = eigensolver_tolerance
        self._result = None
        backend.solve(sigma=None if neff_guess is None else -complex(neff_guess)**2)
        candidates = [(pol, i, getattr(backend, 'neff_'+pol)[i]) for pol in ('TE', 'TM')
                      if polarization in ('both', pol) for i in range(num_modes)]
        candidates.sort(key=lambda item: abs(item[2]-(neff_guess if neff_guess is not None else max(abs(v[2]) for v in candidates))))
        selected = candidates[:num_modes]
        fields = {}
        for component in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
            raw = np.asarray(getattr(backend, component))
            pol = 'TE' if component in ('Ey', 'Hx', 'Hz') else 'TM'
            fields[component] = np.column_stack([raw[:, i] if p == pol else np.zeros(raw.shape[0], complex) for p, i, _ in selected])
        return self._result_from_fields(fields, [v for _, _, v in selected], [p for p, _, _ in selected])


class ModeSolver2D(_WaveguideAPI):
    _physical_axes = ('x', 'y')
    def __init__(self, *, frequency, x_range, y_range, background_material=materials.vacuum):
        self._init_grid(ranges=(x_range, y_range), background_material=background_material, frequency=frequency)
    def _make_backend(self, resolution):
        from .solver_2d import _ModeSolver2D
        return _ModeSolver2D(self.frequency, self.x_range[1]-self.x_range[0], self.y_range[1]-self.y_range[0], *resolution, 1)
    def add_rectangle(self, *, x_range, y_range, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Rectangle(bounds=(x_range, y_range)), material=material, name=name, clip=clip)
    def add_circle(self, *, center, radius, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Circle(center=center, radius=radius), material=material, name=name, clip=clip)
    def add_polygon(self, *, points, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Polygon(points=points), material=material, name=name, clip=clip)
    def mesh(self, *, resolution=None, max_element_size=None, subpixels=8):
        return self._mesh_grid(resolution=resolution, max_element_size=max_element_size, subpixels=subpixels)
    def solve(self, *, num_modes=4, neff_guess=None, eigensolver_tolerance=0.):
        validate_solve(num_modes, neff_guess, eigensolver_tolerance)
        self._ensure_grid()
        self._backend.num_modes = int(num_modes)
        self._backend._eigensolver_tolerance = eigensolver_tolerance
        self._result = None
        self._backend.solve(sigma=None if neff_guess is None else -complex(neff_guess)**2)
        fields = {name: np.array(getattr(self._backend, name), copy=True) for name in ('Ex','Ey','Ez','Hx','Hy','Hz')}
        return self._result_from_fields(fields, self._backend.neff, ['vector']*num_modes)

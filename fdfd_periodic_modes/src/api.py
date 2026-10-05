"""Public fixed-frequency periodic grid solvers."""
import numpy as np
from cem_common import materials, shapes
from cem_common.grid import GridSceneMixin, GridResult, load_grid_result
from cem_common._yee_scene import populate, apply_pml, field_coordinates, validate_solve
from cem_common.errors import ConfigurationError


class PeriodicModeSet(GridResult):
    """Returned Bloch modes and staggered periodic-envelope fields."""


def load_result(path):
    return load_grid_result(path, family='fdfd_periodic_modes', result_type=PeriodicModeSet)


class _PeriodicAPI(GridSceneMixin):
    _supports_conductors = True
    _supports_sibc = False
    _periodic = True
    def _populate_backend(self, backend, resolution, subpixels):
        populate(self, backend, resolution, subpixels)
    def _apply_pml(self, backend, resolution, spec):
        apply_pml(self, backend, resolution, spec)
    def add_pml(self, *, thickness, direction='all', order=3, sigma_max=5.):
        self._record_pml(thickness=thickness, direction=direction, order=order, sigma_max=sigma_max)
    def mesh(self, *, resolution=None, max_element_size=None, subpixels=8):
        return self._mesh_grid(resolution=resolution, max_element_size=max_element_size, subpixels=subpixels)
    def _finish(self, fields, tolerance):
        backend = self._backend
        self._result = PeriodicModeSet('fdfd_periodic_modes', self.mesh_data, self.frequency,
            fields, field_coordinates(self, fields), np.array(backend.neff),
            {'k0': backend.k0, 'field_representation': 'periodic-envelope; staggered-fields',
             'field_normalization': 'native eigenvector normalization', 'context': self._scene_context(),
             'solve_info': {'eigensolver_tolerance': tolerance, 'residuals': getattr(backend, 'refined_residuals', None)}})
        return self.result


class PeriodicModeSolver2D(_PeriodicAPI):
    _physical_axes = ('x', 'z')
    def __init__(self, *, frequency, x_range, z_range, polarization='TE', background_material=materials.vacuum):
        if polarization not in ('TE', 'TM'):
            raise ConfigurationError('polarization must be TE or TM.')
        self.polarization = polarization
        self._init_grid(ranges=(x_range, z_range), background_material=background_material, frequency=frequency)
    def _make_backend(self, resolution):
        from .solver_2d import _PeriodicModeSolver2D
        return _PeriodicModeSolver2D(self.polarization, self.frequency,
            self.x_range[1]-self.x_range[0], self.z_range[1]-self.z_range[0], *resolution, 1)
    def add_rectangle(self, *, x_range, z_range, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Rectangle(bounds=(x_range, z_range)), material=material, name=name, clip=clip)
    def add_circle(self, *, center, radius, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Circle(center=center, radius=radius), material=material, name=name, clip=clip)
    def add_polygon(self, *, points, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Polygon(points=points), material=material, name=name, clip=clip)
    def solve(self, *, num_modes=4, neff_guess=1., eigensolver_tolerance=0., eigensolver='eigs',
              ncv=None, max_restarts=12, random_seed=0, arnoldi_backend='auto'):
        validate_solve(num_modes, neff_guess, eigensolver_tolerance)
        self._ensure_grid()
        backend = self._backend
        backend.num_modes = int(num_modes)
        self._result = None
        backend.solve(guess=1j*backend.k0*neff_guess, tol=eigensolver_tolerance, ncv=ncv,
                      method=eigensolver, max_restarts=max_restarts, random_seed=random_seed, kernel_backend=arnoldi_backend)
        names = ('Ex', 'Hy') if self.polarization == 'TM' else ('Ey', 'Hx')
        fields = {name: np.asarray(getattr(backend, name)).reshape((*getattr(backend, 'shape_'+name.lower()), num_modes), order='F').copy() for name in names}
        return self._finish(fields, eigensolver_tolerance)


class PeriodicModeSolver3D(_PeriodicAPI):
    _physical_axes = ('x', 'y', 'z')
    def __init__(self, *, frequency, x_range, y_range, z_range, background_material=materials.vacuum):
        self._init_grid(ranges=(x_range, y_range, z_range), background_material=background_material, frequency=frequency)
    def _make_backend(self, resolution):
        from .solver_3d import _PeriodicModeSolver3D
        return _PeriodicModeSolver3D(*resolution, *(hi-lo for lo, hi in self._ranges), self.frequency, 1)
    def add_box(self, *, x_range, y_range, z_range, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Box(bounds=(x_range, y_range, z_range)), material=material, name=name, clip=clip)
    def add_sphere(self, *, center, radius, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Sphere(center=center, radius=radius), material=material, name=name, clip=clip)
    def add_cylinder(self, *, center, radius, z_range, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Cylinder(center=center, radius=radius, z_range=z_range), material=material, name=name, clip=clip)
    def solve(self, *, num_modes=4, neff_guess=1., eigensolver_tolerance=0., eigensolver='refined',
              ncv=None, max_restarts=12, random_seed=0, arnoldi_backend='auto'):
        validate_solve(num_modes, neff_guess, eigensolver_tolerance)
        self._ensure_grid()
        backend = self._backend
        backend.num_modes = int(num_modes)
        self._result = None
        backend.solve(sigma_guess=1j*backend.k0*neff_guess, tol=eigensolver_tolerance, ncv=ncv,
                      method=eigensolver, max_restarts=max_restarts, random_seed=random_seed, kernel_backend=arnoldi_backend)
        fields = {name: np.moveaxis(np.array(value, copy=True), 0, -1) for name, value in backend.fields.items()}
        return self._finish(fields, eigensolver_tolerance)

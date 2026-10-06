"""Public scalar TE/TM scattering workflow with physical material geometry."""
import numpy as np
from fdfd_common import materials, shapes
from fdfd_common.grid import GridSceneMixin, GridResult, fractions, load_grid_result
from fdfd_common._yee_scene import material_background, field_coordinates
from fdfd_common.errors import ConfigurationError, BackendCapabilityError


class ScatteringResult(GridResult):
    """Returned three-component total-field/scattered-field grid solution."""
    def show(self, *, block=True):
        from .visualization import show_result
        return show_result(self, block=block)



def load_result(path):
    return load_grid_result(path, family='fdfd_scattering', result_type=ScatteringResult)


class ScatteringSolver2D(GridSceneMixin):
    _physical_axes = ('x', 'y')
    _supports_conductors = True
    _supports_sibc = False
    _periodic = False
    def __init__(self, *, frequency, x_range, y_range, polarization='TE', background_material=materials.vacuum):
        if polarization not in ('TE', 'TM'):
            raise ConfigurationError('polarization must be TE or TM.')
        self.polarization = polarization
        self._source_settings = None
        self._mask_distance = None
        self._init_grid(ranges=(x_range, y_range), background_material=background_material, frequency=frequency)
    def _make_backend(self, resolution):
        from .solver_2d import _ScatteringSolver2D
        return _ScatteringSolver2D(self.frequency, *(hi-lo for lo, hi in self._ranges), *resolution)
    def _validate_record(self, record):
        super()._validate_record(record)
        if record.material == materials.PMC:
            raise BackendCapabilityError('Scattering implements PEC objects; PMC objects are unsupported.')
    def _populate_backend(self, backend, resolution, subpixels):
        eps, mu = materials.bulk_values(self.background_material)
        backend.add_object(eps, mu, np.ones((resolution[1], resolution[0]), dtype=bool))
        for record, _ in self._objects.values():
            # Geometry occupies cells; constitutive values are then sampled
            # separately on each electric and magnetic Yee lattice.
            occupancy, slices = fractions(record.shape, self._ranges, resolution, 1)
            mask = np.zeros(resolution, dtype=bool)
            mask[slices] = occupancy.astype(bool)
            if record.material == materials.PEC:
                if not mask.any():
                    from fdfd_common.errors import GeometryError
                    raise GeometryError('PEC geometry contains no grid cells; refine the mesh.')
                backend.add_pec(mask.T)
            else:
                backend.add_object(*materials.bulk_values(record.material), mask.T)
        for component in ('xx', 'yy', 'zz'):
            setattr(backend, 'cell_eps_r_' + component, getattr(backend, 'ER' + component).T.copy())
            setattr(backend, 'cell_mu_r_' + component, getattr(backend, 'MR' + component).T.copy())
    def _apply_pml(self, backend, resolution, spec):
        if spec['direction'] not in ('x', 'y', 'all'):
            raise BackendCapabilityError('This scattering PML supports paired x/y ends only.')
        axes = (0, 1) if spec['direction']=='all' else (self._physical_axes.index(spec['direction']),)
        for axis in axes:
            lo, hi = self._ranges[axis]
            width = int(np.ceil(spec['thickness']*resolution[axis]/(hi-lo)-1e-12))
            backend.add_UPML(pml_width=width, n=spec['order'], sigma_max=spec['sigma_max'], direction=self._physical_axes[axis])
    def add_pml(self, *, thickness, direction='all', order=3, sigma_max=5.):
        self._record_pml(thickness=thickness, direction=direction, order=order, sigma_max=sigma_max)
    def add_rectangle(self, *, x_range, y_range, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Rectangle(bounds=(x_range, y_range)), material=material, name=name, clip=clip)
    def add_circle(self, *, center, radius, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Circle(center=center, radius=radius), material=material, name=name, clip=clip)
    def add_polygon(self, *, points, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Polygon(points=points), material=material, name=name, clip=clip)
    def mesh(self, *, resolution=None, max_element_size=None):
        return self._mesh_grid(resolution=resolution, max_element_size=max_element_size, subpixels=1)
    def add_source(self, *, kind='plane_wave', angle=0., location=None, amplitude=1.):
        """Set the incident field; angles are degrees from physical +x."""
        if kind not in ('plane_wave', 'point') or not np.isfinite(angle) or not np.isfinite(amplitude):
            raise ConfigurationError('Invalid source kind, angle, or amplitude.')
        if kind == 'point' and (location is None or len(location)!=2 or not np.isfinite(location).all()):
            raise ConfigurationError('A point source needs a finite physical (x,y) location.')
        self._source_settings = dict(src_type=kind, angle_deg=angle, location=location, amplitude=amplitude)
        self._result = None
    def set_source_region(self, *, inset):
        """Set the rectangular total-field region's physical inset in metres."""
        self._mask_distance = materials._positive(inset, 'inset')
        self._result = None
    def solve(self, *, reuse_factorization=True):
        if self._source_settings is None or self._mask_distance is None:
            raise ConfigurationError('Define add_source() and set_source_region(inset=...) before solving.')
        self._ensure_grid()
        backend = self._backend
        config = dict(self._source_settings)
        centre = np.array([.5*(lo+hi) for lo, hi in self._ranges])
        if config['location'] is not None:
            config['location'] = tuple(np.asarray(config['location'])-centre)
        else:
            theta = np.deg2rad(config['angle_deg'])
            config['amplitude'] *= np.exp(-1j*backend.k0*np.dot(centre, (np.cos(theta), np.sin(theta))))
        backend.add_source(polarization=self.polarization, **config)
        backend.set_total_field_region(self._mask_distance)
        if not np.any(backend.Q.diagonal() == 0):
            raise ConfigurationError('Source-region inset leaves no total-field cells.')
        if np.any(backend.pec_field_masks()[backend.primary].ravel() & (backend.Q.diagonal() != 0)):
            raise ConfigurationError('The total-field source region must enclose the PEC scatterer.')
        self._result=None
        operation = backend.solve_total_field_TE if self.polarization=='TE' else backend.solve_total_field_TM
        operation(reuse_factorisation=reuse_factorization)
        fields = {name: np.array(field.T[..., None], copy=True)
                  for name, field in backend.transverse_fields(self.polarization).items()}
        coordinates = field_coordinates(self, fields)
        self._result=ScatteringResult('fdfd_scattering', self.mesh_data, self.frequency, fields,
            coordinates, np.array([],dtype=complex),
            {'k0':backend.k0,'field_representation':'TF/SF fields; staggered-grid',
             'material_background': material_background(self),
             'context':self._scene_context(),'solve_info':{'polarization':self.polarization}})
        return self.result

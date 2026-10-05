"""Physical grid scenes and data-only results; numerical operators remain local."""
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from types import MappingProxyType
import numpy as np
from . import materials, shapes
from .contracts import bounds
from .errors import ConfigurationError, GeometryError, BackendCapabilityError, NoResultError, PersistenceError
from .scene import SceneMixin


@dataclass(frozen=True)
class GridData:
    axes: tuple
    bounds: tuple
    resolution: tuple
    metadata: dict = field(default_factory=dict)
    @property
    def coordinates(self):
        """Cell-centre axes in metres; fields carry their own staggered axes."""
        return tuple(np.linspace(lo, hi, n, endpoint=False)+(hi-lo)/(2*n)
                     for (lo, hi), n in zip(self.bounds, self.resolution))


class GridSceneMixin(SceneMixin):
    def _init_grid(self, *, ranges, background_material, frequency=None):
        self._init_scene(background_material=background_material)
        self._ranges = tuple(bounds(r, axis+'_range') for r, axis in zip(ranges, self._physical_axes))
        for axis, span in zip(self._physical_axes, self._ranges):
            setattr(self, axis+'_range', span)
        if frequency is not None:
            self.frequency = materials._positive(frequency, 'frequency')
            self.omega = 2*np.pi*self.frequency
            self.wavelength = 299792458./self.frequency
        self.mesh_data = self._result = self._backend = None
        self._mesh_settings = None
        self._pmls = []

    @property
    def result(self): return self._result

    def _invalidate(self):
        self.mesh_data = self._result = self._backend = None

    def _validate_record(self, record):
        if not isinstance(record.shape, shapes.Shape) or record.shape.dimension != len(self._physical_axes):
            raise GeometryError('Shape must match the solver physical axes.')
        if isinstance(record.shape, shapes.Segment):
            raise BackendCapabilityError('Grid solvers require finite-volume geometry, not zero-thickness segments.')
        if not record.clip and any(a < lo or b > hi for a, b, (lo, hi) in zip(record.shape.bounds[::2], record.shape.bounds[1::2], self._ranges)):
            raise GeometryError('Shape lies outside the domain; use clip=True explicitly.')
        if isinstance(record.material, materials.Material):
            materials.bulk_values(record.material, form=getattr(self, '_material_form', 'diagonal'))
        elif isinstance(record.material, materials.IdealBoundary):
            if not self._supports_conductors:
                raise BackendCapabilityError('This solver does not implement PEC/PMC object constraints.')
        elif isinstance(record.material, (materials.GoodConductor, materials.SurfaceImpedance)):
            if not self._supports_sibc:
                raise BackendCapabilityError('This solver does not implement SIBC materials.')
        else:
            raise ConfigurationError('Assign a predefined supported material object.')

    def _insert_geometry(self, record):
        self._validate_record(record)
        self._invalidate()
        return record
    def _delete_geometry(self, handle): self._invalidate()
    def _replace_geometry(self, handle, record):
        self._validate_record(record)
        self._invalidate()
        return record

    def _mesh_grid(self, *, resolution=None, max_element_size=None, subpixels=8):
        dim = len(self._physical_axes)
        if resolution is not None and max_element_size is not None:
            raise ConfigurationError('Specify resolution or max_element_size, not both.')
        if resolution is None:
            resolution = (40,)*dim if max_element_size is None else tuple(max(2, int(np.ceil((b-a)/materials._positive(max_element_size, 'max_element_size')))) for a, b in self._ranges)
        raw = (resolution,) if dim == 1 and np.isscalar(resolution) else tuple(resolution)
        if len(raw) != dim or any(isinstance(n, (bool, np.bool_)) or int(n) != n or n < 2 for n in raw):
            raise ConfigurationError('resolution must give at least two integer cells per physical axis.')
        if isinstance(subpixels, bool) or int(subpixels) != subpixels or subpixels < 1:
            raise ConfigurationError('subpixels must be a positive integer.')
        selected = tuple(int(n) for n in raw)
        backend = self._make_backend(selected)
        self._backend = backend
        try:
            self._populate_backend(backend, selected, int(subpixels))
            for pml in self._pmls:
                self._apply_pml(backend, selected, pml)
        except Exception:
            self._invalidate()
            raise
        self.mesh_data = GridData(self._physical_axes, self._ranges, selected,
                                  {'subpixels': int(subpixels), 'context': self._scene_context()})
        self._mesh_settings = dict(resolution=selected, subpixels=int(subpixels))
        self._result = None
        return self.mesh_data

    def _ensure_grid(self):
        if self.mesh_data is None:
            self._mesh_grid(**(self._mesh_settings or {}))

    def show(self, *, block=True):
        if self.result is None:
            raise NoResultError('Call solve() before show(); there is no current result.')
        return self.result.show(block=block)

    def _record_pml(self, *, thickness, direction, order, sigma_max):
        materials._positive(thickness, 'thickness')
        if isinstance(order, bool) or int(order) != order or order < 1 or not np.isfinite(sigma_max) or sigma_max < 0:
            raise ConfigurationError('PML order must be a positive integer and sigma_max finite/nonnegative.')
        supported = tuple(a for a in self._physical_axes if a != 'z' or not self._periodic)
        directions = supported+tuple(a+side for a in supported for side in ('-', '+'))+('all',)
        if direction not in directions:
            raise BackendCapabilityError(f'PML direction must be one of {directions}.')
        selected = supported if direction == 'all' else (direction[0],)
        if any(2*thickness >= self._ranges[self._physical_axes.index(a)][1]-self._ranges[self._physical_axes.index(a)][0] for a in selected):
            raise ConfigurationError('PML leaves no physical interior.')
        self._pmls.append(dict(thickness=float(thickness), direction=direction, order=int(order), sigma_max=float(sigma_max)))
        self._invalidate()


def fractions(shape, spans, resolution, subpixels):
    """Midpoint subpixel occupancy, with no integer/float unit ambiguity."""
    b = shape.bounds
    starts = [max(0, int(np.floor((lo-span[0])*n/(span[1]-span[0])))) for lo, span, n in zip(b[::2], spans, resolution)]
    stops = [min(n, int(np.ceil((hi-span[0])*n/(span[1]-span[0])))) for hi, span, n in zip(b[1::2], spans, resolution)]
    slices = tuple(slice(a, max(a, b)) for a, b in zip(starts, stops))
    result = np.zeros(tuple(max(0, b-a) for a, b in zip(starts, stops)))
    if not result.size:
        return result, slices
    base = [span[0]+np.arange(a, b)*(span[1]-span[0])/n for a, b, span, n in zip(starts, stops, spans, resolution)]
    steps = [(hi-lo)/n for (lo, hi), n in zip(spans, resolution)]
    for offset in product(range(subpixels), repeat=len(spans)):
        points = np.meshgrid(*(axis+(i+.5)*step/subpixels for axis, i, step in zip(base, offset, steps)), indexing='ij', sparse=True)
        result += shape.contains(*points)
    return result/subpixels**len(spans), slices


@dataclass(frozen=True)
class GridResult:
    """Completed grid fields, with explicit coordinates for each component."""
    family: str
    mesh_data: GridData
    frequency: float | None
    fields: dict
    field_coordinates: dict
    neff: np.ndarray
    metadata: dict

    @property
    def solve_info(self): return self.metadata.get('solve_info', {})
    @property
    def beta(self): return self.neff * self.metadata['k0']
    @property
    def attenuation_constant(self): return -self.beta.imag
    def __len__(self): return len(self.neff)

    def plot(self, *, component=None, quantity='real', mode=0, plane=None, position=None):
        from matplotlib.figure import Figure
        fig = Figure(figsize=(7, 5))
        self._draw(fig.subplots(), component, quantity, mode, plane, position)
        fig.tight_layout()
        return fig

    def _draw(self, ax, component, quantity, mode, plane, position):
        if isinstance(mode, bool) or int(mode) != mode or mode < 0:
            raise ConfigurationError('mode must be a zero-based integer.')
        name = next(iter(self.fields)) if component is None else component
        if name not in self.fields:
            raise ConfigurationError(f'Available components: {tuple(self.fields)}.')
        raw = np.asarray(self.fields[name])
        if mode >= raw.shape[-1]:
            raise ConfigurationError('Mode index is out of range.')
        raw = raw[..., mode]
        coordinates = self.field_coordinates[name]
        axes = self.mesh_data.axes
        if raw.ndim == 3:
            plane = 'xy' if plane is None else plane
            if plane not in ('xy', 'xz', 'yz'):
                raise ConfigurationError('plane must be xy, xz, or yz.')
            cut = next(i for i, a in enumerate(axes) if a not in plane)
            position = np.mean(coordinates[cut]) if position is None else float(position)
            raw = np.take(raw, int(np.argmin(abs(coordinates[cut]-position))), axis=cut)
            coordinates = tuple(c for i, c in enumerate(coordinates) if i != cut)
            axes = tuple(a for i, a in enumerate(axes) if i != cut)
        operation = {'real': np.real, 'imag': np.imag, 'magnitude': np.abs, 'phase': np.angle}.get(quantity)
        if operation is None:
            raise ConfigurationError('quantity must be real, imag, magnitude, or phase.')
        values = operation(raw)
        if values.ndim == 1:
            ax.plot(coordinates[0], values)
            ax.set(xlabel=axes[0]+' (m)', ylabel=f'{name} ({quantity})')
        else:
            ax.pcolormesh(*coordinates, values.T, shading='auto', cmap='viridis')
            ax.set(xlabel=axes[0]+' (m)', ylabel=axes[1]+' (m)', aspect='equal')
        ax.set_title(f'{self.family}: {name}, mode {mode}')

    def show(self, *, block=True):
        from matplotlib import pyplot as plt
        from matplotlib.widgets import RadioButtons, Slider
        if not isinstance(block, bool):
            raise ConfigurationError('block must be a boolean.')
        figure, ax = plt.subplots(figsize=(9, 5))
        figure.subplots_adjust(left=.28, bottom=.2)
        selector = RadioButtons(figure.add_axes((.02, .35, .18, .5)), tuple(self.fields))
        count = max(np.asarray(v).shape[-1] for v in self.fields.values())
        slider = Slider(figure.add_axes((.32, .05, .5, .04)), 'mode', 0, max(1, count-1), valinit=0, valstep=1)
        def draw(_=None):
            ax.clear()
            self._draw(ax, selector.value_selected, 'magnitude', min(int(slider.val), count-1), None, None)
            figure.canvas.draw_idle()
        selector.on_clicked(draw)
        slider.on_changed(draw)
        figure._cem_controls = (selector, slider, draw)
        draw()
        plt.show(block=block)
        return figure

    def save(self, path):
        from .persistence import atomic_h5, write_value
        with atomic_h5(path) as handle:
            handle.attrs.update(format='cem-fdfd-results', schema='1.0', solver_family=self.family,
                                time_convention='exp(+i*omega*t)', units='SI', dimension=len(self.mesh_data.axes),
                                result_kind='fields', field_representation=self.metadata['field_representation'])
            write_value(handle, 'result', self)
        return Path(path)


def load_grid_result(path, *, family, result_type=GridResult):
    import h5py
    from .persistence import read_value
    try:
        with h5py.File(path, 'r') as handle:
            for key, expected in dict(format='cem-fdfd-results', schema='1.0', solver_family=family,
                                      time_convention='exp(+i*omega*t)', units='SI', result_kind='fields').items():
                if handle.attrs.get(key) != expected:
                    raise PersistenceError(f'Incompatible grid archive: {key} must be {expected!r}.')
            result = read_value(handle['result'], {result_type.__name__: result_type, 'GridData': GridData})
            if result.family != family or handle.attrs['dimension'] != len(result.mesh_data.axes) or handle.attrs['field_representation'] != result.metadata['field_representation']:
                raise PersistenceError('Grid archive metadata does not match its fields.')
            return result
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise PersistenceError(f'Cannot load grid result: {exc}') from exc

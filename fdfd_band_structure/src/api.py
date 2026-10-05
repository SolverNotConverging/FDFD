"""Material-first band-structure workflow; frequency is an eigenvalue."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from cem_common import materials, shapes
from cem_common.grid import GridSceneMixin, GridData
from cem_common.errors import ConfigurationError, PersistenceError


@dataclass(frozen=True)
class BandStructureResult:
    """Returned complex frequencies in Hz along a physical Bloch-vector path."""
    mesh_data: GridData
    beta_path: np.ndarray
    frequencies: dict
    eigenvalues: dict
    metadata: dict
    @property
    def solve_info(self): return self.metadata['solve_info']
    @property
    def normalized_frequencies(self):
        return {key: value*self.metadata['x_period']/299792458. for key, value in self.frequencies.items()}
    def _draw(self, ax, component, quantity, mode):
        operation={'real':np.real,'imag':np.imag,'magnitude':np.abs}.get(quantity)
        if operation is None:
            raise ConfigurationError('Band quantity must be real, imag, or magnitude.')
        selected=tuple(self.frequencies) if component is None else (component,)
        distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.beta_path,axis=1),axis=0))]
        for pol in selected:
            if pol not in self.frequencies:
                raise ConfigurationError(f'Available polarizations: {tuple(self.frequencies)}.')
            values=self.frequencies[pol]
            modes=range(len(values)) if mode is None else (mode,)
            for i in modes:
                if isinstance(i,bool) or int(i)!=i or not 0<=i<len(values):
                    raise ConfigurationError('mode must be a zero-based band index.')
                ax.plot(distance,operation(values[i]),label=f'{pol} {i}')
        ax.set(xlabel='Distance along Bloch path (rad/m)',ylabel=f'Frequency ({quantity}, Hz)')
        ax.legend()
        ax.grid(alpha=.25)
    def plot(self, *, component=None, quantity='real', mode=None):
        from matplotlib.figure import Figure
        fig=Figure(figsize=(7,5))
        self._draw(fig.subplots(),component,quantity,mode)
        fig.tight_layout()
        return fig
    def show(self, *, block=True):
        from matplotlib import pyplot as plt
        fig,ax=plt.subplots(figsize=(7,5))
        self._draw(ax,None,'real',None)
        fig.tight_layout()
        plt.show(block=block)
        return fig
    def save(self,path):
        from cem_common.persistence import atomic_h5,write_value
        with atomic_h5(path) as handle:
            handle.attrs.update(format='cem-fdfd-results',schema='1.0',solver_family='fdfd_band_structure',
                units='SI',time_convention='exp(+i*omega*t)',dimension=2,result_kind='bands',field_representation='Bloch eigenfrequencies')
            write_value(handle,'result',self)
        return Path(path)


def load_result(path):
    import h5py
    from cem_common.persistence import read_value
    try:
        with h5py.File(path,'r') as handle:
            for key,value in dict(format='cem-fdfd-results',schema='1.0',solver_family='fdfd_band_structure',
                units='SI',time_convention='exp(+i*omega*t)',dimension=2,result_kind='bands',field_representation='Bloch eigenfrequencies').items():
                if handle.attrs.get(key)!=value:
                    raise PersistenceError(f'Incompatible band archive: {key}.')
            return read_value(handle['result'],{'BandStructureResult':BandStructureResult,'GridData':GridData})
    except (OSError,KeyError,ValueError,TypeError) as exc:
        raise PersistenceError(f'Cannot load band result: {exc}') from exc


class BandStructureSolver2D(GridSceneMixin):
    _physical_axes=('x','y')
    _supports_conductors=False
    _supports_sibc=False
    _periodic=True
    _material_form='scalar'
    def __init__(self, *, x_range, y_range, background_material=materials.vacuum):
        self._init_grid(ranges=(x_range,y_range),background_material=background_material)
    def _make_backend(self,resolution):
        from .solver_2d import _BandStructureSolver2D
        eps,mu=materials.bulk_values(self.background_material,form='scalar')
        return _BandStructureSolver2D(self.x_range[1]-self.x_range[0],*resolution,
            b=self.y_range[1]-self.y_range[0],background_er=eps,background_ur=mu)
    def _populate_backend(self,backend,resolution,subpixels):
        center=[(lo+hi)/2 for lo,hi in self._ranges]
        x,y=backend.X2+center[0],backend.Y2+center[1]
        for record,_ in self._objects.values():
            eps,mu=materials.bulk_values(record.material,form='scalar')
            backend.add_object(record.shape.contains(x,y),er=eps,ur=mu)
    def mesh(self, *, resolution=None, max_element_size=None):
        return self._mesh_grid(resolution=resolution,max_element_size=max_element_size,subpixels=1)
    def add_rectangle(self, *, x_range,y_range,material,name=None,clip=False):
        return self.add_geometry(shape=shapes.Rectangle(bounds=(x_range,y_range)),material=material,name=name,clip=clip)
    def add_circle(self, *, center,radius,material,name=None,clip=False):
        return self.add_geometry(shape=shapes.Circle(center=center,radius=radius),material=material,name=name,clip=clip)
    def add_polygon(self, *, points,material,name=None,clip=False):
        return self.add_geometry(shape=shapes.Polygon(points=points),material=material,name=name,clip=clip)
    def make_bloch_path(self, *, points, num_points=40):
        """Sample a polyline of (kx,ky) points in rad/m; return a 2-by-N array."""
        pts=np.asarray(points,dtype=float)
        if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<2 or not np.isfinite(pts).all():
            raise ConfigurationError('points must contain at least two finite Bloch vectors.')
        if isinstance(num_points,bool) or int(num_points)!=num_points or num_points<len(pts):
            raise ConfigurationError('num_points must be an integer at least the number of vertices.')
        lengths=np.linalg.norm(np.diff(pts,axis=0),axis=1)
        if np.any(lengths==0):
            raise ConfigurationError('Consecutive Bloch vertices must differ.')
        counts=np.ones(len(lengths),dtype=int)
        for _ in range(num_points-1-len(lengths)):
            counts[np.argmax(lengths/counts)]+=1
        return np.vstack([*(np.linspace(a,b,n,endpoint=False) for a,b,n in zip(pts[:-1],pts[1:],counts)),pts[-1:]]).T
    def solve(self, *, beta_path, num_modes=4, polarizations=('TE','TM'), eigenvalue_guess=0., eigensolver_tolerance=0.):
        """Solve complex eigenfrequencies; dispersive/SIBC materials are unsupported."""
        from cem_common._yee_scene import validate_solve
        validate_solve(num_modes,eigenvalue_guess,eigensolver_tolerance)
        self._ensure_grid()
        self._backend._eigensolver_tolerance=eigensolver_tolerance
        self._result=None
        native=self._backend.compute_band_structure(np.asarray(beta_path),num_bands=num_modes,
            polarisations=polarizations,eig_sigma=eigenvalue_guess)
        period=self.x_range[1]-self.x_range[0]
        self._result=BandStructureResult(self.mesh_data,np.array(native.beta_path),
            {key:value*299792458./period for key,value in native.frequencies.items()},native.eigenvalues,
            {'x_period':period,'context':self._scene_context(),
             'solve_info':{'eigensolver_tolerance':eigensolver_tolerance,'eigenvalue_guess':eigenvalue_guess}})
        return self.result

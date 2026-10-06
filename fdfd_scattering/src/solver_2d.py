"""Rectangular Yee-grid scattering with the exp(+j*omega*t) convention."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spla
from scipy.special import hankel2
from scipy.constants import epsilon_0, speed_of_light
from fdfd_common.yee import node_average, node_to_cell_difference, occupied_nodes


class _ScatteringSolver2D:
    """TEz/TMz scattering with complete node and cell field lattices.

    Arrays use (y, x) order internally. Ez is at nodes in both directions,
    Hz at cell centres, and the transverse fields on the intervening edges.
    PEC closes the outer boundary, normally behind the absorbing PML.
    """
    c0 = speed_of_light
    eps0 = epsilon_0

    def __init__(self, frequency, x_range, y_range, Nx, Ny):
        self.frequency = float(frequency)
        self.omega = 2*np.pi*self.frequency
        self.k0 = self.omega/self.c0
        self.Nx, self.Ny = int(Nx), int(Ny)
        self.x_range, self.y_range = float(x_range), float(y_range)
        self.dx, self.dy = self.x_range/self.Nx, self.y_range/self.Ny
        self.nodes_x = np.arange(self.Nx+1)*self.dx-self.x_range/2
        self.nodes_y = np.arange(self.Ny+1)*self.dy-self.y_range/2
        self.centres_x = self.nodes_x[:-1]+self.dx/2
        self.centres_y = self.nodes_y[:-1]+self.dy/2
        self.coordinates = {
            'Ez': (self.nodes_x, self.nodes_y),
            'Hx': (self.nodes_x, self.centres_y),
            'Hy': (self.centres_x, self.nodes_y),
            'Hz': (self.centres_x, self.centres_y),
            'Ex': (self.centres_x, self.nodes_y),
            'Ey': (self.nodes_x, self.centres_y),
        }
        for name in ('ERxx', 'ERyy', 'ERzz', 'MRxx', 'MRyy', 'MRzz'):
            setattr(self, name, np.ones((self.Ny, self.Nx), dtype=complex))
        self.pec_cells = np.zeros((self.Ny, self.Nx), dtype=bool)
        self._operators = {}
        self._systems = {}
        self._inset = None
        self._select_polarization('TE')
        self.source = np.zeros(self.N, dtype=complex)
        self.add_mask(0)

    def _select_polarization(self, polarization):
        if polarization not in ('TE', 'TM'):
            raise ValueError('polarization must be TE or TM.')
        self.polarization = polarization
        self.primary = 'Ez' if polarization == 'TE' else 'Hz'
        self.X, self.Y = np.meshgrid(*self.coordinates[self.primary], indexing='xy')
        self.primary_shape = self.X.shape
        self.N = self.X.size

    def _yeeder2d(self):
        if self.polarization not in self._operators:
            dx = node_to_cell_difference(self.Nx, self.k0*self.dx)
            dy = node_to_cell_difference(self.Ny, self.k0*self.dy)
            if self.polarization == 'TE':
                dex = sp.kron(sp.eye(self.Ny+1), dx, format='csr')
                dey = sp.kron(dy, sp.eye(self.Nx+1), format='csr')
            else:
                dex = sp.kron(sp.eye(self.Ny), dx, format='csr')
                dey = sp.kron(dy, sp.eye(self.Nx), format='csr')
            self._operators[self.polarization] = dex, dey, -dex.T, -dey.T
        return self._operators[self.polarization]

    def _field_materials(self):
        return {
            'Ex': node_average(self.ERxx, 0),
            'Ey': node_average(self.ERyy, 1),
            'Ez': node_average(node_average(self.ERzz, 0), 1),
            'Hx': node_average(self.MRxx, 1),
            'Hy': node_average(self.MRyy, 0),
            'Hz': self.MRzz,
        }

    def add_object(self, er_tensor, mr_tensor, region_mask):
        if np.isscalar(er_tensor):
            er_tensor = (er_tensor,)*3
        if np.isscalar(mr_tensor):
            mr_tensor = (mr_tensor,)*3
        for prefix, tensor in (('ER', er_tensor), ('MR', mr_tensor)):
            for component, value in zip(('xx', 'yy', 'zz'), tensor):
                getattr(self, prefix+component)[region_mask] = value
        self._systems.clear()

    def add_pec(self, region_mask):
        """Constrain a metal volume and its boundary at actual Yee sites."""
        self.pec_cells |= region_mask
        self._systems.clear()

    def pec_field_masks(self):
        x_nodes = occupied_nodes(self.pec_cells, 1)
        y_nodes = occupied_nodes(self.pec_cells, 0)
        return {'Ez': occupied_nodes(y_nodes, 1), 'Hz': self.pec_cells,
                'Hx': x_nodes, 'Hy': y_nodes, 'Ex': y_nodes, 'Ey': x_nodes}

    def add_source(self, src_type='plane_wave', angle_deg=0., polarization='TE',
                   location=None, amplitude=1.):
        self._select_polarization(polarization.upper())
        if src_type == 'plane_wave':
            theta = np.deg2rad(angle_deg)
            field = np.exp(-1j*self.k0*(np.cos(theta)*self.X+np.sin(theta)*self.Y))
        elif src_type == 'point':
            if location is None:
                raise ValueError('For a point source supply location=(x0,y0).')
            radius = np.hypot(self.X-location[0], self.Y-location[1])
            radius[radius == 0] = self.dx/50
            field = hankel2(0, self.k0*radius)
            if self.polarization == 'TM':
                field *= -1j/4
        else:
            raise ValueError(f'Unknown src_type {src_type}')
        self.source = (amplitude*field).ravel()
        self.set_total_field_region(self._inset or (0., 0.))

    def add_UPML(self, pml_width=20, n=3, sigma_max=5., direction='both'):
        pml_width = int(pml_width)
        if pml_width <= 0:
            raise ValueError('pml_width must be positive.')
        if not np.isfinite(sigma_max) or sigma_max < 0:
            raise ValueError('sigma_max must be finite and nonnegative.')
        if direction not in ('x', 'y', 'both'):
            raise ValueError("direction must be one of 'x', 'y', or 'both'.")
        limits = (self.Nx, self.Ny) if direction == 'both' else (self.Nx if direction == 'x' else self.Ny,)
        if any(2*pml_width > cells for cells in limits):
            raise ValueError('pml_width must fit in each selected direction.')
        # Private compatibility entry point uses conductivity in S/m.
        sigma_x = np.zeros((self.Ny, self.Nx))
        sigma_y = np.zeros_like(sigma_x)
        for i in range(pml_width):
            value = sigma_max*((pml_width-i)/pml_width)**n
            if direction in ('x', 'both'):
                sigma_x[:, i] = sigma_x[:, -i-1] = value
            if direction in ('y', 'both'):
                sigma_y[i, :] = sigma_y[-i-1, :] = value
        sx = 1-1j*sigma_x/(self.eps0*self.omega)
        sy = 1-1j*sigma_y/(self.eps0*self.omega)
        for prefix in ('ER', 'MR'):
            for component, scale in zip(('xx', 'yy', 'zz'), (sy/sx, sx/sy, sx*sy)):
                getattr(self, prefix+component)[:] *= scale
        self._systems.clear()

    def set_total_field_region(self, inset):
        """Compile the same physical rectangle on each component lattice."""
        inset = (inset, inset) if np.isscalar(inset) else tuple(inset)
        self._inset = inset
        self.field_masks = {}
        for name, (x, y) in self.coordinates.items():
            xx, yy = np.meshgrid(x, y, indexing='xy')
            inside = ((xx >= -self.x_range/2+inset[0]) &
                      (xx <= self.x_range/2-inset[0]) &
                      (yy >= -self.y_range/2+inset[1]) &
                      (yy <= self.y_range/2-inset[1]))
            self.field_masks[name] = sp.diags((~inside).ravel().astype(float), format='csr')
        self.Q = self.field_masks[self.primary]
        return self.Q

    def add_mask(self, value=30):
        if np.isscalar(value):
            return self.set_total_field_region((int(value)*self.dx, int(value)*self.dy))
        if sp.issparse(value):
            value = value.diagonal().reshape(self.primary_shape)
        value = np.asarray(value)
        if value.shape != self.primary_shape:
            raise ValueError(f'mask must be shape {self.primary_shape}')
        self.Q = sp.diags(value.ravel(), format='csr')
        self.field_masks[self.primary] = self.Q
        # General masks interpolate to transverse sites; the public rectangle
        # uses geometric masks evaluated independently at each Yee location.
        if self.polarization == 'TE':
            maps = {'Hx': .5*(value[:-1]+value[1:]),
                    'Hy': .5*(value[:, :-1]+value[:, 1:])}
        else:
            maps = {'Ex': node_average(value, 0), 'Ey': node_average(value, 1)}
        for name, mask in maps.items():
            self.field_masks[name] = sp.diags(mask.ravel(), format='csr')
        return self.Q

    def _inverse_transverse_materials(self):
        tensors = self._field_materials()
        if self.polarization == 'TE':
            return 1/tensors['Hx'], 1/tensors['Hy']
        inverse_ex = 1/tensors['Ex']
        inverse_ey = 1/tensors['Ey']
        # Tangential E is zero on the outer PEC walls. Both end traces exist.
        inverse_ex[[0, -1], :] = 0
        inverse_ey[:, [0, -1]] = 0
        masks = self.pec_field_masks()
        inverse_ex[masks['Ex']] = 0
        inverse_ey[masks['Ey']] = 0
        return inverse_ex, inverse_ey

    def _build_system(self):
        dex, dey, dhx, dhy = self._yeeder2d()
        inverse_x, inverse_y = self._inverse_transverse_materials()
        if self.polarization == 'TE':
            matrix = (dhx @ sp.diags(inverse_y.ravel()) @ dex +
                      dhy @ sp.diags(inverse_x.ravel()) @ dey +
                      sp.diags(self._field_materials()['Ez'].ravel()))
            free = np.ones(self.primary_shape, dtype=bool)
            free[[0, -1], :] = False
            free[:, [0, -1]] = False
        else:
            matrix = (dex @ sp.diags(inverse_y.ravel()) @ dhx +
                      dey @ sp.diags(inverse_x.ravel()) @ dhy +
                      sp.diags(self.MRzz.ravel()))
            free = np.ones(self.primary_shape, dtype=bool)
        free &= ~self.pec_field_masks()[self.primary]
        return matrix.tocsr(), free.ravel()

    def _solve(self, polarization, reuse_factorisation):
        if self.polarization != polarization:
            raise ValueError('The source polarization must match the solve.')
        if polarization not in self._systems:
            matrix, free = self._build_system()
            self._systems[polarization] = matrix, free, None
        matrix, free, factor = self._systems[polarization]
        rhs = (self.Q @ matrix-matrix @ self.Q) @ self.source
        reduced = matrix[free][:, free].tocsc()
        if reuse_factorisation and factor is None:
            factor = spla.factorized(reduced)
            self._systems[polarization] = matrix, free, factor
        values = np.zeros(self.N, dtype=complex)
        values[free] = factor(rhs[free]) if reuse_factorisation else spla.spsolve(reduced, rhs[free])
        field = values.reshape(self.primary_shape)
        setattr(self, self.primary, field)
        return field

    def solve_total_field_TE(self, reuse_factorisation=True):
        return self._solve('TE', reuse_factorisation)

    def solve_total_field_TM(self, reuse_factorisation=True):
        return self._solve('TM', reuse_factorisation)

    def transverse_fields(self, polarization):
        dex, dey, dhx, dhy = self._yeeder2d()
        eta0 = 1/(self.c0*self.eps0)
        inverse_x, inverse_y = self._inverse_transverse_materials()
        scalar = getattr(self, self.primary)

        def derivative(operator, name):
            return (operator @ scalar.ravel() +
                    (operator @ self.Q-self.field_masks[name] @ operator) @ self.source)

        if polarization == 'TE':
            values = {'Ez': scalar.ravel(),
                      'Hx': 1j*derivative(dey, 'Hx')*inverse_x.ravel()/eta0,
                      'Hy': -1j*derivative(dex, 'Hy')*inverse_y.ravel()/eta0}
        else:
            values = {'Hz': scalar.ravel(),
                      'Ex': -1j*eta0*derivative(dhy, 'Ex')*inverse_x.ravel(),
                      'Ey': 1j*eta0*derivative(dhx, 'Ey')*inverse_y.ravel()}
        fields = {name: field.reshape(len(self.coordinates[name][1]), len(self.coordinates[name][0]))
                  for name, field in values.items()}
        masks = self.pec_field_masks()
        for name, field in fields.items():
            field[masks[name]] = 0
        return fields

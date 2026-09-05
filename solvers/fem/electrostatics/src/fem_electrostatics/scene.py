"""Material-first static geometry and explicit conductor potentials."""
from dataclasses import dataclass
import numpy as np
from cem_common import materials, shapes
from cem_common.errors import BackendCapabilityError, ConfigurationError, GeometryError
from cem_common.scene import SceneMixin, GeometryHandle
from cem_common._shape_adapter import native_shape
from . import geometry as native


def static_epsilon(material, dim):
    eps, mu = materials.bulk_values(material, form='static', dimension=dim)
    if not np.array_equal(mu, np.eye(dim)):
        raise BackendCapabilityError('Electrostatics does not use magnetic permeability; use mu=1.')
    return eps


@dataclass(frozen=True)
class _Conductor:
    name: str
    shape: object


class ElectrostaticSceneMixin(SceneMixin):
    def _prepare_geometry(self, record):
        shape = record.shape
        if not isinstance(shape, shapes.Shape) or shape.dimension != self.dim:
            raise GeometryError('Shape must match the electrostatic dimension.')
        if record.clip:
            spans = (self.x_range,) if self.dim == 1 else (self.x_range, self.y_range)
            if self.dim == 1:
                shape = shapes.Interval(bounds=(max(shape.bounds[0], spans[0][0]), min(shape.bounds[1], spans[0][1])))
            else:
                shape = shapes.Intersection(shapes=(shape, shapes.Rectangle(bounds=spans)))
        shape = native_shape(shape, native)
        self.geometry.validate_shape(shape)
        if isinstance(record.material, materials.Material):
            return shape, native.Permittivity.from_input(static_epsilon(record.material, self.dim), self.dim)
        if record.material == materials.PEC:
            return shape, None
        raise BackendCapabilityError('Electrostatics supports bulk dielectric materials and PEC with an explicit potential; PMC/SIBC are not static models.')

    def _insert_geometry(self, record):
        shape, epsilon = self._prepare_geometry(record)
        if epsilon is None:
            self.geometry._changed()
            return _Conductor(record.name, shape)
        return self.geometry.add_material(shape, epsilon, name=record.name)

    def _delete_geometry(self, handle):
        if isinstance(handle, _Conductor):
            self.geometry.potentials[:] = [p for p in self.geometry.potentials if p.name != handle.name]
            self.geometry._changed()
        else:
            self.geometry.remove(handle)

    def _replace_geometry(self, handle, record):
        self._prepare_geometry(record)
        old_materials, old_potentials = list(self.geometry.materials), list(self.geometry.potentials)
        index = old_materials.index(handle) if handle in old_materials else None
        self._delete_geometry(handle)
        try:
            replacement = self._insert_geometry(record)
            if isinstance(handle, _Conductor) and isinstance(replacement, _Conductor):
                for p in old_potentials:
                    if p.name == handle.name:
                        self.geometry.add_potential(replacement.shape, p.value, name=replacement.name)
            elif index is not None and replacement in self.geometry.materials:
                self.geometry.materials.remove(replacement)
                self.geometry.materials.insert(index, replacement)
            return replacement
        except Exception:
            self.geometry.materials[:] = old_materials
            self.geometry.potentials[:] = old_potentials
            raise

    def _require_conductor_potentials(self):
        assigned = {p.name for p in self.geometry.potentials}
        missing = [native.name for _, native in self._objects.values() if isinstance(native, _Conductor) and native.name not in assigned]
        if missing:
            raise ConfigurationError(f'Assign an explicit potential before meshing conductors: {missing}. Floating charge-constrained conductors are not implemented.')

    def set_potential(self, *, geometry, potential, name=None):
        """Prescribe volts on a conductor handle, shape, or named domain boundary."""
        if isinstance(geometry, GeometryHandle):
            _, handle = self._owned(geometry)
            if not isinstance(handle, _Conductor):
                raise ConfigurationError('Potential on an object handle requires material=materials.PEC.')
            if not np.isfinite(float(potential)):
                raise ConfigurationError('Potential must be finite.')
            self.geometry.potentials[:] = [p for p in self.geometry.potentials if p.name != handle.name]
            return self.geometry.add_potential(handle.shape, potential, name=handle.name)
        shape = geometry if isinstance(geometry, str) else native_shape(geometry, native)
        return self.geometry.add_potential(shape, potential, name=name)

    def add_charge_density(self, *, geometry, density, name=None):
        """Assign volume charge density in C/m^3 to a dielectric handle or shape."""
        if isinstance(geometry, GeometryHandle):
            _, handle = self._owned(geometry)
            if isinstance(handle, _Conductor):
                raise ConfigurationError('Volume charge cannot be assigned inside a PEC conductor.')
            shape = handle.shape
        else:
            shape = native_shape(geometry, native)
        return self.geometry.add_charge(shape, density, name=name)

    def add_layer(self, *, x_range, material, name=None, clip=False):
        shape = shapes.Interval(bounds=x_range) if self.dim == 1 else shapes.Rectangle(bounds=(x_range, self.y_range))
        return self.add_geometry(shape=shape, material=material, name=name, clip=clip)
    def add_rectangle(self, *, x_range, y_range, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Rectangle(bounds=(x_range, y_range)), material=material, name=name, clip=clip)
    def add_circle(self, *, center, radius, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Circle(center=center, radius=radius), material=material, name=name, clip=clip)
    def add_polygon(self, *, points, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Polygon(points=points), material=material, name=name, clip=clip)

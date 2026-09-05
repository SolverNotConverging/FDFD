"""Material-first geometry API for the periodic FEM implementations."""
from dataclasses import replace
from cem_common import materials, shapes
from cem_common.errors import BackendCapabilityError, ConfigurationError, GeometryError
from cem_common.scene import SceneMixin
from cem_common._shape_adapter import native_shape
from . import geometry as native
from .materials import Material as NativeMaterial


class PeriodicSceneMixin(SceneMixin):
    def _prepare_geometry(self, record):
        if not isinstance(record.shape, shapes.Shape) or record.shape.dimension != len(self._physical_axes):
            raise GeometryError('Shape must match the solver dimension.')
        shape = record.shape
        spans = [getattr(self, axis+'_range') for axis in self._physical_axes]
        outside = any(lo < span[0] or hi > span[1] for lo, hi, span in zip(shape.bounds[::2], shape.bounds[1::2], spans))
        if outside:
            if not record.clip:
                raise GeometryError('Geometry lies outside the solver domain; use clip=True explicitly.')
            if shape.dimension == 1:
                shape = shapes.Interval(bounds=(max(shape.bounds[0], spans[0][0]), min(shape.bounds[1], spans[0][1])))
            else:
                shape = shapes.Intersection(shapes=(shape, (shapes.Box(bounds=spans) if shape.dimension == 3 else shapes.Rectangle(bounds=spans))))
        if isinstance(record.material, materials.Material):
            epsilon, mu = materials.bulk_values(record.material)
            material = NativeMaterial(epsilon, mu)
        elif isinstance(record.material, (materials.IdealBoundary, materials.GoodConductor, materials.SurfaceImpedance)):
            material = record.material
        else:
            raise ConfigurationError('Assign a predefined material object.')
        return native_shape(shape, native), material

    def _insert_geometry(self, record):
        shape, material = self._prepare_geometry(record)
        if isinstance(material, NativeMaterial):
            return self.geometry.add_region(shape, material, name=record.name)
        if not isinstance(material, materials.IdealBoundary):
            raise BackendCapabilityError('Periodic FEM does not implement SIBC; use a supported PEC/PMC assignment.')
        return self.geometry.add_boundary(shape, material.kind, name=record.name)

    def _delete_geometry(self, native_handle):
        self.geometry.remove(native_handle)

    def _replace_geometry(self, native_handle, record):
        self._prepare_geometry(record)
        collections = (self.geometry.regions, self.geometry.boundaries)
        collection = next(c for c in collections if native_handle in c)
        index = collection.index(native_handle)
        collection.pop(index)
        try:
            replacement = self._insert_geometry(record)
        except Exception:
            collection.insert(index, native_handle)
            raise
        target = next(c for c in collections if replacement in c)
        if target is collection:
            target.remove(replacement)
            target.insert(index, replacement)
        self.geometry._changed()
        return replacement

    def set_boundary(self, *, material):
        """Set the exterior PEC/PMC wall; internal SIBC uses geometry assignment."""
        if not isinstance(material, materials.IdealBoundary):
            raise BackendCapabilityError('Exterior walls support PEC/PMC; assign SIBC to a conductor shape.')
        self.geometry.set_outer_boundary(material.kind)


class PeriodicScene2D(PeriodicSceneMixin):
    _physical_axes = ('x', 'z')
    def add_rectangle(self, *, x_range, z_range, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Rectangle(bounds=(x_range, z_range)), material=material, name=name, clip=clip)
    def add_circle(self, *, center, radius, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Circle(center=center, radius=radius), material=material, name=name, clip=clip)
    def add_polygon(self, *, points, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Polygon(points=points), material=material, name=name, clip=clip)


class PeriodicScene3D(PeriodicSceneMixin):
    _physical_axes = ('x', 'y', 'z')
    def add_box(self, *, x_range, y_range, z_range, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Box(bounds=(x_range, y_range, z_range)), material=material, name=name, clip=clip)
    def add_sphere(self, *, center, radius, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Sphere(center=center, radius=radius), material=material, name=name, clip=clip)
    def add_cylinder(self, *, center, radius, z_range, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Cylinder(center=center, radius=radius, z_range=z_range), material=material, name=name, clip=clip)

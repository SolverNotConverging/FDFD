"""Material-first geometry for the separate waveguide-scattering formulation."""
from cem_common import materials, shapes
from cem_common.errors import BackendCapabilityError, ConfigurationError, GeometryError
from cem_common.scene import SceneMixin, GeometryHandle
from cem_common._shape_adapter import native_shape
from . import geometry as native
from .materials import Material as NativeMaterial


class ScatteringSceneMixin(SceneMixin):
    _physical_axes = ('x', 'z')
    def add_geometry(self, *, shape, material, name=None, clip=False, background=False):
        """Assign a material; background objects also belong to the straight lead."""
        return self._register_geometry(shape=shape, material=material, name=name, clip=clip, background=background)

    def _prepare_geometry(self, record):
        self._require_region_geometry()
        shape = record.shape
        if not isinstance(shape, shapes.Shape) or shape.dimension != 2:
            raise GeometryError('Scattering geometry must be a shared 2D shape in the xz plane.')
        if record.clip:
            if isinstance(shape, shapes.Segment):
                raise BackendCapabilityError('Clipping PEC segments is not supported; specify their physical endpoints.')
            shape = shapes.Intersection(shapes=(shape, shapes.Rectangle(bounds=(self.x_range, self.z_range))))
        if isinstance(record.material, materials.Material):
            eps, mu = materials.bulk_values(record.material, form='scalar')
            if not record.material.is_passive:
                raise ConfigurationError('Integrated scattering power accounting requires passive materials.')
            if record.background and (type(shape) is not shapes.Rectangle or tuple(shape.bounds[2:]) != self.z_range):
                raise GeometryError('Background-guide shapes must be rectangles spanning the complete z range.')
            self.geometry._inside_domain(shape.bounds[:2], shape.bounds[2:])
            return native_shape(shape, native), NativeMaterial(eps, mu)
        if record.material != materials.PEC or not isinstance(shape, shapes.Segment):
            raise BackendCapabilityError('Scattering supports PEC Segment sheets; solid PEC/PMC and SIBC are not implemented.')
        if shape.start[0] != shape.end[0]:
            raise BackendCapabilityError('Scattering PEC sheets must be parallel to z.')
        z = (min(shape.start[1], shape.end[1]), max(shape.start[1], shape.end[1]))
        if not record.background and z == self.z_range:
            raise ConfigurationError(
                'An inserted PEC sheet must be compact and lie strictly inside the z range.'
            )
        return shape, record.material

    def _insert_geometry(self, record):
        shape, material = self._prepare_geometry(record)
        if isinstance(material, NativeMaterial):
            result = self.geometry._append(self.geometry._next_name('region', record.name), shape, material, record.background)
        else:
            z = (min(shape.start[1], shape.end[1]), max(shape.start[1], shape.end[1]))
            if record.background and z != self.z_range:
                raise GeometryError('A background PEC segment must span the complete z range.')
            result = self.geometry.add_pec(x=shape.start[0], z='all' if record.background else z,
                                           background=record.background, name=record.name)
        self._invalidate()
        return result

    def _delete_geometry(self, handle):
        if handle in self.geometry.regions:
            self.geometry.regions.remove(handle)
        elif handle in self.geometry.pec_sheets:
            if any(slot.sheet_name == handle.name for slot in self.geometry.pec_slots):
                raise GeometryError('Remove dependent slots before removing their PEC sheet.')
            self.geometry.pec_sheets.remove(handle)
        else:
            raise GeometryError('Geometry handle has been removed.')
        self._invalidate()

    def _replace_geometry(self, handle, record):
        self._prepare_geometry(record)
        collection = self.geometry.regions if handle in self.geometry.regions else self.geometry.pec_sheets
        index = collection.index(handle)
        self._delete_geometry(handle)
        try:
            replacement = self._insert_geometry(record)
        except Exception:
            collection.insert(index, handle)
            raise
        if replacement in collection:
            collection.remove(replacement)
            collection.insert(index, replacement)
        return replacement

    def add_rectangle(self, *, x_range, z_range, material, name=None, clip=False, background=False):
        return self.add_geometry(shape=shapes.Rectangle(bounds=(x_range, z_range)), material=material, name=name, clip=clip, background=background)
    def add_circle(self, *, center, radius, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Circle(center=center, radius=radius), material=material, name=name, clip=clip)
    def add_polygon(self, *, points, material, name=None, clip=False):
        return self.add_geometry(shape=shapes.Polygon(points=points), material=material, name=name, clip=clip)
    def add_slot(self, *, geometry, z_range, name=None):
        """Cut a finite opening in an existing background PEC sheet."""
        _, handle = self._owned(geometry)
        result = self.geometry.add_slot(pec=handle, z=z_range, name=name)
        self._invalidate()
        return result

    def remove(self, *, geometry):
        """Remove an owned geometry object or the slot returned by add_slot()."""
        if isinstance(geometry, native.PECSlot):
            for index, slot in enumerate(self.geometry.pec_slots):
                if slot is geometry:
                    del self.geometry.pec_slots[index]
                    self._invalidate()
                    return
            raise GeometryError('Slot does not belong to this solver or was removed.')
        super().remove(geometry=geometry)

    def set_material_field(self, *, material, background_material):
        """Use named actual/background SpatialMaterial fields instead of objects."""
        if self._objects:
            raise ConfigurationError('Spatial material fields cannot be mixed with geometry objects.')
        if not isinstance(material, materials.SpatialMaterial) or not isinstance(background_material, materials.SpatialMaterial):
            raise ConfigurationError('Both fields must be predefined SpatialMaterial objects.')
        self._material_actual = material.epsilon
        self._material_background = background_material.epsilon
        self._spatial_materials = (material, background_material)
        self._invalidate()

    def set_boundary(self, *, material):
        if material != materials.PEC:
            raise BackendCapabilityError('Scattering transverse walls currently support PEC only.')
        self._boundary_kind = 'pec'
        self._invalidate()

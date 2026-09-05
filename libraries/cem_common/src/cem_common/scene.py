"""Material assignment and editing contracts, independent of discretization."""
from dataclasses import dataclass, replace
from weakref import ref
from . import shapes, materials
from .errors import ConfigurationError, GeometryError, BackendCapabilityError


@dataclass(frozen=True)
class GeometryRecord:
    name: str
    shape: shapes.Shape
    material: materials.MaterialAssignment
    clip: bool = False
    background: bool = False


class GeometryHandle:
    """Stable reference to a solver-owned object; edits go through the solver."""
    def __init__(self, owner, identifier):
        self._owner = ref(owner)
        self._id = identifier
    @property
    def id(self): return self._id
    @property
    def name(self): return self._record.name
    @property
    def shape(self): return self._record.shape
    @property
    def material(self): return self._record.material
    @property
    def clip(self): return self._record.clip
    @property
    def background(self): return self._record.background
    @property
    def _record(self):
        owner = self._owner()
        if owner is None or self.id not in owner._objects:
            raise GeometryError('This geometry object has been removed.')
        return owner._objects[self.id][0]


class SceneMixin:
    """Store physical objects; family adapters implement native insertion/removal."""
    def __setattr__(self, name, value):
        if name == 'frequency' and name in self.__dict__ and self.__dict__[name] != value:
            raise ConfigurationError('frequency is fixed for this solver; construct a new solver or use the supported sweep operation.')
        super().__setattr__(name, value)
    def _init_scene(self, *, background_material):
        if not isinstance(background_material, materials.Material):
            raise ConfigurationError('background_material must be a bulk Material object.')
        self._background_material = background_material
        self._objects = {}
        self._next_object_id = 1

    @property
    def background_material(self): return self._background_material
    @property
    def objects(self): return tuple(GeometryHandle(self, key) for key in self._objects)
    @property
    def plane(self): return ''.join(self._physical_axes)

    def _register_geometry(self, *, shape, material, name=None, clip=False, background=False):
        if not isinstance(shape, shapes.Shape):
            raise GeometryError('shape must be a cem_common.shapes object.')
        if shape.dimension != len(self._physical_axes):
            raise BackendCapabilityError(f'{type(shape).__name__} does not match the {self.plane} domain.')
        if not isinstance(material, (materials.Material, materials.IdealBoundary,
                                     materials.SurfaceImpedance, materials.GoodConductor)):
            raise ConfigurationError('material must be a material object; define it before assigning geometry.')
        if not isinstance(clip, bool) or not isinstance(background, bool):
            raise ConfigurationError('clip and background must be booleans.')
        identifier = self._next_object_id
        name = f'object_{identifier}' if name is None else name
        if not isinstance(name, str) or not name.strip() or any(v[0].name == name for v in self._objects.values()):
            raise GeometryError('Geometry names must be nonempty and unique within the solver.')
        record = GeometryRecord(name, shape, material, clip, background)
        native = self._insert_geometry(record)
        self._objects[identifier] = (record, native)
        self._next_object_id += 1
        return GeometryHandle(self, identifier)

    def add_geometry(self, *, shape, material, name=None, clip=False):
        """Assign a predefined material to a continuous shape in metres."""
        return self._register_geometry(shape=shape, material=material, name=name, clip=clip)

    def _owned(self, geometry):
        if not isinstance(geometry, GeometryHandle) or geometry._owner() is not self or geometry.id not in self._objects:
            raise GeometryError('Geometry handle does not belong to this solver or was removed.')
        return self._objects[geometry.id]

    def remove(self, *, geometry):
        """Remove an object and invalidate its mesh/result."""
        record, native = self._owned(geometry)
        self._delete_geometry(native)
        del self._objects[geometry.id]

    def _edit_geometry(self, geometry, changes):
        record, native = self._owned(geometry)
        updated = replace(record, **changes)
        # The adapter validates and creates a replacement before retiring the
        # original, preserving its position in the material-precedence order.
        replacement = self._replace_geometry(native, updated)
        self._objects[geometry.id] = (updated, replacement)
        return geometry

    def set_material(self, *, geometry, material):
        """Reassign a predefined material and invalidate mesh/result."""
        return self._edit_geometry(geometry, {'material': material})

    def set_shape(self, *, geometry, shape):
        """Replace a shape in metres and invalidate mesh/result."""
        return self._edit_geometry(geometry, {'shape': shape})

    def _scene_context(self):
        from .contracts import _context_value
        return {'background_material': _context_value(self.background_material),
                'objects': [_context_value(item[0]) for item in self._objects.values()]}

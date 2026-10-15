"""Geometry registry shared by every solver: named shapes with materials."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from weakref import ref

import numpy as np

from . import materials, shapes
from .errors import BackendCapabilityError, ConfigurationError, GeometryError


@dataclass(frozen=True)
class GeometryRecord:
    name: str
    shape: shapes.Shape
    material: materials.MaterialAssignment
    clip: bool = False


class GeometryHandle:
    """Stable reference to a solver-owned object; edits go through the solver."""

    def __init__(self, owner, identifier):
        self._owner = ref(owner)
        self._id = identifier

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._record.name

    @property
    def shape(self):
        return self._record.shape

    @property
    def material(self):
        return self._record.material

    @property
    def clip(self):
        return self._record.clip

    @property
    def _record(self):
        owner = self._owner()
        if owner is None or self.id not in owner._objects:
            raise GeometryError("This geometry object has been removed.")
        return owner._objects[self.id]

    def __repr__(self):
        return f"GeometryHandle(name={self.name!r}, material={self.material.name!r})"


def describe(value):
    """Describe configured geometry and materials as plain data for saved results."""
    if is_dataclass(value):
        return {
            "type": type(value).__name__,
            **{
                item.name: describe(getattr(value, item.name))
                for item in fields(value)
                if not item.name.startswith("_")
            },
        }
    if isinstance(value, Mapping):
        return {key: describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(describe(item) for item in value)
    if value is None or isinstance(value, (str, int, float, complex, bool, np.ndarray, np.generic)):
        return value
    return {"type": type(value).__name__}


class Scene:
    """Ordered collection of named shapes, each assigned one material.

    Later objects take precedence where shapes overlap. Every edit
    invalidates the mesh and the current result.
    """

    def _init_scene(self, *, background_material):
        if not isinstance(background_material, materials.Material):
            raise ConfigurationError("background_material must be a bulk Material object.")
        self._background_material = background_material
        self._objects = {}
        self._next_object_id = 1

    @property
    def background_material(self):
        """The bulk material filling the domain before any object is added."""
        return self._background_material

    @property
    def objects(self):
        """Handles to the added objects, in precedence order."""
        return tuple(GeometryHandle(self, key) for key in self._objects)

    def add_geometry(self, *, shape, material, name=None, clip=False):
        """Assign ``material`` to ``shape`` (coordinates in metres).

        Parameters
        ----------
        shape : fdfd.shapes.Shape
            Any shape whose dimension matches the solver's physical axes.
        material : Material, GoodConductor, SurfaceImpedance, or materials.PEC/PMC
            The assignment; each solver checks which kinds it supports.
        name : str, optional
            Unique label; defaults to ``object_<n>``.
        clip : bool
            Allow the shape to extend beyond the domain. Without it an
            out-of-domain shape raises :class:`GeometryError`.

        Returns
        -------
        GeometryHandle
            Pass it to :meth:`remove`, :meth:`set_material` or :meth:`set_shape`.
        """
        if not isinstance(shape, shapes.Shape):
            raise GeometryError("shape must be an fdfd.shapes object.")
        if shape.dimension != len(self._physical_axes):
            plane = "".join(self._physical_axes)
            raise BackendCapabilityError(f"{type(shape).__name__} does not match the {plane} domain.")
        if not isinstance(
            material,
            (
                materials.Material,
                materials.IdealBoundary,
                materials.SurfaceImpedance,
                materials.GoodConductor,
            ),
        ):
            raise ConfigurationError(
                "material must be a material object; define it before assigning geometry."
            )
        if not isinstance(clip, bool):
            raise ConfigurationError("clip must be a boolean.")
        identifier = self._next_object_id
        name = f"object_{identifier}" if name is None else name
        if (
            not isinstance(name, str)
            or not name.strip()
            or any(r.name == name for r in self._objects.values())
        ):
            raise GeometryError("Geometry names must be nonempty and unique within the solver.")
        record = GeometryRecord(name, shape, material, clip)
        self._validate_record(record)
        self._invalidate()
        self._objects[identifier] = record
        self._next_object_id += 1
        return GeometryHandle(self, identifier)

    def _owned(self, geometry):
        if (
            not isinstance(geometry, GeometryHandle)
            or geometry._owner() is not self
            or geometry.id not in self._objects
        ):
            raise GeometryError("Geometry handle does not belong to this solver or was removed.")
        return self._objects[geometry.id]

    def remove(self, *, geometry):
        """Remove an object added with ``add_*``."""
        self._owned(geometry)
        self._invalidate()
        del self._objects[geometry.id]

    def _edit(self, geometry, changes):
        updated = replace(self._owned(geometry), **changes)
        self._validate_record(updated)
        self._invalidate()
        self._objects[geometry.id] = updated
        return geometry

    def set_material(self, *, geometry, material):
        """Reassign the material of an existing object."""
        return self._edit(geometry, {"material": material})

    def set_shape(self, *, geometry, shape):
        """Replace the shape of an existing object (metres)."""
        return self._edit(geometry, {"shape": shape})

    def _scene_context(self):
        return {
            "background_material": describe(self.background_material),
            "objects": [describe(record) for record in self._objects.values()],
        }

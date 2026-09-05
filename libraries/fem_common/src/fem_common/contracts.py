"""Small shared contracts without coupling numerical implementations."""
from __future__ import annotations

from dataclasses import fields, dataclass, is_dataclass
from importlib import import_module
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .errors import ConfigurationError, NoResultError


def bounds(value, name="range"):
    """Normalize a physical extent or pair without implicit unit conversion."""
    raw = np.asarray(value)
    if raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise ConfigurationError(f"{name} must be a positive extent or finite increasing pair.")
    try:
        pair = (0.0, float(value)) if raw.ndim == 0 else tuple(float(v) for v in value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ConfigurationError(f"Invalid {name}.") from exc
    if len(pair) != 2 or not np.isfinite(pair).all() or pair[1] <= pair[0]:
        raise ConfigurationError(f"{name} must be a positive extent or finite increasing pair.")
    return pair


@dataclass(frozen=True)
class MeshSnapshot:
    """Physical mesh arrays for result inspection, independent of a mesher."""
    coordinates: np.ndarray
    elements: np.ndarray
    axes: tuple[str, ...]
    info: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @property
    def nodes(self):
        return self.coordinates

    def __post_init__(self):
        points = np.array(self.coordinates, dtype=float, copy=True)
        cells = np.array(self.elements, dtype=np.int64, copy=True)
        if points.ndim == 1:
            points = points[:, None]
        if points.ndim != 2 or points.shape[1] != len(self.axes) or not np.isfinite(points).all():
            raise ConfigurationError("Mesh coordinates do not match the physical axes.")
        if cells.ndim != 2 or np.any(cells < 0) or np.any(cells >= len(points)):
            raise ConfigurationError("Invalid mesh connectivity.")
        points.flags.writeable = cells.flags.writeable = False
        object.__setattr__(self, "coordinates", points)
        object.__setattr__(self, "elements", cells)
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "info", MappingProxyType(dict(self.info)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def mesh_snapshot(mesh, *, axes=None):
    if mesh is None or isinstance(mesh, MeshSnapshot):
        return mesh
    native = getattr(mesh, "mesh", mesh)
    points = getattr(mesh, "nodes", None)
    if points is None:
        points = native.p.T
    cells = getattr(mesh, "elements", None)
    if cells is None:
        cells = native.t.T
    points = np.asarray(points)
    dimension = 1 if points.ndim == 1 else points.shape[1]
    info = getattr(mesh, "info", {})
    if is_dataclass(info):
        info = {item.name: getattr(info, item.name) for item in fields(info)}
    metadata = {}
    for name in ("element_tags", "physical_names", "boundary_facets", "slave_nodes", "master_nodes", "geometry_revision"):
        value = getattr(mesh, name, None)
        if value is not None:
            metadata[name] = value
    return MeshSnapshot(points, cells, tuple(axes or ("x", "y", "z")[:dimension]), info, metadata)


def _context_value(value):
    """Describe configured geometry as data, without serializing callbacks."""
    if is_dataclass(value):
        return {"type": type(value).__name__, **{item.name: _context_value(getattr(value, item.name))
                for item in fields(value) if not item.name.startswith("_")}}
    if isinstance(value, Mapping):
        return {key: _context_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_context_value(item) for item in value)
    if callable(value):
        return {"type": "callback", "serialized": False}
    if value is None or isinstance(value, (str, int, float, complex, bool, np.ndarray, np.generic)):
        return value
    return {"type": type(value).__name__}


class FEMSolverMixin:
    """Result presentation and finalization shared by public solver classes."""
    def show(self, *, block: bool = True):
        if self.result is None:
            raise NoResultError("Call solve() before show(); the solver has no current result.")
        return self.result.show(block=block)

    def _finish_result(self, result):
        axes = getattr(self, "_physical_axes", None)
        if axes is None:
            axes = ("x", "z") if "periodic" in type(self).__module__ and hasattr(self, "z_span") and not hasattr(self, "y_span") else None
        snapshot = mesh_snapshot(self.mesh_data, axes=axes)
        geometry = getattr(self, "geometry", None)
        context = {key: _context_value(getattr(geometry, key)) for key in
            ("background", "exterior", "regions", "materials", "boundaries", "potentials", "charges",
             "pmls", "outer_boundary", "pec_sheets", "pec_slots") if hasattr(geometry, key)}
        context.update({key: _context_value(getattr(self, key)) for key in
            ("outer_potential", "transverse_boundary", "pml") if hasattr(self, key)})
        snapshot = MeshSnapshot(snapshot.coordinates, snapshot.elements, snapshot.axes,
            snapshot.info, {**snapshot.metadata, "context": context})
        object.__setattr__(result, "_mesh_snapshot", snapshot)
        self._result = result
        return result


class ElectromagneticSolverMixin(FEMSolverMixin):
    @property
    def wavelength(self):
        """Free-space wavelength in metres, derived from frequency in hertz."""
        return 299_792_458.0 / self.frequency


class ResultMixin:
    """The supported plotting and persistence surface of a numerical result."""
    @property
    def solve_info(self):
        return MappingProxyType({key: value for key, value in self.metadata.items()
            if key.startswith(("adaptive_", "max_refinements", "residual", "eigensolver", "solver", "converged", "rejected", "ndofs"))})

    def _result_api(self):
        family = type(self).__module__.split(".")[0]
        return import_module(f"{family}.result_api")

    @property
    def mesh_data(self):
        snapshot = getattr(self, "_mesh_snapshot", None)
        if snapshot is not None:
            return snapshot
        native = getattr(self, "mesh", None)
        return mesh_snapshot(native) if native is not None else None

    def show(self, *, block: bool = True):
        if not isinstance(block, bool):
            raise ConfigurationError("block must be a boolean.")
        return self._result_api().show_result(self, block=block)

    def plot(self, *, component: str | None = None, quantity: str = "real", mode: int = 0):
        return self._result_api().plot_result(self, component=component, quantity=quantity, mode=mode)

    def save(self, path):
        return self._result_api().save_result(self, path)

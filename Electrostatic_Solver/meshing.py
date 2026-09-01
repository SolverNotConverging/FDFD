"""Gmsh discretization imported directly into scikit-fem meshes."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from .exceptions import MeshError
from .geometry import Circle, GeometryModel, Interval, Polygon, Rectangle


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_GMSH_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class MeshInfo:
    nodes: int
    elements: int
    minimum_edge: float
    maximum_edge: float
    requested_maximum_edge: float
    material_aware: bool
    interface_refinement: float | None
    boundary_refinement: float | None
    material_element_sizes: dict[str, float]


@dataclass(frozen=True, slots=True)
class FEMMesh:
    mesh: object
    nodes: FloatArray
    elements: IntArray
    element_tags: NDArray[np.int32]
    physical_names: dict[int, str]
    info: MeshInfo
    geometry_revision: int


def _positive(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise MeshError(f"{name} must be finite and positive.")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise MeshError(f"{name} must be finite and positive.")
    return numeric


def _factor(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise MeshError(f"{name} must be in (0, 1] or None.")
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 < numeric <= 1.0:
        raise MeshError(f"{name} must be in (0, 1] or None.")
    return numeric


def _import_gmsh() -> object:
    try:
        import gmsh
    except Exception as exc:  # pragma: no cover - depends on host installation
        raise MeshError(
            "Gmsh could not be imported; install the package dependencies with "
            "`pip install -e Electrostatic_Solver[test]`."
        ) from exc
    return gmsh


def _mesh_extrema(points: FloatArray, elements: IntArray) -> tuple[float, float]:
    if elements.shape[1] == 2:
        lengths = np.linalg.norm(points[elements[:, 1]] - points[elements[:, 0]], axis=1)
    else:
        pairs = np.concatenate(
            (elements[:, [0, 1]], elements[:, [1, 2]], elements[:, [2, 0]])
        )
        lengths = np.linalg.norm(points[pairs[:, 1]] - points[pairs[:, 0]], axis=1)
    return float(lengths.min()), float(lengths.max())


def _material_targets(
    geometry: GeometryModel,
    maximum: float,
    material_aware: bool,
) -> tuple[dict[int, float], dict[int, str]]:
    table = geometry.material_table
    names = {1: "background", **{index: region.name for index, region in enumerate(geometry.materials, start=2)}}
    if not material_aware:
        return {tag: maximum for tag in table}, names
    scales = {tag: material.dk_scale for tag, material in table.items()}
    reference = max(min(scales.values(), default=1.0), np.finfo(float).tiny)
    return (
        {
            tag: min(maximum, maximum * reference / max(scale, np.finfo(float).tiny))
            for tag, scale in scales.items()
        },
        names,
    )


def _gmsh_session(gmsh: object) -> tuple[bool, str, dict[str, float]]:
    owned = not bool(gmsh.isInitialized())
    if owned:
        gmsh.initialize()
    option_names = (
        "General.Terminal",
        "Mesh.ElementOrder",
        "Mesh.MeshSizeMax",
        "Mesh.MeshSizeMin",
        "Mesh.MeshSizeExtendFromBoundary",
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
    )
    previous = {} if owned else {name: float(gmsh.option.getNumber(name)) for name in option_names}
    return owned, "" if owned else str(gmsh.model.getCurrent()), previous


def _finish_gmsh(
    gmsh: object,
    owned: bool,
    previous_model: str,
    previous_options: dict[str, float],
    model_name: str,
    model_added: bool,
) -> None:
    if owned:
        gmsh.finalize()
        return
    if model_added:
        try:
            gmsh.model.setCurrent(model_name)
            gmsh.model.remove()
            if previous_model:
                gmsh.model.setCurrent(previous_model)
        except Exception:
            pass
    for name, value in previous_options.items():
        try:
            gmsh.option.setNumber(name, value)
        except Exception:
            pass


def discretize_1d(
    geometry: GeometryModel,
    *,
    max_element_size: float,
    material_aware: bool = True,
    interface_refinement: float | None = 0.7,
    boundary_refinement: float | None = 0.5,
) -> FEMMesh:
    """Generate a conforming Gmsh line mesh with high-Dk local sizing."""

    if geometry.dim != 1:
        raise MeshError("discretize_1d requires 1D geometry.")
    maximum = _positive(max_element_size, "max_element_size")
    interface_factor = _factor(interface_refinement, "interface_refinement")
    boundary_factor = _factor(boundary_refinement, "boundary_refinement")
    if not isinstance(material_aware, (bool, np.bool_)):
        raise MeshError("material_aware must be a boolean.")

    cuts = {geometry.x_span[0], geometry.x_span[1]}
    for shape in geometry.all_area_shapes():
        assert isinstance(shape, Interval)
        cuts.update(shape.x)
    ordered = sorted(cuts)
    centers = np.asarray([[(left + right) * 0.5] for left, right in zip(ordered[:-1], ordered[1:], strict=True)])
    segment_tags = geometry.material_indices_at(centers)
    targets, names = _material_targets(geometry, maximum, bool(material_aware))

    constrained_points = {geometry.x_span[0], geometry.x_span[1]}
    for potential in geometry.potentials:
        if isinstance(potential.shape, Interval):
            constrained_points.update(potential.shape.x)
        elif potential.shape == "left":
            constrained_points.add(geometry.x_span[0])
        elif potential.shape == "right":
            constrained_points.add(geometry.x_span[1])

    point_sizes: list[float] = []
    for index, coordinate in enumerate(ordered):
        adjacent = []
        if index:
            adjacent.append(targets[int(segment_tags[index - 1])])
        if index < len(segment_tags):
            adjacent.append(targets[int(segment_tags[index])])
        target = min(adjacent)
        if interface_factor is not None and index not in (0, len(ordered) - 1):
            left_tag = int(segment_tags[index - 1])
            right_tag = int(segment_tags[index])
            if left_tag != right_tag:
                target *= interface_factor
        if boundary_factor is not None and coordinate in constrained_points:
            target *= boundary_factor
        point_sizes.append(target)

    gmsh = _import_gmsh()
    xmin, xmax = geometry.x_span
    scale = 1.0 / (xmax - xmin)
    model_name = f"electrostatic_1d_{uuid4().hex}"
    with _GMSH_LOCK:
        owned, previous_model, previous_options = _gmsh_session(gmsh)
        model_added = False
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(model_name)
            model_added = True
            occ = gmsh.model.occ
            point_tags = [
                occ.addPoint((coordinate - xmin) * scale, 0.0, 0.0, point_sizes[index] * scale)
                for index, coordinate in enumerate(ordered)
            ]
            line_tags = [
                occ.addLine(point_tags[index], point_tags[index + 1])
                for index in range(len(point_tags) - 1)
            ]
            occ.synchronize()
            for tag in sorted(set(int(value) for value in segment_tags)):
                entities = [line_tags[index] for index, value in enumerate(segment_tags) if int(value) == tag]
                gmsh.model.addPhysicalGroup(1, entities, tag)
                gmsh.model.setPhysicalName(1, tag, names[tag])
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.option.setNumber("Mesh.MeshSizeMax", maximum * scale)
            gmsh.option.setNumber("Mesh.MeshSizeMin", max(min(point_sizes) * scale * 0.1, 1e-12))
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.model.mesh.generate(1)
            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            points = np.asarray(coordinates, dtype=float).reshape(-1, 3)[:, :1] / scale + xmin
            element_tags, connectivity = gmsh.model.mesh.getElementsByType(1)
            if not len(element_tags):
                raise MeshError("Gmsh generated no first-order line elements.")
            lookup = {int(tag): index for index, tag in enumerate(node_tags)}
            raw = np.asarray(connectivity, dtype=np.int64).reshape(-1, 2)
            edges = np.fromiter(
                (lookup[int(tag)] for tag in raw.ravel()), dtype=np.int64, count=raw.size
            ).reshape(-1, 2)
        except MeshError:
            raise
        except Exception as exc:
            raise MeshError(f"Gmsh failed to generate the 1D electrostatic mesh: {exc}") from exc
        finally:
            _finish_gmsh(gmsh, owned, previous_model, previous_options, model_name, model_added)

    order = np.argsort(points[:, 0])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    points = points[order]
    edges = inverse[edges]
    centers = points[edges].mean(axis=1)
    tags = geometry.material_indices_at(centers)
    try:
        from skfem import MeshLine
    except Exception as exc:  # pragma: no cover - depends on host installation
        raise MeshError("scikit-fem could not be imported; install the package dependencies.") from exc
    mesh = MeshLine(points.T, edges.T)
    hmin, hmax = _mesh_extrema(points, edges)
    return FEMMesh(
        mesh=mesh,
        nodes=points,
        elements=edges,
        element_tags=tags,
        physical_names=names,
        info=MeshInfo(
            nodes=len(points),
            elements=len(edges),
            minimum_edge=hmin,
            maximum_edge=hmax,
            requested_maximum_edge=maximum,
            material_aware=bool(material_aware),
            interface_refinement=interface_factor,
            boundary_refinement=boundary_factor,
            material_element_sizes=MappingProxyType({names[tag]: value for tag, value in targets.items()}),  # type: ignore[arg-type]
        ),
        geometry_revision=geometry.revision,
    )


def _add_occ_shape(gmsh: object, shape: object, origin: tuple[float, float], scale: float) -> tuple[int, int]:
    occ = gmsh.model.occ
    x0, y0 = origin
    if isinstance(shape, Circle):
        return 2, occ.addDisk(
            (shape.center[0] - x0) * scale,
            (shape.center[1] - y0) * scale,
            0.0,
            shape.radius * scale,
            shape.radius * scale,
        )
    if isinstance(shape, Polygon):
        points = [occ.addPoint((x - x0) * scale, (y - y0) * scale, 0.0) for x, y in shape.points]
        lines = [occ.addLine(points[index], points[(index + 1) % len(points)]) for index in range(len(points))]
        loop = occ.addCurveLoop(lines)
        return 2, occ.addPlaneSurface([loop])
    raise MeshError(f"unsupported curved Gmsh shape {type(shape).__name__}.")


def _add_distance_field(
    gmsh: object,
    curves: list[int],
    target: float,
    maximum: float,
    transition: float,
    scale: float,
    fields: list[int],
) -> None:
    if not curves:
        return
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "CurvesList", curves)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", target * scale)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", maximum * scale)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", transition * scale)
    fields.append(threshold)


def discretize_2d(
    geometry: GeometryModel,
    *,
    max_element_size: float,
    material_aware: bool = True,
    interface_refinement: float | None = 0.7,
    boundary_refinement: float | None = 0.5,
    interface_refinement_width: float | None = None,
    boundary_refinement_width: float | None = None,
) -> FEMMesh:
    """Generate a conforming triangular mesh with high-Dk and boundary refinement."""

    if geometry.dim != 2 or geometry.y_span is None:
        raise MeshError("discretize_2d requires 2D geometry.")
    maximum = _positive(max_element_size, "max_element_size")
    interface_factor = _factor(interface_refinement, "interface_refinement")
    boundary_factor = _factor(boundary_refinement, "boundary_refinement")
    if not isinstance(material_aware, (bool, np.bool_)):
        raise MeshError("material_aware must be a boolean.")
    interface_width = None if interface_refinement_width is None else _positive(interface_refinement_width, "interface_refinement_width")
    boundary_width = None if boundary_refinement_width is None else _positive(boundary_refinement_width, "boundary_refinement_width")
    if interface_width is not None and interface_factor is None:
        raise MeshError("interface_refinement_width requires interface_refinement.")
    if boundary_width is not None and boundary_factor is None:
        raise MeshError("boundary_refinement_width requires boundary_refinement.")

    targets, names = _material_targets(geometry, maximum, bool(material_aware))
    xmin, xmax = geometry.x_span
    ymin, ymax = geometry.y_span
    width, height = xmax - xmin, ymax - ymin
    scale = 1.0 / max(width, height)
    gmsh = _import_gmsh()
    model_name = f"electrostatic_2d_{uuid4().hex}"

    with _GMSH_LOCK:
        owned, previous_model, previous_options = _gmsh_session(gmsh)
        model_added = False
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(model_name)
            model_added = True
            occ = gmsh.model.occ

            xcuts = {xmin, xmax}
            ycuts = {ymin, ymax}
            shapes = geometry.all_area_shapes()
            for shape in shapes:
                if isinstance(shape, Rectangle):
                    xcuts.update(shape.x)
                    ycuts.update(shape.y)
            xs = sorted(xcuts)
            ys = sorted(ycuts)
            cells = [
                (
                    2,
                    occ.addRectangle(
                        (xs[ix] - xmin) * scale,
                        (ys[iy] - ymin) * scale,
                        0.0,
                        (xs[ix + 1] - xs[ix]) * scale,
                        (ys[iy + 1] - ys[iy]) * scale,
                    ),
                )
                for ix in range(len(xs) - 1)
                for iy in range(len(ys) - 1)
            ]
            curved_shapes = list(
                dict.fromkeys(shape for shape in shapes if not isinstance(shape, Rectangle))
            )
            curved = [
                _add_occ_shape(gmsh, shape, (xmin, ymin), scale)
                for shape in curved_shapes
            ]
            inputs = [*cells, *curved]
            if len(inputs) > 1:
                _, fragment_map = occ.fragment(
                    [inputs[0]], inputs[1:], removeObject=True, removeTool=True
                )
                if len(fragment_map) != len(inputs):
                    raise MeshError("Gmsh returned incomplete OCC fragment provenance.")
                curved_membership = {
                    shape: {
                        int(entity)
                        for dimension, entity in fragment_map[len(cells) + index]
                        if dimension == 2
                    }
                    for index, shape in enumerate(curved_shapes)
                }
            else:
                curved_membership = {}
            occ.synchronize()

            surfaces = [int(entity) for _, entity in gmsh.model.getEntities(2)]
            if not surfaces:
                raise MeshError("Gmsh produced no surfaces after geometry fragmentation.")
            surface_centers: dict[int, tuple[float, float]] = {}
            surface_tags: dict[int, int] = {}
            surface_potentials: dict[int, tuple[int, ...]] = {}
            for entity in surfaces:
                center = occ.getCenterOfMass(2, entity)
                x = center[0] / scale + xmin
                y = center[1] / scale + ymin
                surface_centers[entity] = (x, y)
                material_tag = 1
                for index, region in enumerate(geometry.materials, start=2):
                    belongs = (
                        bool(region.shape.contains(x, y))
                        if isinstance(region.shape, Rectangle)
                        else entity in curved_membership.get(region.shape, set())
                    )
                    if belongs:
                        material_tag = index
                surface_tags[entity] = material_tag
                owners: list[int] = []
                for potential in geometry.potentials:
                    if isinstance(potential.shape, str):
                        continue
                    belongs = (
                        bool(potential.shape.contains(x, y))
                        if isinstance(potential.shape, Rectangle)
                        else entity in curved_membership.get(potential.shape, set())
                    )
                    if belongs:
                        owners.append(potential.id)
                surface_potentials[entity] = tuple(owners)

            grouped: dict[int, list[int]] = {}
            for entity, tag in surface_tags.items():
                grouped.setdefault(tag, []).append(entity)
            for tag, entities in grouped.items():
                gmsh.model.addPhysicalGroup(2, entities, tag)
                gmsh.model.setPhysicalName(2, tag, names[tag])

            fields: list[int] = []
            target_values = [maximum]
            if material_aware:
                for tag, entities in grouped.items():
                    target = targets[tag]
                    if target >= maximum * (1.0 - 32.0 * np.finfo(float).eps):
                        continue
                    field = gmsh.model.mesh.field.add("Constant")
                    gmsh.model.mesh.field.setNumber(field, "VIn", target * scale)
                    gmsh.model.mesh.field.setNumber(field, "VOut", maximum * scale)
                    gmsh.model.mesh.field.setNumbers(field, "SurfacesList", entities)
                    fields.append(field)
                    target_values.append(target)

            interface_curves: dict[float, list[int]] = {}
            boundary_curves: dict[float, list[int]] = {}
            surface_set = set(surfaces)
            for _, curve in gmsh.model.getEntities(1):
                upward, _ = gmsh.model.getAdjacencies(1, curve)
                adjacent = [int(entity) for entity in upward if int(entity) in surface_set]
                adjacent_tags = {surface_tags[entity] for entity in adjacent}
                if interface_factor is not None and len(adjacent_tags) > 1:
                    target = interface_factor * min(targets[tag] for tag in adjacent_tags)
                    interface_curves.setdefault(target, []).append(int(curve))
                is_outer = len(adjacent) == 1
                is_potential_edge = len(adjacent) == 2 and surface_potentials[adjacent[0]] != surface_potentials[adjacent[1]]
                if boundary_factor is not None and (is_outer or is_potential_edge):
                    target = boundary_factor * min(targets[surface_tags[entity]] for entity in adjacent)
                    boundary_curves.setdefault(target, []).append(int(curve))

            for target, curves in interface_curves.items():
                _add_distance_field(
                    gmsh,
                    curves,
                    target,
                    maximum,
                    interface_width or 2.0 * target,
                    scale,
                    fields,
                )
                target_values.append(target)
            for target, curves in boundary_curves.items():
                _add_distance_field(
                    gmsh,
                    curves,
                    target,
                    maximum,
                    boundary_width or 3.0 * target,
                    scale,
                    fields,
                )
                target_values.append(target)
            if len(fields) == 1:
                gmsh.model.mesh.field.setAsBackgroundMesh(fields[0])
            elif fields:
                combined = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(combined, "FieldsList", fields)
                gmsh.model.mesh.field.setAsBackgroundMesh(combined)

            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.option.setNumber("Mesh.MeshSizeMax", maximum * scale)
            gmsh.option.setNumber("Mesh.MeshSizeMin", max(min(target_values) * scale * 0.05, 1e-12))
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.model.mesh.generate(2)

            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            points = np.asarray(coordinates, dtype=float).reshape(-1, 3)[:, :2] / scale
            points += np.asarray((xmin, ymin))
            element_tags, connectivity = gmsh.model.mesh.getElementsByType(2)
            if not len(element_tags):
                raise MeshError("Gmsh generated no first-order triangular elements.")
            lookup = {int(tag): index for index, tag in enumerate(node_tags)}
            raw = np.asarray(connectivity, dtype=np.int64).reshape(-1, 3)
            triangles = np.fromiter(
                (lookup[int(tag)] for tag in raw.ravel()), dtype=np.int64, count=raw.size
            ).reshape(-1, 3)
        except MeshError:
            raise
        except Exception as exc:
            raise MeshError(f"Gmsh failed to generate the 2D electrostatic mesh: {exc}") from exc
        finally:
            _finish_gmsh(gmsh, owned, previous_model, previous_options, model_name, model_added)

    element_centers = points[triangles].mean(axis=1)
    tags = geometry.material_indices_at(element_centers)
    try:
        from skfem import MeshTri
    except Exception as exc:  # pragma: no cover - depends on host installation
        raise MeshError("scikit-fem could not be imported; install the package dependencies.") from exc
    mesh = MeshTri(points.T, triangles.T)
    hmin, hmax = _mesh_extrema(points, triangles)
    return FEMMesh(
        mesh=mesh,
        nodes=points,
        elements=triangles,
        element_tags=tags,
        physical_names=names,
        info=MeshInfo(
            nodes=len(points),
            elements=len(triangles),
            minimum_edge=hmin,
            maximum_edge=hmax,
            requested_maximum_edge=maximum,
            material_aware=bool(material_aware),
            interface_refinement=interface_factor,
            boundary_refinement=boundary_factor,
            material_element_sizes=MappingProxyType({names[tag]: value for tag, value in targets.items()}),  # type: ignore[arg-type]
        ),
        geometry_revision=geometry.revision,
    )


__all__ = ["FEMMesh", "MeshInfo", "discretize_1d", "discretize_2d"]

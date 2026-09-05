"""One- and two-dimensional conforming FEM discretization."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from skfem import MeshLine, MeshTri

from .exceptions import MeshError
from .geometry import Circle, GeometryModel1D, GeometryModel2D, Interval, Polygon, Rectangle


_GMSH_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class MeshInfo:
    nodes: int
    elements: int
    minimum_edge: float
    maximum_edge: float
    requested_maximum_edge: float
    element_order: int
    material_aware: bool = False
    interface_refinement: float | None = None
    boundary_refinement: float | None = None
    refinement_regions: int = 0


@dataclass(frozen=True, slots=True)
class FEMMesh1D:
    mesh: MeshLine
    nodes: NDArray[np.float64]
    info: MeshInfo
    geometry_revision: int


@dataclass(frozen=True, slots=True)
class FEMMesh2D:
    mesh: MeshTri
    element_tags: NDArray[np.int32]
    physical_names: dict[int, str]
    boundary_facets: dict[str, NDArray[np.int64]]
    info: MeshInfo
    geometry_revision: int

    @property
    def nodes(self) -> NDArray[np.float64]:
        return np.asarray(self.mesh.p.T, dtype=float)

    @property
    def elements(self) -> NDArray[np.int64]:
        return np.asarray(self.mesh.t.T, dtype=np.int64)


def discretize_1d(
    geometry: GeometryModel1D,
    *,
    resolution: int | None = None,
    max_element_size: float | None = None,
    element_order: int = 1,
    vacuum_wavenumber: float | None = None,
    wavelength_elements: int = 10,
    material_aware: bool = True,
) -> FEMMesh1D:
    """Create an interface-conforming, material-aware line mesh.

    ``resolution`` and ``max_element_size`` define the target in the
    lowest-index interval.  In material-aware mode that size is reduced in
    proportion to the local index estimate
    ``sqrt(max(abs(epsilon)) * max(abs(mu)))``, so a high-Dk interval receives
    more elements without moving its exact material interfaces.  When
    ``vacuum_wavenumber`` is supplied, the local size is also limited to one
    material wavelength divided by ``wavelength_elements``.

    Set ``material_aware=False`` for a uniform target size.  The shortest
    wavelength is still respected, but its target is then applied throughout
    the domain instead of only in the high-wavenumber intervals.
    """

    if isinstance(element_order, (bool, np.bool_)) or element_order != 1:
        raise MeshError("The current 1D backend supports element_order=1.")
    width = geometry.x_span[1] - geometry.x_span[0]
    if isinstance(max_element_size, (bool, np.bool_)):
        raise MeshError("max_element_size must be finite and positive.")
    if resolution is None and max_element_size is None:
        resolution = 24
    if resolution is not None:
        if isinstance(resolution, bool) or int(resolution) != resolution or resolution < 2:
            raise MeshError("resolution must be an integer of at least two elements.")
        target = width / int(resolution)
        if max_element_size is not None:
            target = min(target, float(max_element_size))
    else:
        target = float(max_element_size)  # type: ignore[arg-type]
    if not np.isfinite(target) or target <= 0.0:
        raise MeshError("max_element_size must be finite and positive.")
    if (
        isinstance(wavelength_elements, (bool, np.bool_))
        or int(wavelength_elements) != wavelength_elements
        or wavelength_elements < 4
    ):
        raise MeshError("wavelength_elements must be an integer of at least four.")
    wavelength_count = int(wavelength_elements)
    if not isinstance(material_aware, (bool, np.bool_)):
        raise MeshError("material_aware must be a boolean.")
    if vacuum_wavenumber is not None:
        k0 = float(vacuum_wavenumber)
        if not np.isfinite(k0) or k0 <= 0.0:
            raise MeshError("vacuum_wavenumber must be finite and positive.")
    else:
        k0 = None

    interfaces = {geometry.x_span[0], geometry.x_span[1]}
    for region in geometry.regions:
        if isinstance(region.shape, Interval):
            interfaces.update(region.shape.x)
    for boundary in geometry.boundaries:
        if isinstance(boundary.shape, Interval):
            interfaces.update(boundary.shape.x)
    for pml in geometry.pmls:
        if pml.direction in ("x-", "x", "all"):
            interfaces.add(geometry.x_span[0] + pml.thickness)
        if pml.direction in ("x+", "x", "all"):
            interfaces.add(geometry.x_span[1] - pml.thickness)

    ordered = sorted(
        value
        for value in interfaces
        if geometry.x_span[0] <= value <= geometry.x_span[1]
    )
    midpoints = np.asarray(
        [
            0.5 * (left + right)
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        ],
        dtype=float,
    )
    epsilon, mu = geometry.material_at(midpoints)
    local_index = np.sqrt(
        np.max(np.abs(epsilon), axis=0) * np.max(np.abs(mu), axis=0)
    )
    if not np.isfinite(local_index).all():
        raise MeshError("The local material wavenumber estimate is not finite.")
    reference_index = max(float(np.min(local_index)), np.finfo(float).tiny)
    if bool(material_aware):
        size_density = np.maximum(local_index / reference_index, 1.0)
        wavelength_density = np.maximum(local_index, np.finfo(float).tiny)
    else:
        size_density = np.ones_like(local_index)
        wavelength_density = np.full_like(
            local_index,
            max(float(np.max(local_index)), np.finfo(float).tiny),
        )

    nodes: list[float] = []
    for segment, (left, right) in enumerate(
        zip(ordered[:-1], ordered[1:], strict=True)
    ):
        local_target = target / float(size_density[segment])
        if k0 is not None:
            local_k = k0 * float(wavelength_density[segment])
            wavelength_target = (
                np.inf
                if local_k <= np.finfo(float).tiny
                else 2.0 * np.pi / (local_k * wavelength_count)
            )
            local_target = min(local_target, wavelength_target)
        count = max(1, int(np.ceil((right - left) / local_target)))
        nodes.extend(np.linspace(left, right, count + 1)[:-1])
    nodes.append(ordered[-1])
    coordinates = np.asarray(nodes, dtype=float)
    edges = np.diff(coordinates)
    return FEMMesh1D(
        mesh=MeshLine(coordinates),
        nodes=coordinates,
        info=MeshInfo(
            nodes=coordinates.size,
            elements=coordinates.size - 1,
            minimum_edge=float(edges.min()),
            maximum_edge=float(edges.max()),
            requested_maximum_edge=float(target),
            element_order=element_order,
            material_aware=bool(material_aware),
        ),
        geometry_revision=geometry.revision,
    )


def _add_occ_shape(gmsh: object, shape: object, origin: tuple[float, float], scale: float) -> tuple[int, int]:
    from cem_common.shapes import Shape
    from cem_common._occ import add_shape
    if isinstance(shape, Shape):
        return add_shape(gmsh, shape, origin, scale)
    occ = gmsh.model.occ
    x0, y0 = origin
    if isinstance(shape, Rectangle):
        return (
            2,
            occ.addRectangle(
                (shape.x[0] - x0) * scale,
                (shape.y[0] - y0) * scale,
                0.0,
                (shape.x[1] - shape.x[0]) * scale,
                (shape.y[1] - shape.y[0]) * scale,
            ),
        )
    if isinstance(shape, Circle):
        outer = occ.addDisk(
            (shape.center[0] - x0) * scale,
            (shape.center[1] - y0) * scale,
            0.0,
            shape.radius * scale,
            shape.radius * scale,
        )
        if shape.inner_radius is None:
            return 2, outer
        inner = occ.addDisk(
            (shape.center[0] - x0) * scale,
            (shape.center[1] - y0) * scale,
            0.0,
            shape.inner_radius * scale,
            shape.inner_radius * scale,
        )
        cut, _ = occ.cut([(2, outer)], [(2, inner)], removeObject=True, removeTool=True)
        if len(cut) != 1:
            raise MeshError("Gmsh could not construct the annular material region.")
        return cut[0]
    if isinstance(shape, Polygon):
        points = [occ.addPoint((x - x0) * scale, (y - y0) * scale, 0.0) for x, y in shape.points]
        lines = [occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
        return 2, occ.addPlaneSurface([occ.addCurveLoop(lines)])
    raise MeshError(f"Unsupported Gmsh shape {type(shape).__name__}.")


def _edge_extrema(points: NDArray[np.float64], triangles: NDArray[np.int64]) -> tuple[float, float]:
    pairs = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    lengths = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    return float(lengths.min()), float(lengths.max())


def _material_wavenumber_scale(material: object) -> float:
    """Return a conservative local ``k/k0`` proxy for a diagonal material."""

    epsilon = np.asarray(material.eps_r, dtype=np.complex128)  # type: ignore[attr-defined]
    permeability = np.asarray(material.mu_r, dtype=np.complex128)  # type: ignore[attr-defined]
    return float(
        np.sqrt(np.max(np.abs(epsilon)) * np.max(np.abs(permeability)))
    )


def _surface_curves(
    gmsh: object,
    surfaces: set[int],
    solve_surfaces: set[int],
) -> list[int]:
    """Find curves on the boundary of a selected set of solve surfaces."""

    curves: list[int] = []
    for _, curve in gmsh.model.getEntities(1):
        upward, _ = gmsh.model.getAdjacencies(1, curve)
        adjacent = {int(entity) for entity in upward if int(entity) in solve_surfaces}
        if adjacent & surfaces and (adjacent - surfaces or len(adjacent) == 1):
            curves.append(int(curve))
    return curves


def _facet_matches_shape(
    shape: object,
    endpoints: NDArray[np.float64],
    midpoint: NDArray[np.float64],
    tolerance: float,
) -> bool:
    """Return whether a mesh boundary facet belongs to a placed solid.

    Testing only the chord midpoint is insufficient for the inner wall of an
    annulus: the midpoint lies just inside the inner disk and therefore outside
    the annular solid.  OCC mesh nodes on a circular curve retain the exact
    radius, so endpoint radii give a stable classification for both walls.
    """

    if bool(shape.contains(midpoint[0], midpoint[1])):  # type: ignore[union-attr]
        return True
    if isinstance(shape, Rectangle):
        return bool(
            shape.x[0] - tolerance <= midpoint[0] <= shape.x[1] + tolerance
            and shape.y[0] - tolerance <= midpoint[1] <= shape.y[1] + tolerance
        )
    if isinstance(shape, Circle):
        center = np.asarray(shape.center, dtype=float)
        radii = np.linalg.norm(endpoints - center, axis=1)
        targets = [shape.radius]
        if shape.inner_radius is not None:
            targets.append(shape.inner_radius)
        return any(np.all(np.abs(radii - radius) <= tolerance) for radius in targets)
    if isinstance(shape, Polygon):
        point = midpoint
        vertices = np.asarray(shape.points, dtype=float)
        for index in range(len(vertices)):
            start = vertices[index - 1]
            end = vertices[index]
            delta = end - start
            fraction = float(
                np.clip(np.dot(point - start, delta) / np.dot(delta, delta), 0.0, 1.0)
            )
            if np.linalg.norm(point - (start + fraction * delta)) <= tolerance:
                return True
    return False


def discretize_2d(
    geometry: GeometryModel2D,
    *,
    max_element_size: float | None = None,
    resolution: tuple[int, int] | None = None,
    element_order: int = 1,
    material_aware: bool = True,
    vacuum_wavenumber: float | None = None,
    wavelength_elements: int = 10,
    interface_refinement: float | None = None,
    interface_refinement_width: float | None = None,
    boundary_refinement: float | None = 0.5,
    boundary_refinement_width: float | None = None,
    _refinement_scale: float = 1.0,
) -> FEMMesh2D:
    """Generate a conforming, optionally material-aware Gmsh mesh.

    ``max_element_size`` is a global Gmsh characteristic-size target.  With
    ``material_aware=True`` each material receives a smaller local target in
    proportion to its propagation-wavenumber proxy
    ``sqrt(max(abs(epsilon_i)) * max(abs(mu_i)))``.  An optional
    ``interface_refinement`` in ``(0, 1]`` further scales the edge target near
    material jumps.  Refinement controls stored on ``geometry`` are applied
    without changing material or boundary provenance.
    """

    if isinstance(element_order, (bool, np.bool_)) or element_order not in (1, 2):
        raise MeshError("The 2D backend supports element_order=1 or 2.")
    width = geometry.x_span[1] - geometry.x_span[0]
    height = geometry.y_span[1] - geometry.y_span[0]
    if isinstance(max_element_size, (bool, np.bool_)):
        raise MeshError("max_element_size must be finite and positive.")
    if resolution is not None:
        if isinstance(resolution, (str, bytes)):
            raise MeshError("resolution must be a two-entry integer tuple.")
        try:
            entries = tuple(resolution)
        except TypeError as exc:
            raise MeshError("resolution must be a two-entry integer tuple.") from exc
        if len(entries) != 2:
            raise MeshError("resolution must be a two-entry integer tuple.")
        parsed: list[int] = []
        for value in entries:
            if (
                isinstance(value, (bool, np.bool_, str, bytes))
                or not np.isscalar(value)
            ):
                raise MeshError("resolution entries must be finite integers.")
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise MeshError("resolution entries must be finite integers.") from exc
            if not np.isfinite(numeric) or numeric != int(numeric):
                raise MeshError("resolution entries must be finite integers.")
            parsed.append(int(numeric))
        nx, ny = parsed
        if nx < 2 or ny < 2:
            raise MeshError("both resolution entries must be at least two.")
        resolution_size = min(width / nx, height / ny)
        max_element_size = (
            resolution_size
            if max_element_size is None
            else min(float(max_element_size), resolution_size)
        )
    if max_element_size is None:
        max_element_size = min(width, height) / 24.0
    maximum = float(max_element_size)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise MeshError("max_element_size must be finite and positive.")
    if not isinstance(material_aware, (bool, np.bool_)):
        raise MeshError("material_aware must be a boolean.")
    if (
        isinstance(wavelength_elements, (bool, np.bool_))
        or int(wavelength_elements) != wavelength_elements
        or wavelength_elements < 4
    ):
        raise MeshError("wavelength_elements must be an integer of at least four.")
    wavelength_count = int(wavelength_elements)
    if vacuum_wavenumber is not None:
        k0 = float(vacuum_wavenumber)
        if not np.isfinite(k0) or k0 <= 0.0:
            raise MeshError("vacuum_wavenumber must be finite and positive.")
    else:
        k0 = None
    if interface_refinement is not None:
        if isinstance(interface_refinement, (bool, np.bool_)):
            raise MeshError("interface_refinement must be in (0, 1] or None.")
        interface_factor = float(interface_refinement)
        if not np.isfinite(interface_factor) or not 0.0 < interface_factor <= 1.0:
            raise MeshError("interface_refinement must be in (0, 1] or None.")
    else:
        interface_factor = None
    if interface_refinement_width is not None:
        if isinstance(interface_refinement_width, (bool, np.bool_)):
            raise MeshError("interface_refinement_width must be finite and positive.")
        interface_width = float(interface_refinement_width)
        if not np.isfinite(interface_width) or interface_width <= 0.0:
            raise MeshError("interface_refinement_width must be finite and positive.")
        if interface_factor is None:
            raise MeshError(
                "interface_refinement_width requires interface_refinement."
            )
    else:
        interface_width = None
    if boundary_refinement is not None:
        if isinstance(boundary_refinement, (bool, np.bool_)):
            raise MeshError("boundary_refinement must be in (0, 1] or None.")
        boundary_factor = float(boundary_refinement)
        if not np.isfinite(boundary_factor) or not 0.0 < boundary_factor <= 1.0:
            raise MeshError("boundary_refinement must be in (0, 1] or None.")
    else:
        boundary_factor = None
    if boundary_refinement_width is not None:
        if isinstance(boundary_refinement_width, (bool, np.bool_)):
            raise MeshError("boundary_refinement_width must be finite and positive.")
        boundary_width = float(boundary_refinement_width)
        if not np.isfinite(boundary_width) or boundary_width <= 0.0:
            raise MeshError("boundary_refinement_width must be finite and positive.")
        if boundary_factor is None:
            raise MeshError(
                "boundary_refinement_width requires boundary_refinement."
            )
    else:
        boundary_width = None
    refinement_scale = float(_refinement_scale)
    if not np.isfinite(refinement_scale) or refinement_scale <= 0.0:
        raise MeshError("internal refinement scale must be finite and positive.")

    try:
        import gmsh
    except Exception as exc:  # pragma: no cover - installation dependent
        raise MeshError("Gmsh could not be imported; install the fem_waveguide_modes dependencies.") from exc

    xmin, xmax = geometry.x_span
    ymin, ymax = geometry.y_span
    scale = 1.0 / max(width, height)
    changed_option_names = (
        "General.Terminal",
        "Mesh.MeshSizeMax",
        "Mesh.MeshSizeMin",
        "Mesh.MeshSizeExtendFromBoundary",
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
        "Mesh.Algorithm",
    )
    with _GMSH_LOCK:
        owned = not bool(gmsh.isInitialized())
        if owned:
            gmsh.initialize()
        previous_options = (
            {}
            if owned
            else {
                name: float(gmsh.option.getNumber(name))
                for name in changed_option_names
            }
        )
        previous_model = "" if owned else str(gmsh.model.getCurrent())
        model_name = f"fem_modes_{uuid4().hex}"
        model_added = False
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(model_name)
            model_added = True
            occ = gmsh.model.occ

            # Rectangular partitions force material and PML interfaces into the
            # mesh before curved regions are fragmented into those cells.
            xcuts = {xmin, xmax}
            ycuts = {ymin, ymax}
            pml_x, pml_y = geometry.pml_interfaces()
            xcuts.update(pml_x)
            ycuts.update(pml_y)
            mesh_items = (
                *geometry.regions,
                *geometry.boundaries,
                *geometry.refinements,
            )
            for item in mesh_items:
                if isinstance(item.shape, Rectangle):
                    xcuts.update(item.shape.x)
                    ycuts.update(item.shape.y)
            xs = sorted(value for value in xcuts if xmin <= value <= xmax)
            ys = sorted(value for value in ycuts if ymin <= value <= ymax)
            base_cells: list[tuple[int, int]] = []
            for ix in range(len(xs) - 1):
                for iy in range(len(ys) - 1):
                    base_cells.append(
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
                    )
            curved_items = [
                item
                for item in mesh_items
                if not isinstance(item.shape, Rectangle)
            ]
            curved = [
                _add_occ_shape(gmsh, item.shape, (xmin, ymin), scale)
                for item in curved_items
            ]
            inputs = [*base_cells, *curved]
            if len(inputs) > 1:
                fragments, fragment_map = occ.fragment(
                    [inputs[0]], inputs[1:], removeObject=True, removeTool=True
                )
                surfaces = [item for item in fragments if item[0] == 2]
                if len(fragment_map) != len(inputs):
                    raise MeshError("Gmsh returned an incomplete OCC fragment provenance map.")
            else:
                surfaces = inputs
                fragment_map = [[inputs[0]]]
            domain_surfaces = {
                int(entity)
                for mapped in fragment_map[:len(base_cells)]
                for dimension, entity in mapped
                if dimension == 2
            }
            curved_surface_membership = {
                item.id: {
                    int(entity)
                    for dimension, entity in fragment_map[len(base_cells) + index]
                    if dimension == 2
                }
                for index, item in enumerate(curved_items)
            }
            occ.synchronize()

            solve_surfaces: list[int] = []
            excluded_surfaces: list[int] = []
            surface_material_tags: dict[int, int] = {}
            surface_centers: dict[int, tuple[float, float]] = {}
            physical_names = {1: "background"}
            for index, region in enumerate(geometry.regions, start=2):
                physical_names[index] = region.name

            for _, entity in gmsh.model.getEntities(2):
                if entity not in domain_surfaces:
                    excluded_surfaces.append(entity)
                    continue
                center = occ.getCenterOfMass(2, entity)
                px = center[0] / scale + xmin
                py = center[1] / scale + ymin
                surface_centers[entity] = (px, py)
                boundary_owners = [
                    boundary
                    for boundary in geometry.boundaries
                    if (
                        entity in curved_surface_membership[boundary.id]
                        if boundary.id in curved_surface_membership
                        else bool(boundary.shape.contains(px, py))  # type: ignore[union-attr]
                    )
                ]
                if len(boundary_owners) > 1:
                    names = ", ".join(
                        repr(boundary.name) for boundary in boundary_owners
                    )
                    raise MeshError(
                        "Overlapping conductor regions are ambiguous; merge them into "
                        f"one boundary object or make them merely adjacent: {names}."
                    )
                if boundary_owners:
                    excluded_surfaces.append(entity)
                    continue
                tag = 1
                for index, region in enumerate(geometry.regions, start=2):
                    contains_region = (
                        entity in curved_surface_membership[region.id]
                        if region.id in curved_surface_membership
                        else bool(region.shape.contains(px, py))  # type: ignore[union-attr]
                    )
                    if contains_region:
                        tag = index
                solve_surfaces.append(entity)
                surface_material_tags[entity] = tag

            if not solve_surfaces:
                raise MeshError("Perfect-conductor/impedance objects remove the complete solve domain.")
            if excluded_surfaces:
                occ.remove([(2, entity) for entity in excluded_surfaces], recursive=True)
                occ.synchronize()
            grouped: dict[int, list[int]] = {}
            for entity in solve_surfaces:
                grouped.setdefault(surface_material_tags[entity], []).append(entity)
            for tag, entities in grouped.items():
                gmsh.model.addPhysicalGroup(2, entities, tag)
                gmsh.model.setPhysicalName(2, tag, physical_names[tag])

            solve_surface_set = set(solve_surfaces)
            materials_by_tag = {
                1: geometry.background,
                **{
                    index: region.material
                    for index, region in enumerate(geometry.regions, start=2)
                },
            }
            local_wavenumbers = {
                tag: _material_wavenumber_scale(materials_by_tag[tag])
                for tag in grouped
            }
            if not np.isfinite(tuple(local_wavenumbers.values())).all():
                raise MeshError("The local material wavenumber estimate is not finite.")
            reference_wavenumber = max(
                min(local_wavenumbers.values(), default=1.0),
                np.finfo(float).tiny,
            )
            material_sizes: dict[int, float] = {}
            uniform_wavelength_target = maximum
            if k0 is not None:
                maximum_local_k = k0 * max(
                    max(local_wavenumbers.values(), default=1.0),
                    np.finfo(float).tiny,
                )
                uniform_wavelength_target = min(
                    maximum,
                    2.0 * np.pi / (maximum_local_k * wavelength_count),
                )
            for tag in grouped:
                if material_aware:
                    local_wavenumber = max(
                        local_wavenumbers[tag],
                        np.finfo(float).tiny,
                    )
                    local_target = min(
                        maximum,
                        maximum * reference_wavenumber / local_wavenumber,
                    )
                    if k0 is not None:
                        local_target = min(
                            local_target,
                            2.0
                            * np.pi
                            / (k0 * local_wavenumber * wavelength_count),
                        )
                    material_sizes[tag] = local_target
                else:
                    material_sizes[tag] = uniform_wavelength_target

            sizing_fields: list[int] = []
            target_sizes: list[float] = [maximum]

            def add_constant_field(entities: list[int], target: float) -> None:
                if not entities:
                    return
                field = gmsh.model.mesh.field.add("Constant")
                gmsh.model.mesh.field.setNumber(field, "VIn", target * scale)
                gmsh.model.mesh.field.setNumber(field, "VOut", maximum * scale)
                gmsh.model.mesh.field.setNumbers(field, "SurfacesList", entities)
                sizing_fields.append(field)
                target_sizes.append(target)

            if material_aware:
                for tag, entities in grouped.items():
                    target = material_sizes[tag]
                    if target < maximum * (1.0 - 32.0 * np.finfo(float).eps):
                        add_constant_field(entities, target)
            elif uniform_wavelength_target < maximum * (
                1.0 - 32.0 * np.finfo(float).eps
            ):
                add_constant_field(solve_surfaces, uniform_wavelength_target)

            # Explicit user sizing regions are pure meshing controls.  Their
            # OCC fragments are selected using exact provenance for curved
            # shapes and center membership for axis-aligned rectangles.
            refinement_surfaces: dict[int, set[int]] = {}
            for refinement in geometry.refinements:
                if refinement.id in curved_surface_membership:
                    selected = (
                        curved_surface_membership[refinement.id] & solve_surface_set
                    )
                else:
                    selected = {
                        entity
                        for entity in solve_surfaces
                        if bool(
                            refinement.shape.contains(*surface_centers[entity])
                        )
                    }
                refinement_surfaces[refinement.id] = selected
                target = min(
                    maximum,
                    refinement.max_element_size * refinement_scale,
                )
                add_constant_field(sorted(selected), target)
                if refinement.transition_width > 0.0 and selected:
                    curves = _surface_curves(gmsh, selected, solve_surface_set)
                    if curves:
                        distance = gmsh.model.mesh.field.add("Distance")
                        gmsh.model.mesh.field.setNumbers(
                            distance, "CurvesList", curves
                        )
                        gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)
                        threshold = gmsh.model.mesh.field.add("Threshold")
                        gmsh.model.mesh.field.setNumber(
                            threshold, "InField", distance
                        )
                        gmsh.model.mesh.field.setNumber(
                            threshold, "SizeMin", target * scale
                        )
                        gmsh.model.mesh.field.setNumber(
                            threshold, "SizeMax", maximum * scale
                        )
                        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
                        gmsh.model.mesh.field.setNumber(
                            threshold,
                            "DistMax",
                            refinement.transition_width * scale,
                        )
                        sizing_fields.append(threshold)

            # Refine only genuine material jumps, not artificial PML/base-cell
            # partitions.  Curves are grouped by target so each jump can use
            # the smaller of its two local material sizes.
            if interface_factor is not None:
                interface_curves: dict[float, list[int]] = {}
                for _, curve in gmsh.model.getEntities(1):
                    upward, _ = gmsh.model.getAdjacencies(1, curve)
                    adjacent = [
                        int(entity)
                        for entity in upward
                        if int(entity) in solve_surface_set
                    ]
                    tags = {surface_material_tags[entity] for entity in adjacent}
                    if len(tags) < 2:
                        continue
                    material_values = {
                        (
                            tuple(materials_by_tag[tag].eps_r),
                            tuple(materials_by_tag[tag].mu_r),
                        )
                        for tag in tags
                    }
                    if len(material_values) < 2:
                        continue
                    target = interface_factor * min(
                        material_sizes[tag] for tag in tags
                    )
                    interface_curves.setdefault(target, []).append(int(curve))

                for target, curves in interface_curves.items():
                    distance = gmsh.model.mesh.field.add("Distance")
                    gmsh.model.mesh.field.setNumbers(
                        distance, "CurvesList", curves
                    )
                    gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)
                    threshold = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
                    gmsh.model.mesh.field.setNumber(
                        threshold, "SizeMin", target * scale
                    )
                    gmsh.model.mesh.field.setNumber(
                        threshold, "SizeMax", maximum * scale
                    )
                    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
                    transition = (
                        interface_width
                        if interface_width is not None
                        else 2.0 * target
                    )
                    gmsh.model.mesh.field.setNumber(
                        threshold, "DistMax", transition * scale
                    )
                    sizing_fields.append(threshold)
                    target_sizes.append(target)

            # Every curve with exactly one adjacent solve surface is either an
            # outer PEC/PMC wall or an internal PEC/PMC/impedance hole.  A
            # distance field therefore refines all electromagnetic walls
            # without relying on fragile post-mesh facet classification.
            if boundary_factor is not None:
                boundary_curves: dict[float, list[int]] = {}
                for _, curve in gmsh.model.getEntities(1):
                    upward, _ = gmsh.model.getAdjacencies(1, curve)
                    adjacent = [
                        int(entity)
                        for entity in upward
                        if int(entity) in solve_surface_set
                    ]
                    if len(adjacent) != 1:
                        continue
                    material_tag = surface_material_tags[adjacent[0]]
                    target = boundary_factor * material_sizes[material_tag]
                    boundary_curves.setdefault(target, []).append(int(curve))

                for target, curves in boundary_curves.items():
                    distance = gmsh.model.mesh.field.add("Distance")
                    gmsh.model.mesh.field.setNumbers(
                        distance, "CurvesList", curves
                    )
                    gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)
                    threshold = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
                    gmsh.model.mesh.field.setNumber(
                        threshold, "SizeMin", target * scale
                    )
                    gmsh.model.mesh.field.setNumber(
                        threshold, "SizeMax", maximum * scale
                    )
                    transition = (
                        boundary_width
                        if boundary_width is not None
                        else 3.0 * target
                    )
                    fine_band = min(target, 0.5 * transition)
                    gmsh.model.mesh.field.setNumber(
                        threshold, "DistMin", fine_band * scale
                    )
                    gmsh.model.mesh.field.setNumber(
                        threshold, "DistMax", transition * scale
                    )
                    sizing_fields.append(threshold)
                    target_sizes.append(target)

            if len(sizing_fields) == 1:
                gmsh.model.mesh.field.setAsBackgroundMesh(sizing_fields[0])
            elif sizing_fields:
                combined = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(
                    combined, "FieldsList", sizing_fields
                )
                gmsh.model.mesh.field.setAsBackgroundMesh(combined)

            gmsh.option.setNumber("Mesh.MeshSizeMax", maximum * scale)
            gmsh.option.setNumber(
                "Mesh.MeshSizeMin",
                max(min(target_sizes) * scale * 0.05, 1e-12),
            )
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            # Gmsh's Delaunay algorithm handles the large gradients in
            # distance/threshold background fields more robustly than the
            # default Frontal-Delaunay algorithm.
            gmsh.option.setNumber("Mesh.Algorithm", 5)
            gmsh.model.mesh.generate(2)

            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            points = np.asarray(coordinates, dtype=float).reshape(-1, 3)[:, :2] / scale
            points += np.asarray((xmin, ymin))
            gmsh_element_tags, connectivity = gmsh.model.mesh.getElementsByType(2)
            if len(gmsh_element_tags) == 0:
                raise MeshError("Gmsh generated no first-order triangular elements.")
            lookup = {int(tag): index for index, tag in enumerate(node_tags)}
            raw = np.asarray(connectivity, dtype=np.int64).reshape(-1, 3)
            triangles = np.fromiter(
                (lookup[int(tag)] for tag in raw.ravel()),
                dtype=np.int64,
                count=raw.size,
            ).reshape(-1, 3)
            material_by_element: dict[int, int] = {}
            for entity, material_tag in surface_material_tags.items():
                _, tags_by_type, _ = gmsh.model.mesh.getElements(2, entity)
                for tags_for_entity in tags_by_type:
                    material_by_element.update(
                        (int(element), material_tag) for element in tags_for_entity
                    )
            try:
                material_tags = np.fromiter(
                    (material_by_element[int(element)] for element in gmsh_element_tags),
                    dtype=np.int32,
                    count=len(gmsh_element_tags),
                )
            except KeyError as exc:
                raise MeshError("Gmsh lost a material surface association during meshing.") from exc
        except MeshError:
            raise
        except Exception as exc:
            raise MeshError(f"Gmsh failed to generate the FEM mode mesh: {exc}") from exc
        finally:
            if owned:
                gmsh.finalize()
            else:
                if model_added:
                    try:
                        gmsh.model.setCurrent(model_name)
                        gmsh.model.remove()
                        if previous_model:
                            gmsh.model.setCurrent(previous_model)
                    except Exception:
                        pass
                for option_name, option_value in previous_options.items():
                    try:
                        gmsh.option.setNumber(option_name, option_value)
                    except Exception:
                        pass

    mesh = MeshTri(points.T, triangles.T)
    boundary_indices = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    facets = mesh.facets[:, boundary_indices].T
    midpoints = points[facets].mean(axis=1)
    kinds: dict[str, list[int]] = {"outer_pec": [], "outer_pmc": []}
    tolerance = 1e-8 * max(width, height)
    for local_index, facet_index in enumerate(boundary_indices):
        px, py = midpoints[local_index]
        external = (
            abs(px - xmin) <= tolerance
            or abs(px - xmax) <= tolerance
            or abs(py - ymin) <= tolerance
            or abs(py - ymax) <= tolerance
        )
        if external:
            kinds[f"outer_{geometry.outer_boundary}"].append(int(facet_index))
            continue
        matched = False
        for boundary in reversed(geometry.boundaries):
            if _facet_matches_shape(
                boundary.shape,
                points[facets[local_index]],
                midpoints[local_index],
                tolerance,
            ):
                kinds.setdefault(boundary.name, []).append(int(facet_index))
                kinds.setdefault(boundary.kind, []).append(int(facet_index))
                matched = True
                break
        if not matched:
            raise MeshError(
                "An internal boundary facet could not be associated with its "
                "PEC, PMC, or impedance geometry. Refine the geometry or report "
                f"the facet near ({px:.9g}, {py:.9g})."
            )

    hmin, hmax = _edge_extrema(points, triangles)
    return FEMMesh2D(
        mesh=mesh,
        element_tags=material_tags,
        physical_names=physical_names,
        boundary_facets={name: np.asarray(values, dtype=np.int64) for name, values in kinds.items()},
        info=MeshInfo(
            nodes=points.shape[0],
            elements=triangles.shape[0],
            minimum_edge=hmin,
            maximum_edge=hmax,
            requested_maximum_edge=maximum,
            element_order=element_order,
            material_aware=bool(material_aware),
            interface_refinement=interface_factor,
            boundary_refinement=boundary_factor,
            refinement_regions=len(geometry.refinements),
        ),
        geometry_revision=geometry.revision,
    )


__all__ = [
    "FEMMesh1D",
    "FEMMesh2D",
    "MeshInfo",
    "discretize_1d",
    "discretize_2d",
]

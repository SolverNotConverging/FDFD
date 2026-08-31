"""Conforming tetrahedral meshes for one cell periodic in ``z``."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from skfem import MeshTet

from .exceptions import MeshError
from .geometry import BoundaryRegion, Box, Cylinder, GeometryModel3D, Shape3D, Sphere


_GMSH_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class MeshInfo3D:
    nodes: int
    elements: int
    minimum_edge: float
    maximum_edge: float
    requested_maximum_edge: float
    element_order: int = 1
    material_aware: bool = True
    refinement_regions: int = 0


@dataclass(frozen=True, slots=True)
class PeriodicMesh3D:
    """A physical tetrahedral mesh and its periodic-topology metadata."""

    mesh: MeshTet
    element_tags: NDArray[np.int32]
    physical_names: Mapping[int, str]
    boundary_facets: Mapping[str, NDArray[np.int64]]
    periodic_node_pairs: NDArray[np.int64]
    periodic_edge_pairs: NDArray[np.int64]
    edge_nodes: NDArray[np.int64]
    cell_edges: NDArray[np.int64]
    cell_edge_signs: NDArray[np.int8]
    info: MeshInfo3D
    geometry_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "physical_names", MappingProxyType(dict(self.physical_names)))
        object.__setattr__(
            self,
            "boundary_facets",
            MappingProxyType(
                {
                    str(name): np.asarray(values, dtype=np.int64)
                    for name, values in self.boundary_facets.items()
                }
            ),
        )

    @property
    def nodes(self) -> NDArray[np.float64]:
        return np.asarray(self.mesh.p.T, dtype=np.float64)

    @property
    def elements(self) -> NDArray[np.int64]:
        return np.asarray(self.mesh.t.T, dtype=np.int64)


FEMPeriodicMesh3D = PeriodicMesh3D


def _material_scale(material: object) -> float:
    epsilon = np.asarray(getattr(material, "eps_r"), dtype=np.complex128)
    mu = np.asarray(getattr(material, "mu_r"), dtype=np.complex128)
    return float(np.sqrt(np.max(np.abs(epsilon)) * np.max(np.abs(mu))))


def _add_occ_shape(gmsh: object, shape: Shape3D, origin: tuple[float, float, float], scale: float) -> tuple[int, int]:
    occ = gmsh.model.occ
    x0, y0, z0 = origin
    if isinstance(shape, Box):
        return (
            3,
            occ.addBox(
                (shape.x[0] - x0) * scale,
                (shape.y[0] - y0) * scale,
                (shape.z[0] - z0) * scale,
                (shape.x[1] - shape.x[0]) * scale,
                (shape.y[1] - shape.y[0]) * scale,
                (shape.z[1] - shape.z[0]) * scale,
            ),
        )
    if isinstance(shape, Cylinder):
        return (
            3,
            occ.addCylinder(
                (shape.center[0] - x0) * scale,
                (shape.center[1] - y0) * scale,
                (shape.z[0] - z0) * scale,
                0.0,
                0.0,
                (shape.z[1] - shape.z[0]) * scale,
                shape.radius * scale,
            ),
        )
    if isinstance(shape, Sphere):
        return (
            3,
            occ.addSphere(
                (shape.center[0] - x0) * scale,
                (shape.center[1] - y0) * scale,
                (shape.center[2] - z0) * scale,
                shape.radius * scale,
            ),
        )
    raise MeshError(f"Unsupported three-dimensional OCC shape {type(shape).__name__}.")


def _edge_extrema(points: NDArray[np.float64], tetrahedra: NDArray[np.int64]) -> tuple[float, float]:
    local_edges = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
    lengths = np.concatenate(
        [
            np.linalg.norm(points[tetrahedra[:, first]] - points[tetrahedra[:, second]], axis=1)
            for first, second in local_edges
        ]
    )
    return float(np.min(lengths)), float(np.max(lengths))


def _cell_edge_signs(mesh: MeshTet) -> NDArray[np.int8]:
    local_edges = np.asarray(mesh.elem.refdom.edges, dtype=np.int64)
    signs = np.empty((mesh.nelements, local_edges.shape[0]), dtype=np.int8)
    cells = np.asarray(mesh.t.T, dtype=np.int64)
    for column, (first, second) in enumerate(local_edges):
        signs[:, column] = np.where(
            cells[:, first] < cells[:, second], 1, -1
        ).astype(np.int8)
    return signs


def _periodic_edge_pairs(
    edges: NDArray[np.int64], node_pairs: NDArray[np.int64]
) -> NDArray[np.int64]:
    node_map = {int(slave): int(master) for slave, master in node_pairs}
    edge_lookup = {tuple(int(value) for value in edge): index for index, edge in enumerate(edges)}
    result: list[tuple[int, int, int]] = []
    for slave_edge, (first, second) in enumerate(edges):
        if int(first) not in node_map or int(second) not in node_map:
            continue
        mapped = (node_map[int(first)], node_map[int(second)])
        key = tuple(sorted(mapped))
        try:
            master_edge = edge_lookup[key]
        except KeyError as exc:
            raise MeshError(
                "A periodic slave edge has no translated master edge."
            ) from exc
        sign = 1 if mapped == key else -1
        result.append((slave_edge, master_edge, sign))
    return np.asarray(result, dtype=np.int64).reshape(-1, 3)


def discretize_3d(
    geometry: GeometryModel3D,
    *,
    max_element_size: float | None = None,
    wavelength_elements: int = 8,
    material_aware: bool = True,
    element_order: int = 1,
    k0: float | None = None,
    _refinement_scale: float = 1.0,
) -> PeriodicMesh3D:
    """Generate a first-order tetrahedral mesh with matching ``z`` faces."""

    if not isinstance(geometry, GeometryModel3D):
        raise TypeError("geometry must be a GeometryModel3D instance.")
    if element_order != 1:
        raise MeshError("Periodic 3D v1 supports first-order tetrahedra only.")
    if isinstance(wavelength_elements, bool) or int(wavelength_elements) != wavelength_elements or wavelength_elements < 4:
        raise MeshError("wavelength_elements must be an integer of at least four.")
    xmin, xmax = geometry.x_span
    ymin, ymax = geometry.y_span
    zmin, zmax = geometry.z_span
    widths = (xmax - xmin, ymax - ymin, zmax - zmin)
    reference = max(widths)
    maximum = min(widths) / 8.0 if max_element_size is None else float(max_element_size)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise MeshError("max_element_size must be finite and positive.")
    refinement_scale = float(_refinement_scale)
    if not np.isfinite(refinement_scale) or refinement_scale <= 0.0:
        raise MeshError("internal refinement scale must be finite and positive.")
    if k0 is not None:
        if not np.isfinite(k0) or k0 <= 0.0:
            raise MeshError("k0 must be finite and positive.")
        # The global size belongs to the low-index background.  High-index
        # partitions are refined locally below instead of forcing the entire
        # cell to use their shortest wavelength.
        material_scales = (
            [_material_scale(geometry.background)] if material_aware else [1.0]
        )
        maximum = min(
            maximum,
            2.0 * np.pi / (float(k0) * max(material_scales) * int(wavelength_elements)),
        )
    maximum *= refinement_scale

    try:
        import gmsh
    except Exception as exc:  # pragma: no cover - installation dependent
        raise MeshError("Gmsh could not be imported for periodic 3D meshing.") from exc

    scale = 1.0 / reference
    changed_options = (
        "General.Terminal",
        "Mesh.MeshSizeMax",
        "Mesh.MeshSizeMin",
        "Mesh.MeshSizeExtendFromBoundary",
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
    )
    with _GMSH_LOCK:
        owned = not bool(gmsh.isInitialized())
        if owned:
            gmsh.initialize()
        previous_options = (
            {}
            if owned
            else {name: float(gmsh.option.getNumber(name)) for name in changed_options}
        )
        previous_model = "" if owned else str(gmsh.model.getCurrent())
        model_name = f"fem_periodic_3d_{uuid4().hex}"
        model_added = False
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(model_name)
            model_added = True
            occ = gmsh.model.occ

            xcuts = {xmin, xmax}
            ycuts = {ymin, ymax}
            for pml in geometry.pmls:
                sides = geometry._pml_sides(pml.direction)
                if "x-" in sides:
                    xcuts.add(xmin + pml.thickness)
                if "x+" in sides:
                    xcuts.add(xmax - pml.thickness)
                if "y-" in sides:
                    ycuts.add(ymin + pml.thickness)
                if "y+" in sides:
                    ycuts.add(ymax - pml.thickness)
            xs = sorted(xcuts)
            ys = sorted(ycuts)
            base_cells: list[tuple[int, int]] = []
            for ix in range(len(xs) - 1):
                for iy in range(len(ys) - 1):
                    base_cells.append(
                        (
                            3,
                            occ.addBox(
                                (xs[ix] - xmin) * scale,
                                (ys[iy] - ymin) * scale,
                                0.0,
                                (xs[ix + 1] - xs[ix]) * scale,
                                (ys[iy + 1] - ys[iy]) * scale,
                                (zmax - zmin) * scale,
                            ),
                        )
                    )
            scene_items = (*geometry.regions, *geometry.boundaries, *geometry.refinements)
            tools = [
                _add_occ_shape(gmsh, item.shape, (xmin, ymin, zmin), scale)
                for item in scene_items
            ]
            inputs = [*base_cells, *tools]
            if len(inputs) > 1:
                fragments, fragment_map = occ.fragment(
                    [inputs[0]], inputs[1:], removeObject=True, removeTool=True
                )
                volumes = [item for item in fragments if item[0] == 3]
                if len(fragment_map) != len(inputs):
                    raise MeshError("Gmsh returned an incomplete fragment provenance map.")
            else:
                volumes = inputs
                fragment_map = [[inputs[0]]]
            memberships = {
                item.id: {
                    int(entity)
                    for dimension, entity in fragment_map[len(base_cells) + index]
                    if dimension == 3
                }
                for index, item in enumerate(scene_items)
            }
            occ.synchronize()

            # Preserve OCC ownership of conductor surfaces before their volume
            # descendants are removed.  Using this provenance is essential for
            # curved cylinders and spheres: the centroid of a flat mesh facet
            # does not lie on the corresponding analytic surface.
            conductor_surfaces: dict[int, set[int]] = {}
            for boundary in geometry.boundaries:
                descendants = sorted(memberships[boundary.id])
                entities = gmsh.model.getBoundary(
                    [(3, entity) for entity in descendants],
                    combined=True,
                    oriented=False,
                    recursive=False,
                )
                conductor_surfaces[boundary.id] = {
                    int(entity) for dimension, entity in entities if dimension == 2
                }

            physical_names = {1: "background"}
            for tag, region in enumerate(geometry.regions, start=2):
                physical_names[tag] = region.name
            solve_volumes: list[int] = []
            excluded_volumes: list[int] = []
            volume_material_tags: dict[int, int] = {}
            for _, entity in gmsh.model.getEntities(3):
                owners = [
                    boundary
                    for boundary in geometry.boundaries
                    if entity in memberships[boundary.id]
                ]
                if len(owners) > 1:
                    raise MeshError("Overlapping PEC/PMC volumes are ambiguous.")
                if owners:
                    excluded_volumes.append(int(entity))
                    continue
                material_tag = 1
                for tag, region in enumerate(geometry.regions, start=2):
                    if entity in memberships[region.id]:
                        material_tag = tag
                solve_volumes.append(int(entity))
                volume_material_tags[int(entity)] = material_tag
            if not solve_volumes:
                raise MeshError("PEC/PMC objects remove the complete solve domain.")
            if excluded_volumes:
                occ.remove([(3, entity) for entity in excluded_volumes], recursive=True)
                occ.synchronize()
            solve_set = set(solve_volumes)
            grouped: dict[int, list[int]] = {}
            for entity in solve_volumes:
                grouped.setdefault(volume_material_tags[entity], []).append(entity)
            for tag, entities in grouped.items():
                gmsh.model.addPhysicalGroup(3, entities, tag)
                gmsh.model.setPhysicalName(3, tag, physical_names[tag])

            boundary_surfaces: list[int] = []
            for _, surface in gmsh.model.getEntities(2):
                upward, _ = gmsh.model.getAdjacencies(2, surface)
                adjacent = [int(value) for value in upward if int(value) in solve_set]
                if len(adjacent) == 1:
                    boundary_surfaces.append(int(surface))
            boundary_surface_set = set(boundary_surfaces)
            conductor_surface_owners: dict[int, BoundaryRegion] = {}
            for boundary in geometry.boundaries:
                for surface in conductor_surfaces[boundary.id] & boundary_surface_set:
                    previous = conductor_surface_owners.setdefault(surface, boundary)
                    if previous is not boundary:
                        raise MeshError("A conductor surface belongs to multiple boundary objects.")
            scaled_period = (zmax - zmin) * scale
            coordinate_tolerance = 2.0e-6
            master_surfaces: list[int] = []
            slave_surfaces: list[int] = []
            for surface in boundary_surfaces:
                bounds = gmsh.model.getBoundingBox(2, surface)
                if abs(bounds[2]) <= coordinate_tolerance and abs(bounds[5]) <= coordinate_tolerance:
                    master_surfaces.append(surface)
                elif abs(bounds[2] - scaled_period) <= coordinate_tolerance and abs(bounds[5] - scaled_period) <= coordinate_tolerance:
                    slave_surfaces.append(surface)
            if not master_surfaces or len(master_surfaces) != len(slave_surfaces):
                raise MeshError("The two periodic z faces do not have matching topology.")

            unmatched = set(slave_surfaces)
            periodic_surface_pairs: list[tuple[int, int]] = []
            for master in master_surfaces:
                mb = gmsh.model.getBoundingBox(2, master)
                descriptor = np.asarray((mb[0], mb[1], mb[3], mb[4]))
                matches = []
                for slave in unmatched:
                    sb = gmsh.model.getBoundingBox(2, slave)
                    other = np.asarray((sb[0], sb[1], sb[3], sb[4]))
                    if np.allclose(descriptor, other, rtol=0.0, atol=coordinate_tolerance):
                        matches.append(slave)
                if len(matches) != 1:
                    raise MeshError("Could not pair periodic z-face fragments bijectively.")
                slave = matches[0]
                unmatched.remove(slave)
                periodic_surface_pairs.append((slave, master))
                gmsh.model.mesh.setPeriodic(
                    2,
                    [slave],
                    [master],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, scaled_period, 0, 0, 0, 1],
                )
            if unmatched:
                raise MeshError("One or more periodic slave surfaces are unpaired.")

            surface_groups: dict[str, set[int]] = {
                "periodic_master": set(master_surfaces),
                "periodic_slave": set(slave_surfaces),
                "outer_pec": set(),
                "outer_pmc": set(),
            }
            for surface in boundary_surfaces:
                owner = conductor_surface_owners.get(surface)
                if owner is not None:
                    surface_groups.setdefault(owner.kind, set()).add(surface)
                    surface_groups.setdefault(owner.name, set()).add(surface)
                    continue
                if surface in master_surfaces or surface in slave_surfaces:
                    continue
                bounds = gmsh.model.getBoundingBox(2, surface)
                on_outer_transverse_wall = (
                    (abs(bounds[0]) <= coordinate_tolerance and abs(bounds[3]) <= coordinate_tolerance)
                    or (
                        abs(bounds[0] - widths[0] * scale) <= coordinate_tolerance
                        and abs(bounds[3] - widths[0] * scale) <= coordinate_tolerance
                    )
                    or (abs(bounds[1]) <= coordinate_tolerance and abs(bounds[4]) <= coordinate_tolerance)
                    or (
                        abs(bounds[1] - widths[1] * scale) <= coordinate_tolerance
                        and abs(bounds[4] - widths[1] * scale) <= coordinate_tolerance
                    )
                )
                if not on_outer_transverse_wall:
                    raise MeshError(
                        f"Boundary surface {surface} has no OCC conductor owner."
                    )
                surface_groups[f"outer_{geometry.outer_boundary}"].add(surface)

            target_sizes = [maximum]
            sizing_fields: list[int] = []
            background_scale = max(
                _material_scale(geometry.background), np.finfo(float).tiny
            )
            if material_aware:
                for region in geometry.regions:
                    selected = sorted(memberships[region.id] & solve_set)
                    if not selected:
                        continue
                    ratio = _material_scale(region.material) / background_scale
                    target = maximum / max(1.0, ratio)
                    target = max(target, maximum * 0.1)
                    if target >= maximum * (1.0 - 1e-12):
                        continue
                    field = gmsh.model.mesh.field.add("Constant")
                    gmsh.model.mesh.field.setNumber(field, "VIn", target * scale)
                    gmsh.model.mesh.field.setNumber(field, "VOut", maximum * scale)
                    gmsh.model.mesh.field.setNumbers(field, "VolumesList", selected)
                    sizing_fields.append(field)
                    target_sizes.append(target)
            for refinement in geometry.refinements:
                selected = sorted(memberships[refinement.id] & solve_set)
                if not selected:
                    continue
                target = min(maximum, refinement.max_element_size * refinement_scale)
                field = gmsh.model.mesh.field.add("Constant")
                gmsh.model.mesh.field.setNumber(field, "VIn", target * scale)
                gmsh.model.mesh.field.setNumber(field, "VOut", maximum * scale)
                gmsh.model.mesh.field.setNumbers(field, "VolumesList", selected)
                sizing_fields.append(field)
                target_sizes.append(target)

            # Internal PEC objects create geometric edges and current
            # crowding that require local resolution.  Uniform outer guide
            # walls are already mesh-conforming and refining all six faces
            # would turn the entire volume into a boundary layer.
            pec_surface_groups = (
                (sorted(set(surface_groups.get("pec", set()))), 0.35),
            )
            for pec_surfaces, refinement_factor in pec_surface_groups:
                if not pec_surfaces:
                    continue
                distance = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(
                    distance, "SurfacesList", pec_surfaces
                )
                gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)
                threshold = gmsh.model.mesh.field.add("Threshold")
                pec_target = maximum * refinement_factor
                gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
                gmsh.model.mesh.field.setNumber(
                    threshold, "SizeMin", pec_target * scale
                )
                gmsh.model.mesh.field.setNumber(
                    threshold, "SizeMax", maximum * scale
                )
                gmsh.model.mesh.field.setNumber(
                    threshold, "DistMin", 0.25 * maximum * scale
                )
                gmsh.model.mesh.field.setNumber(
                    threshold, "DistMax", 2.0 * maximum * scale
                )
                sizing_fields.append(threshold)
                target_sizes.append(pec_target)
            if len(sizing_fields) == 1:
                gmsh.model.mesh.field.setAsBackgroundMesh(sizing_fields[0])
            elif sizing_fields:
                combined = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(combined, "FieldsList", sizing_fields)
                gmsh.model.mesh.field.setAsBackgroundMesh(combined)

            gmsh.option.setNumber("Mesh.MeshSizeMax", maximum * scale)
            gmsh.option.setNumber(
                "Mesh.MeshSizeMin", max(min(target_sizes) * scale * 0.05, 1e-12)
            )
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), maximum * scale)
            gmsh.model.mesh.generate(3)

            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            raw_points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
            points = raw_points / scale + np.asarray((xmin, ymin, zmin))
            node_lookup = {int(tag): index for index, tag in enumerate(node_tags)}
            element_tags, connectivity = gmsh.model.mesh.getElementsByType(4)
            if len(element_tags) == 0:
                raise MeshError("Gmsh generated no first-order tetrahedra.")
            raw_tetrahedra = np.asarray(connectivity, dtype=np.int64).reshape(-1, 4)
            tetrahedra = np.fromiter(
                (node_lookup[int(tag)] for tag in raw_tetrahedra.ravel()),
                dtype=np.int64,
                count=raw_tetrahedra.size,
            ).reshape(-1, 4)

            material_by_element: dict[int, int] = {}
            for entity, material_tag in volume_material_tags.items():
                if entity not in solve_set:
                    continue
                _, tags_by_type, _ = gmsh.model.mesh.getElements(3, entity)
                for tags in tags_by_type:
                    material_by_element.update(
                        (int(element), material_tag) for element in tags
                    )
            try:
                material_tags = np.fromiter(
                    (material_by_element[int(element)] for element in element_tags),
                    dtype=np.int32,
                    count=len(element_tags),
                )
            except KeyError as exc:
                raise MeshError("Gmsh lost a material-volume association.") from exc

            pair_map: dict[int, int] = {}
            for slave_surface, master_surface in periodic_surface_pairs:
                returned_master, slave_tags, master_tags, _ = gmsh.model.mesh.getPeriodicNodes(
                    2, slave_surface, includeHighOrderNodes=False
                )
                if int(returned_master) != int(master_surface):
                    raise MeshError("Gmsh returned the wrong periodic master surface.")
                for slave_tag, master_tag in zip(slave_tags, master_tags, strict=True):
                    slave_node = node_lookup[int(slave_tag)]
                    master_node = node_lookup[int(master_tag)]
                    previous = pair_map.setdefault(slave_node, master_node)
                    if previous != master_node:
                        raise MeshError("A periodic slave node maps to multiple masters.")
            periodic_node_pairs = np.asarray(
                sorted(pair_map.items()), dtype=np.int64
            ).reshape(-1, 2)
            z_slave_nodes = np.flatnonzero(
                np.isclose(points[:, 2], zmax, rtol=0.0, atol=1e-8 * reference)
            )
            if set(int(value) for value in z_slave_nodes) != set(pair_map):
                raise MeshError("Gmsh periodic mapping does not cover the complete z+ face.")
            if periodic_node_pairs.size:
                translated = points[periodic_node_pairs[:, 0]] - np.asarray((0.0, 0.0, zmax - zmin))
                error = np.max(
                    np.abs(translated - points[periodic_node_pairs[:, 1]])
                )
                if error > 1e-8 * reference:
                    raise MeshError("Periodic node coordinates fail the translation check.")

            boundary_node_facets: dict[str, list[tuple[int, int, int]]] = {}
            for name, surfaces in surface_groups.items():
                facets: list[tuple[int, int, int]] = []
                for surface in sorted(surfaces):
                    _, connectivity = gmsh.model.mesh.getElementsByType(2, int(surface))
                    raw_facets = np.asarray(connectivity, dtype=np.int64)
                    if raw_facets.size == 0:
                        continue
                    for node_triplet in raw_facets.reshape(-1, 3):
                        try:
                            facet = tuple(
                                sorted(node_lookup[int(node)] for node in node_triplet)
                            )
                        except KeyError as exc:
                            raise MeshError(
                                f"Boundary surface {surface} references an unused mesh node."
                            ) from exc
                        facets.append(facet)  # type: ignore[arg-type]
                boundary_node_facets[name] = facets
        except MeshError:
            raise
        except Exception as exc:
            raise MeshError(f"Gmsh failed to generate the periodic 3D mesh: {exc}") from exc
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
                for name, value in previous_options.items():
                    try:
                        gmsh.option.setNumber(name, value)
                    except Exception:
                        pass

    mesh = MeshTet(points.T, tetrahedra.T)
    edge_nodes = np.asarray(mesh.edges.T, dtype=np.int64)
    edge_pairs = _periodic_edge_pairs(edge_nodes, periodic_node_pairs)
    boundary_indices = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    facet_lookup = {
        tuple(sorted(int(node) for node in mesh.facets[:, facet_index])): int(facet_index)
        for facet_index in boundary_indices
    }
    boundary_kinds: dict[str, list[int]] = {}
    assigned: set[int] = set()
    for name, facets in boundary_node_facets.items():
        selected: list[int] = []
        for facet in facets:
            try:
                facet_index = facet_lookup[facet]
            except KeyError as exc:
                raise MeshError(
                    f"Gmsh boundary group {name!r} contains a non-boundary tetrahedron facet."
                ) from exc
            selected.append(facet_index)
            assigned.add(facet_index)
        boundary_kinds[name] = sorted(set(selected))
    missing = set(int(value) for value in boundary_indices) - assigned
    if missing:
        raise MeshError(
            f"{len(missing)} tetrahedron boundary facet(s) have no Gmsh OCC owner."
        )

    hmin, hmax = _edge_extrema(points, tetrahedra)
    return PeriodicMesh3D(
        mesh=mesh,
        element_tags=np.asarray(material_tags, dtype=np.int32),
        physical_names=physical_names,
        boundary_facets={
            name: np.asarray(values, dtype=np.int64)
            for name, values in boundary_kinds.items()
        },
        periodic_node_pairs=np.asarray(periodic_node_pairs, dtype=np.int64),
        periodic_edge_pairs=edge_pairs,
        edge_nodes=edge_nodes,
        cell_edges=np.asarray(mesh.t2e.T, dtype=np.int64),
        cell_edge_signs=_cell_edge_signs(mesh),
        info=MeshInfo3D(
            nodes=points.shape[0],
            elements=tetrahedra.shape[0],
            minimum_edge=hmin,
            maximum_edge=hmax,
            requested_maximum_edge=maximum,
            material_aware=bool(material_aware),
            refinement_regions=len(geometry.refinements),
        ),
        geometry_revision=geometry.revision,
    )


__all__ = [
    "FEMPeriodicMesh3D",
    "MeshInfo3D",
    "PeriodicMesh3D",
    "discretize_3d",
]

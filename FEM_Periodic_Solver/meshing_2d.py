"""Gmsh construction of conforming triangular periodic cells."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from skfem import MeshTri

from .exceptions import ConfigurationError, MeshError
from .geometry import Circle, GeometryModel2D, Polygon, Rectangle


@dataclass(frozen=True, slots=True)
class MeshInfo:
    nodes: int
    elements: int
    minimum_edge: float
    maximum_edge: float
    requested_maximum_edge: float
    element_order: int = 1


@dataclass(frozen=True, slots=True)
class FEMPeriodicMesh2D:
    """Physical mesh plus exact periodic and boundary associations."""

    mesh: MeshTri
    element_tags: NDArray[np.int32]
    physical_names: dict[int, str]
    boundary_facets: dict[str, NDArray[np.int64]]
    slave_nodes: NDArray[np.int64]
    master_nodes: NDArray[np.int64]
    info: MeshInfo
    geometry_revision: int

    @property
    def nodes(self) -> NDArray[np.float64]:
        return np.asarray(self.mesh.p.T, dtype=np.float64)

    @property
    def elements(self) -> NDArray[np.int64]:
        return np.asarray(self.mesh.t.T, dtype=np.int64)

    @property
    def period(self) -> float:
        top = self.nodes[self.slave_nodes, 1]
        bottom = self.nodes[self.master_nodes, 1]
        return float(np.median(top - bottom))


def _material_scale(material: object) -> float:
    """Return the largest local relative wavenumber scale ``|sqrt(eps*mu)|``."""

    epsilon = np.asarray(getattr(material, "eps_r"), dtype=np.complex128)
    mu = np.asarray(getattr(material, "mu_r"), dtype=np.complex128)
    return float(np.sqrt(np.max(np.abs(epsilon * mu))))


def _add_occ_shape(gmsh: object, shape: object, origin: tuple[float, float], scale: float) -> tuple[int, int]:
    occ = gmsh.model.occ
    x0, z0 = origin
    if isinstance(shape, Rectangle):
        tag = occ.addRectangle(
            (shape.x[0] - x0) * scale,
            (shape.z[0] - z0) * scale,
            0.0,
            (shape.x[1] - shape.x[0]) * scale,
            (shape.z[1] - shape.z[0]) * scale,
        )
        return 2, int(tag)
    if isinstance(shape, Circle):
        outer = occ.addDisk(
            (shape.center[0] - x0) * scale,
            (shape.center[1] - z0) * scale,
            0.0,
            shape.radius * scale,
            shape.radius * scale,
        )
        if shape.inner_radius is None:
            return 2, int(outer)
        inner = occ.addDisk(
            (shape.center[0] - x0) * scale,
            (shape.center[1] - z0) * scale,
            0.0,
            shape.inner_radius * scale,
            shape.inner_radius * scale,
        )
        result, _ = occ.cut([(2, outer)], [(2, inner)], removeObject=True, removeTool=True)
        surfaces = [tag for dimension, tag in result if dimension == 2]
        if len(surfaces) != 1:
            raise MeshError("Gmsh could not construct the annular region.")
        return 2, int(surfaces[0])
    if isinstance(shape, Polygon):
        points = [
            occ.addPoint((x - x0) * scale, (z - z0) * scale, 0.0)
            for x, z in shape.points
        ]
        lines = [
            occ.addLine(points[index], points[(index + 1) % len(points)])
            for index in range(len(points))
        ]
        loop = occ.addCurveLoop(lines)
        return 2, int(occ.addPlaneSurface([loop]))
    raise MeshError(f"Unsupported 2D OCC shape {type(shape).__name__}.")


def _edge_extrema(points: NDArray[np.float64], triangles: NDArray[np.int64]) -> tuple[float, float]:
    edges = np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]))
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    return float(lengths.min()), float(lengths.max())


def _boundary_curves(gmsh: object, surfaces: list[int]) -> set[int]:
    if not surfaces:
        return set()
    values = gmsh.model.getBoundary(
        [(2, int(surface)) for surface in surfaces],
        combined=True,
        oriented=False,
        recursive=False,
    )
    return {int(tag) for dimension, tag in values if dimension == 1}


def _pair_periodic_curves(
    gmsh: object,
    solve_surfaces: list[int],
    *,
    width: float,
    period: float,
    scale: float,
) -> list[tuple[int, int]]:
    # OpenCASCADE bounding boxes include a small model-space padding (usually
    # about 1e-7), so entity classification needs a looser tolerance than the
    # eventual node-coordinate validation.
    tolerance = 2e-6 * max(width, period) * scale
    boundary = _boundary_curves(gmsh, solve_surfaces)
    bottom: list[tuple[float, float, int]] = []
    top: list[tuple[float, float, int]] = []
    for curve in boundary:
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, curve)
        if abs(ymin) <= tolerance and abs(ymax) <= tolerance:
            bottom.append((float(xmin), float(xmax), curve))
        elif abs(ymin - period * scale) <= tolerance and abs(ymax - period * scale) <= tolerance:
            top.append((float(xmin), float(xmax), curve))
    bottom.sort()
    top.sort()
    if not bottom or len(bottom) != len(top):
        raise MeshError(
            "The z-min and z-max boundaries do not have matching topology; "
            "split seam-crossing objects into matching pieces."
        )
    pairs: list[tuple[int, int]] = []
    affine = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, period * scale,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    for (master_min, master_max, master), (slave_min, slave_max, slave) in zip(bottom, top, strict=True):
        if abs(master_min - slave_min) > tolerance or abs(master_max - slave_max) > tolerance:
            raise MeshError("Periodic boundary segments do not match after z translation.")
        gmsh.model.mesh.setPeriodic(1, [slave], [master], affine)
        pairs.append((slave, master))
    return pairs


def _facet_groups(
    mesh: MeshTri,
    lookup: dict[int, int],
    gmsh: object,
    curve_groups: dict[str, set[int]],
) -> dict[str, NDArray[np.int64]]:
    facets = np.asarray(mesh.facets.T, dtype=np.int64)
    facet_lookup = {tuple(sorted(map(int, edge))): index for index, edge in enumerate(facets)}
    output: dict[str, NDArray[np.int64]] = {}
    for name, curves in curve_groups.items():
        selected: set[int] = set()
        for curve in curves:
            _, connectivity = gmsh.model.mesh.getElementsByType(1, int(curve))
            raw = np.asarray(connectivity, dtype=np.int64)
            if raw.size == 0:
                continue
            for first_tag, second_tag in raw.reshape(-1, 2):
                if int(first_tag) not in lookup or int(second_tag) not in lookup:
                    continue
                key = tuple(sorted((lookup[int(first_tag)], lookup[int(second_tag)])))
                try:
                    selected.add(facet_lookup[key])
                except KeyError as exc:
                    raise MeshError(f"Gmsh boundary curve {curve} is not a triangle facet.") from exc
        values = np.asarray(sorted(selected), dtype=np.int64)
        values.setflags(write=False)
        output[name] = values
    return output


def discretize_periodic_2d(
    geometry: GeometryModel2D,
    *,
    max_element_size: float,
    element_order: int = 1,
) -> FEMPeriodicMesh2D:
    """Create a first-order triangular mesh with a Gmsh periodic node map."""

    if not isinstance(geometry, GeometryModel2D):
        raise TypeError("geometry must be a GeometryModel2D instance.")
    maximum = float(max_element_size)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ConfigurationError("max_element_size must be finite and positive.")
    if int(element_order) != 1:
        raise ConfigurationError("v1 supports first-order triangles only.")
    try:
        import gmsh
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MeshError("Gmsh is required to discretize periodic geometry.") from exc

    xmin, xmax = geometry.x_span
    zmin, zmax = geometry.z_span
    width = xmax - xmin
    period = zmax - zmin
    scale = 1.0 / max(width, period)
    owned = not bool(gmsh.isInitialized())
    model_name = f"fem_periodic_2d_{uuid4().hex}"
    previous_model = ""
    model_added = False
    try:
        if owned:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
        else:
            try:
                previous_model = str(gmsh.model.getCurrent())
            except Exception:
                previous_model = ""
        gmsh.model.add(model_name)
        model_added = True
        occ = gmsh.model.occ
        base = (2, int(occ.addRectangle(0.0, 0.0, 0.0, width * scale, period * scale)))
        objects = [*geometry.regions, *geometry.boundaries, *geometry.refinements]
        inputs = [base] + [
            _add_occ_shape(gmsh, item.shape, (xmin, zmin), scale)
            for item in objects
        ]
        if len(inputs) > 1:
            fragments, provenance = occ.fragment(
                [inputs[0]], inputs[1:], removeObject=True, removeTool=True
            )
            if len(provenance) != len(inputs):
                raise MeshError("Gmsh returned incomplete fragment provenance.")
            all_surfaces = [int(tag) for dimension, tag in fragments if dimension == 2]
        else:
            all_surfaces = [base[1]]
            provenance = [[base]]
        occ.synchronize()

        descendants: dict[int, set[int]] = {}
        for item_index, item in enumerate(objects, start=1):
            descendants[item.id] = {
                int(tag) for dimension, tag in provenance[item_index] if dimension == 2
            }
        boundary_surfaces = {
            boundary.id: set(descendants[boundary.id])
            for boundary in geometry.boundaries
        }
        boundary_curves = {
            identifier: _boundary_curves(gmsh, sorted(surfaces))
            for identifier, surfaces in boundary_surfaces.items()
        }
        removed = set().union(*boundary_surfaces.values()) if boundary_surfaces else set()
        if removed:
            occ.remove([(2, surface) for surface in sorted(removed)], recursive=False)
            occ.synchronize()
        solve_surfaces = [surface for surface in all_surfaces if surface not in removed]
        if not solve_surfaces:
            raise MeshError("Boundary objects remove the entire periodic cell.")

        physical_names: dict[int, str] = {1: "background"}
        material_for_surface: dict[int, int] = {surface: 1 for surface in solve_surfaces}
        for tag, region in enumerate(geometry.regions, start=2):
            physical_names[tag] = region.name
            for surface in descendants[region.id] & set(solve_surfaces):
                material_for_surface[surface] = tag
        for tag in sorted(set(material_for_surface.values())):
            surfaces = [surface for surface, value in material_for_surface.items() if value == tag]
            group = gmsh.model.addPhysicalGroup(2, surfaces)
            gmsh.model.setPhysicalName(2, group, physical_names[tag])

        curve_groups: dict[str, set[int]] = {"periodic_master": set(), "periodic_slave": set()}
        periodic_pairs = _pair_periodic_curves(
            gmsh,
            solve_surfaces,
            width=width,
            period=period,
            scale=scale,
        )
        curve_groups["periodic_slave"].update(slave for slave, _ in periodic_pairs)
        curve_groups["periodic_master"].update(master for _, master in periodic_pairs)
        outer_curves = _boundary_curves(gmsh, solve_surfaces)
        tolerance = 2e-6 * max(width, period) * scale
        outer_x: set[int] = set()
        for curve in outer_curves:
            x0, _, _, x1, _, _ = gmsh.model.getBoundingBox(1, curve)
            if (abs(x0) <= tolerance and abs(x1) <= tolerance) or (
                abs(x0 - width * scale) <= tolerance and abs(x1 - width * scale) <= tolerance
            ):
                outer_x.add(curve)
        curve_groups[f"outer_{geometry.outer_boundary}"] = outer_x
        for boundary in geometry.boundaries:
            curves = boundary_curves[boundary.id]
            curve_groups.setdefault(boundary.kind, set()).update(curves)
            curve_groups[boundary.name] = curves

        # Low-index background regions retain the requested coarse size.
        # Higher local wavenumber regions receive proportionally more elements
        # per wavelength, while explicit refinement regions remain overrides.
        fields: list[int] = []
        target_sizes = [maximum]
        background_scale = max(_material_scale(geometry.background), np.finfo(float).tiny)
        solve_set = set(solve_surfaces)
        for region in geometry.regions:
            surfaces = sorted(descendants[region.id] & solve_set)
            if not surfaces:
                continue
            ratio = _material_scale(region.material) / background_scale
            target = maximum / max(1.0, ratio)
            target = max(target, maximum * 0.1)
            if target >= maximum * (1.0 - 1e-12):
                continue
            field = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumber(field, "VIn", target * scale)
            gmsh.model.mesh.field.setNumber(field, "VOut", maximum * scale)
            gmsh.model.mesh.field.setNumbers(field, "SurfacesList", surfaces)
            fields.append(field)
            target_sizes.append(target)
        for refinement in geometry.refinements:
            surfaces = sorted(descendants[refinement.id] & solve_set)
            if not surfaces:
                continue
            target = min(maximum, refinement.max_element_size)
            field = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumber(field, "VIn", target * scale)
            gmsh.model.mesh.field.setNumber(field, "VOut", maximum * scale)
            gmsh.model.mesh.field.setNumbers(field, "SurfacesList", surfaces)
            fields.append(field)
            target_sizes.append(target)

        # Resolve the field singularity and conductor-edge current crowding
        # with a smooth distance field around internal and outer PEC curves.
        pec_curve_groups = (
            (sorted(set(curve_groups.get("pec", set()))), 0.3),
            (sorted(set(curve_groups.get("outer_pec", set()))), 0.6),
        )
        for pec_curves, refinement_factor in pec_curve_groups:
            if not pec_curves:
                continue
            distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance, "CurvesList", pec_curves)
            gmsh.model.mesh.field.setNumber(distance, "Sampling", 80)
            threshold = gmsh.model.mesh.field.add("Threshold")
            pec_target = maximum * refinement_factor
            gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMin", pec_target * scale)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", maximum * scale)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.25 * maximum * scale)
            gmsh.model.mesh.field.setNumber(threshold, "DistMax", 2.0 * maximum * scale)
            fields.append(threshold)
            target_sizes.append(pec_target)
        if len(fields) == 1:
            gmsh.model.mesh.field.setAsBackgroundMesh(fields[0])
        elif fields:
            combined = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(combined, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(combined)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", maximum * scale)
        gmsh.option.setNumber(
            "Mesh.MeshSizeMin", max(min(target_sizes) * scale * 0.05, 1e-12)
        )
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.model.mesh.generate(2)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        all_points = np.asarray(coordinates, dtype=float).reshape(-1, 3)[:, :2] / scale
        all_points += np.asarray((xmin, zmin))
        all_lookup = {int(tag): index for index, tag in enumerate(node_tags)}
        element_tags, connectivity = gmsh.model.mesh.getElementsByType(2)
        if len(element_tags) == 0:
            raise MeshError("Gmsh generated no first-order triangles.")
        raw = np.asarray(connectivity, dtype=np.int64).reshape(-1, 3)
        used_tags = np.unique(raw)
        points = np.asarray([all_points[all_lookup[int(tag)]] for tag in used_tags], dtype=float)
        lookup = {int(tag): index for index, tag in enumerate(used_tags)}
        triangles = np.fromiter(
            (lookup[int(tag)] for tag in raw.ravel()), dtype=np.int64, count=raw.size
        ).reshape(-1, 3)

        element_to_material: dict[int, int] = {}
        for surface, material_tag in material_for_surface.items():
            surface_elements, _ = gmsh.model.mesh.getElementsByType(2, int(surface))
            element_to_material.update((int(element), material_tag) for element in surface_elements)
        try:
            element_materials = np.fromiter(
                (element_to_material[int(element)] for element in element_tags),
                dtype=np.int32,
                count=len(element_tags),
            )
        except KeyError as exc:
            raise MeshError("Gmsh lost a material association while meshing.") from exc

        periodic_node_map: dict[int, int] = {}
        for slave_curve, _ in periodic_pairs:
            _, slave_tags, master_tags, _ = gmsh.model.mesh.getPeriodicNodes(
                1, int(slave_curve), True
            )
            for slave_tag, master_tag in zip(slave_tags, master_tags, strict=True):
                if int(slave_tag) in lookup and int(master_tag) in lookup:
                    slave_node = lookup[int(slave_tag)]
                    master_node = lookup[int(master_tag)]
                    previous = periodic_node_map.setdefault(slave_node, master_node)
                    if previous != master_node:
                        raise MeshError(
                            "Gmsh returned conflicting master nodes at a periodic curve junction."
                        )
        if not periodic_node_map:
            raise MeshError("Gmsh produced no periodic node correspondence.")
        slave_nodes = np.asarray(sorted(periodic_node_map), dtype=np.int64)
        master_nodes = np.asarray(
            [periodic_node_map[int(slave)] for slave in slave_nodes], dtype=np.int64
        )

        mesh = MeshTri(points.T, triangles.T)
        groups = _facet_groups(mesh, lookup, gmsh, curve_groups)
    except (MeshError, ConfigurationError):
        raise
    except Exception as exc:
        raise MeshError(f"Gmsh failed to generate the 2D periodic mesh: {exc}") from exc
    finally:
        if owned:
            gmsh.finalize()
        elif model_added:
            try:
                gmsh.model.setCurrent(model_name)
                gmsh.model.remove()
                if previous_model:
                    gmsh.model.setCurrent(previous_model)
            except Exception:
                pass

    # Verify the geometric correspondence independently of Gmsh's metadata.
    tolerance_physical = 2e-8 * max(width, period)
    differences = points[slave_nodes] - points[master_nodes]
    expected = np.asarray((0.0, period))
    if np.max(np.linalg.norm(differences - expected, axis=1)) > tolerance_physical:
        raise MeshError("Periodic node pairs are not related by the cell translation.")
    # The two *topologies* must match, but the one-sided material traces may
    # differ: placing a dielectric interface at the cell seam is a valid
    # periodic laminate.  Objects that genuinely cross the seam still need to
    # be split explicitly so Gmsh can produce matching face partitions.
    hmin, hmax = _edge_extrema(points, triangles)
    slave_nodes.setflags(write=False)
    master_nodes.setflags(write=False)
    element_materials.setflags(write=False)
    return FEMPeriodicMesh2D(
        mesh=mesh,
        element_tags=element_materials,
        physical_names=physical_names,
        boundary_facets=groups,
        slave_nodes=slave_nodes,
        master_nodes=master_nodes,
        info=MeshInfo(
            nodes=points.shape[0],
            elements=triangles.shape[0],
            minimum_edge=hmin,
            maximum_edge=hmax,
            requested_maximum_edge=maximum,
        ),
        geometry_revision=geometry.revision,
    )


__all__ = ["FEMPeriodicMesh2D", "MeshInfo", "discretize_periodic_2d"]

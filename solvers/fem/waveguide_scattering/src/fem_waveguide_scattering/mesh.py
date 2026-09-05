"""Gmsh-backed conforming triangular mesh generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Literal
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from skfem import MeshTri

from .exceptions import MeshError
from .geometry import Circle, GeometryModel, PECSegment, Polygon, Rectangle, Shape


_GMSH_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class MeshInfo:
    """User-facing mesh metadata, in SI units."""

    nodes: int
    elements: int
    minimum_edge: float
    maximum_edge: float
    requested_maximum_edge: float


@dataclass(slots=True)
class Mesh2D:
    """scikit-fem mesh plus material and ideal-PEC topology metadata."""

    mesh: MeshTri
    element_tags: NDArray[np.int32]
    physical_names: dict[int, str]
    info: MeshInfo
    background_pec_facets: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    actual_pec_facets: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    released_pec_facets: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    pec_slot_facets: dict[str, NDArray[np.int32]] = field(default_factory=dict)
    inserted_pec_facets: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )

    def elements_in(self, region: str | int) -> NDArray[np.int64]:
        if isinstance(region, str):
            inverse = {name: tag for tag, name in self.physical_names.items()}
            if region not in inverse:
                raise MeshError(f"No physical region was found for material {region!r}.")
            region = inverse[region]
        return np.flatnonzero(self.element_tags == int(region))

    def pec_facets(
        self, profile: Literal["actual", "background"]
    ) -> NDArray[np.int32]:
        """Return global ``MeshTri`` facet indices for one PEC profile."""

        if profile == "actual":
            return self.actual_pec_facets
        if profile == "background":
            return self.background_pec_facets
        raise ValueError("profile must be 'actual' or 'background'.")

    def facets_in_slot(self, slot: str) -> NDArray[np.int32]:
        """Return global facet indices released by one named PEC slot."""

        try:
            return self.pec_slot_facets[slot]
        except KeyError as exc:
            raise MeshError(f"No PEC slot named {slot!r} exists in this mesh.") from exc


def _add_occ_shape(
    gmsh: object,
    shape: Shape,
    *,
    origin: tuple[float, float],
    scale: float,
) -> int:
    from cem_common.shapes import Shape
    from cem_common._occ import add_shape
    if isinstance(shape, Shape):
        return add_shape(gmsh, shape, origin, scale)[1]
    occ = gmsh.model.occ
    x0, z0 = origin
    if isinstance(shape, Rectangle):
        return occ.addRectangle(
            (shape.x[0] - x0) * scale,
            (shape.z[0] - z0) * scale,
            0.0,
            (shape.x[1] - shape.x[0]) * scale,
            (shape.z[1] - shape.z[0]) * scale,
        )
    if isinstance(shape, Circle):
        return occ.addDisk(
            (shape.center[0] - x0) * scale,
            (shape.center[1] - z0) * scale,
            0.0,
            shape.radius * scale,
            shape.radius * scale,
        )
    if isinstance(shape, Polygon):
        points = [
            occ.addPoint((x - x0) * scale, (z - z0) * scale, 0.0)
            for x, z in shape.points
        ]
        lines = [
            occ.addLine(points[i], points[(i + 1) % len(points)])
            for i in range(len(points))
        ]
        return occ.addPlaneSurface([occ.addCurveLoop(lines)])
    raise TypeError(f"Unsupported geometry shape {type(shape).__name__}.")


def _edge_range(points: NDArray[np.float64], triangles: NDArray[np.int64]) -> tuple[float, float]:
    pairs = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
    )
    lengths = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    return float(lengths.min()), float(lengths.max())


def _facets_on_pec_segments(
    mesh: MeshTri,
    segments: tuple[PECSegment, ...],
) -> NDArray[np.int32]:
    """Locate mesh facets covering constant-x PEC segments."""

    if not segments:
        return np.empty(0, dtype=np.int32)
    facet_points = np.asarray(mesh.p[:, mesh.facets], dtype=float)
    coordinate_scale = max(1.0, float(np.max(np.abs(mesh.p))))
    tolerance = 512.0 * np.finfo(float).eps * coordinate_scale
    facet_z_min = np.min(facet_points[1], axis=0)
    facet_z_max = np.max(facet_points[1], axis=0)
    nondegenerate = facet_z_max - facet_z_min > tolerance
    selected: list[NDArray[np.int64]] = []
    for segment in segments:
        on_x = np.all(
            np.abs(facet_points[0] - segment.x) <= tolerance,
            axis=0,
        )
        within_z = (facet_z_min >= segment.z[0] - tolerance) & (
            facet_z_max <= segment.z[1] + tolerance
        )
        facets = np.flatnonzero(on_x & within_z & nondegenerate)
        if facets.size == 0:
            raise MeshError(
                f"PEC segment {segment.name!r} at x={segment.x:g}, "
                f"z={segment.z} is not represented by mesh facets."
            )
        selected.append(facets)
    return np.asarray(np.unique(np.concatenate(selected)), dtype=np.int32)


def _material_index(eps_r: complex, mu_r: complex) -> float:
    """Return the scalar magnitude index used for local wavelength sizing."""

    return float(np.sqrt(abs(complex(eps_r) * complex(mu_r))))


def _material_target_sizes(
    geometry: GeometryModel,
    maximum_size: float,
    *,
    enabled: bool,
    refinement_factor: float,
) -> dict[int, float]:
    """Return per-physical-tag targets relative to the exterior wavelength."""

    materials = {
        1: geometry.exterior,
        **{region.physical_tag: region.material for region in geometry.regions},
    }
    if not enabled:
        return {tag: maximum_size for tag in materials}
    reference_index = max(
        _material_index(geometry.exterior.eps_r, geometry.exterior.mu_r),
        1.0,
    )
    targets = {1: maximum_size}
    targets.update(
        {
            tag: maximum_size
            * min(
                1.0,
                refinement_factor
                * reference_index
                / max(
                    _material_index(material.eps_r, material.mu_r),
                    np.finfo(float).tiny,
                ),
            )
            for tag, material in materials.items()
            if tag != 1
        }
    )
    return targets


def _pec_curve_tags(
    gmsh: object,
    geometry: GeometryModel,
    *,
    origin: tuple[float, float],
    scale: float,
) -> list[int]:
    """Find OCC curve entities coincident with actual PEC segments."""

    segments = geometry.pec_segments(profile="actual")
    if not segments:
        return []
    x0, z0 = origin
    tolerance = 2e-6
    selected: set[int] = set()
    for _, curve in gmsh.model.getEntities(1):
        bounds = gmsh.model.getBoundingBox(1, curve)
        curve_x_min, curve_z_min = float(bounds[0]), float(bounds[1])
        curve_x_max, curve_z_max = float(bounds[3]), float(bounds[4])
        if curve_z_max - curve_z_min <= tolerance:
            continue
        for segment in segments:
            segment_x = (segment.x - x0) * scale
            segment_z_min = (segment.z[0] - z0) * scale
            segment_z_max = (segment.z[1] - z0) * scale
            if (
                abs(curve_x_min - segment_x) <= tolerance
                and abs(curve_x_max - segment_x) <= tolerance
                and curve_z_min >= segment_z_min - tolerance
                and curve_z_max <= segment_z_max + tolerance
            ):
                selected.add(int(curve))
                break
    if not selected:
        raise MeshError("Gmsh produced no curve entities for the actual PEC geometry.")
    return sorted(selected)


def _configure_size_fields(
    gmsh: object,
    geometry: GeometryModel,
    *,
    grouped_surfaces: dict[int, list[int]],
    maximum_size: float,
    cad_scale: float,
    refine_dielectrics: bool,
    dielectric_refinement_factor: float,
    refine_pec: bool,
    pec_refinement_factor: float,
    pec_refinement_distance: float | None,
) -> None:
    """Install dielectric-restricted and PEC-distance Gmsh size fields."""

    target_sizes = _material_target_sizes(
        geometry,
        maximum_size,
        enabled=refine_dielectrics,
        refinement_factor=dielectric_refinement_factor,
    )
    fields: list[int] = []
    for physical_tag, surfaces in grouped_surfaces.items():
        target = target_sizes.get(physical_tag, maximum_size)
        if target >= maximum_size * (1.0 - 1e-12):
            continue
        constant = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            constant,
            "F",
            f"{target * cad_scale:.17g}",
        )
        restricted = gmsh.model.mesh.field.add("Restrict")
        gmsh.model.mesh.field.setNumber(restricted, "InField", constant)
        gmsh.model.mesh.field.setNumbers(restricted, "SurfacesList", surfaces)
        fields.append(restricted)

    pec_curves = (
        _pec_curve_tags(
            gmsh,
            geometry,
            origin=(geometry.x_span[0], geometry.z_span[0]),
            scale=cad_scale,
        )
        if refine_pec
        else []
    )
    if pec_curves:
        minimum_material_size = min(target_sizes.values(), default=maximum_size)
        pec_size = pec_refinement_factor * minimum_material_size
        transition_distance = (
            3.0 * minimum_material_size
            if pec_refinement_distance is None
            else pec_refinement_distance
        )
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "CurvesList", pec_curves)
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(
            threshold, "SizeMin", pec_size * cad_scale
        )
        gmsh.model.mesh.field.setNumber(
            threshold, "SizeMax", maximum_size * cad_scale
        )
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(
            threshold,
            "DistMax",
            transition_distance * cad_scale,
        )
        fields.append(threshold)

    if fields:
        background = fields[0]
        if len(fields) > 1:
            background = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(background, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(background)


def generate_mesh(
    geometry: GeometryModel,
    *,
    max_element_size: float,
    x_partitions: tuple[float, ...] = (),
    z_partitions: tuple[float, ...] = (),
    refine_dielectrics: bool = True,
    dielectric_refinement_factor: float = 0.5,
    refine_pec: bool = True,
    pec_refinement_factor: float = 0.5,
    pec_refinement_distance: float | None = None,
) -> Mesh2D:
    """Generate a triangular mesh whose material and PML interfaces conform.

    Gmsh's OCC fragment operation is used for every material shape.  Material
    values are still evaluated from :class:`GeometryModel` at quadrature
    points; physical tags provide stable diagnostics and export metadata.
    """

    if not np.isfinite(max_element_size) or max_element_size <= 0.0:
        raise MeshError("max_element_size must be finite and positive.")
    if not isinstance(refine_dielectrics, (bool, np.bool_)):
        raise MeshError("refine_dielectrics must be a boolean.")
    if not isinstance(refine_pec, (bool, np.bool_)):
        raise MeshError("refine_pec must be a boolean.")
    if (
        not np.isfinite(dielectric_refinement_factor)
        or dielectric_refinement_factor <= 0.0
        or dielectric_refinement_factor > 1.0
    ):
        raise MeshError(
            "dielectric_refinement_factor must be in the interval (0, 1]."
        )
    if (
        not np.isfinite(pec_refinement_factor)
        or pec_refinement_factor <= 0.0
        or pec_refinement_factor > 1.0
    ):
        raise MeshError("pec_refinement_factor must be in the interval (0, 1].")
    if pec_refinement_distance is not None and (
        not np.isfinite(pec_refinement_distance)
        or pec_refinement_distance <= 0.0
    ):
        raise MeshError("pec_refinement_distance must be finite and positive.")
    try:
        import gmsh
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MeshError(
            "The Gmsh Python binding could not be loaded. Install the mesh "
            "dependencies and, on Windows, run through 'conda run' so Gmsh's "
            "DLL directory is active."
        ) from exc

    with _GMSH_LOCK:
        owned = not bool(gmsh.isInitialized())
        if owned:
            gmsh.initialize()
        previous_model = "" if owned else str(gmsh.model.getCurrent())
        model_name = f"cem_scattering_{uuid4().hex}"
        model_added = False
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(model_name)
            model_added = True
            occ = gmsh.model.occ
            xmin, xmax = geometry.x_span
            zmin, zmax = geometry.z_span
            cad_scale = 1.0 / max(xmax - xmin, zmax - zmin)

            def grid_coordinates(
                values: tuple[float, ...], lower: float, upper: float, name: str
            ) -> list[float]:
                converted = [float(value) for value in values]
                if not np.isfinite(converted).all():
                    raise MeshError(f"{name} partition coordinates must be finite.")
                return sorted({lower, upper, *(v for v in converted if lower < v < upper)})

            rectangles = [
                region for region in geometry.regions if isinstance(region.shape, Rectangle)
            ]
            x_grid = grid_coordinates(
                (
                    *x_partitions,
                    *(value for region in rectangles for value in region.shape.x),
                    *(sheet.x for sheet in geometry.pec_sheets),
                ),
                xmin,
                xmax,
                "x",
            )
            z_grid = grid_coordinates(
                (
                    *z_partitions,
                    *(value for region in rectangles for value in region.shape.z),
                    *(value for sheet in geometry.pec_sheets for value in sheet.z),
                    *(value for slot in geometry.pec_slots for value in slot.z),
                ),
                zmin,
                zmax,
                "z",
            )
            grid_cells: list[tuple[tuple[int, int], float, float]] = []
            for ix in range(len(x_grid) - 1):
                for iz in range(len(z_grid) - 1):
                    entity = (
                        2,
                        occ.addRectangle(
                            (x_grid[ix] - xmin) * cad_scale,
                            (z_grid[iz] - zmin) * cad_scale,
                            0.0,
                            (x_grid[ix + 1] - x_grid[ix]) * cad_scale,
                            (z_grid[iz + 1] - z_grid[iz]) * cad_scale,
                        ),
                    )
                    grid_cells.append(
                        (
                            entity,
                            0.5 * (x_grid[ix] + x_grid[ix + 1]),
                            0.5 * (z_grid[iz] + z_grid[iz + 1]),
                        )
                    )
            grid_surfaces = [entity for entity, _, _ in grid_cells]
            curved_regions = [
                region for region in geometry.regions if not isinstance(region.shape, Rectangle)
            ]
            material_surfaces = [
                (
                    2,
                    _add_occ_shape(
                        gmsh,
                        region.shape,
                        origin=(xmin, zmin),
                        scale=cad_scale,
                    ),
                )
                for region in curved_regions
            ]
            # Begin with disjoint rectangular cells rather than overlapping
            # full-area partition slabs.  One BooleanFragments operation then
            # makes every cell/material boundary conforming without relying on
            # lower-dimensional embedded curves, which are fragile when a line
            # crosses several already-fragmented faces.
            if len(grid_surfaces) + len(material_surfaces) > 1:
                fragmented, provenance = occ.fragment(
                    [grid_surfaces[0]],
                    [*grid_surfaces[1:], *material_surfaces],
                    removeObject=True,
                    removeTool=True,
                )
                solve_surfaces = [entity for entity in fragmented if entity[0] == 2]
                surface_tags = {entity: 1 for _, entity in solve_surfaces}
                region_provenance = {
                    id(region): provenance[len(grid_surfaces) + index]
                    for index, region in enumerate(curved_regions)
                }
                # Match GeometryModel.material_at precedence: background
                # layers first, then finite perturbations, each in insertion
                # order.  Boolean-fragment provenance is reliable even when a
                # fragment's centre of mass lies outside a concave polygon.
                for region in (*geometry.background_regions, *geometry.perturbations):
                    if isinstance(region.shape, Rectangle):
                        outputs = (
                            output
                            for index, (_, center_x, center_z) in enumerate(grid_cells)
                            if bool(region.contains(center_x, center_z))
                            for output in provenance[index]
                        )
                    else:
                        outputs = iter(region_provenance[id(region)])
                    for dimension, entity in outputs:
                        if dimension == 2:
                            surface_tags[entity] = region.physical_tag
            else:
                solve_surfaces = grid_surfaces
                surface_tags = {grid_surfaces[0][1]: 1}
                _, center_x, center_z = grid_cells[0]
                for region in (*geometry.background_regions, *geometry.perturbations):
                    if bool(region.contains(center_x, center_z)):
                        surface_tags[grid_surfaces[0][1]] = region.physical_tag
            occ.synchronize()

            grouped: dict[int, list[int]] = {}
            for _, entity in gmsh.model.getEntities(2):
                tag = surface_tags.get(entity)
                if tag is None:
                    raise MeshError(
                        f"Gmsh surface {entity} lost its material provenance."
                    )
                grouped.setdefault(tag, []).append(entity)
            for tag, entities in grouped.items():
                gmsh.model.addPhysicalGroup(2, entities, tag)
                gmsh.model.setPhysicalName(2, tag, geometry.physical_names.get(tag, f"region_{tag}"))

            _configure_size_fields(
                gmsh,
                geometry,
                grouped_surfaces=grouped,
                maximum_size=float(max_element_size),
                cad_scale=cad_scale,
                refine_dielectrics=bool(refine_dielectrics),
                dielectric_refinement_factor=float(dielectric_refinement_factor),
                refine_pec=bool(refine_pec),
                pec_refinement_factor=float(pec_refinement_factor),
                pec_refinement_distance=(
                    None
                    if pec_refinement_distance is None
                    else float(pec_refinement_distance)
                ),
            )

            gmsh.option.setNumber(
                "Mesh.MeshSizeMax", float(max_element_size * cad_scale)
            )
            minimum_target = min(
                _material_target_sizes(
                    geometry,
                    float(max_element_size),
                    enabled=bool(refine_dielectrics),
                    refinement_factor=float(dielectric_refinement_factor),
                ).values(),
                default=float(max_element_size),
            )
            if refine_pec and geometry.pec_segments(profile="actual"):
                minimum_target *= float(pec_refinement_factor)
            gmsh.option.setNumber(
                "Mesh.MeshSizeMin",
                min(float(max_element_size) * 0.25, minimum_target) * cad_scale,
            )
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.model.mesh.generate(2)

            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            points = np.asarray(coordinates, dtype=float).reshape(-1, 3)[:, :2]
            points = points / cad_scale + np.asarray((xmin, zmin), dtype=float)
            element_tags, connectivity = gmsh.model.mesh.getElementsByType(2)
            if len(element_tags) == 0:
                element_types = tuple(
                    int(value) for value in gmsh.model.mesh.getElementTypes(dim=2)
                )
                raise MeshError(
                    "Gmsh generated no first-order triangular elements; "
                    f"two-dimensional element types were {element_types}."
                )
            node_lookup = {int(tag): i for i, tag in enumerate(node_tags)}
            raw = np.asarray(connectivity, dtype=np.int64).reshape(-1, 3)
            triangles = np.fromiter(
                (node_lookup[int(tag)] for tag in raw.ravel()),
                dtype=np.int64,
                count=raw.size,
            ).reshape(-1, 3)
        except MeshError:
            raise
        except Exception as exc:
            raise MeshError(f"Gmsh failed to generate the solve mesh: {exc}") from exc
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
                    # Never mask the original meshing error during cleanup.
                    pass

    centroids = points[triangles].mean(axis=1)
    region_tags = geometry.region_tag_at(centroids[:, 0], centroids[:, 1])
    hmin, hmax = _edge_range(points, triangles)
    skmesh = MeshTri(points.T, triangles.T)
    background_pec_facets = _facets_on_pec_segments(
        skmesh, geometry.pec_segments(profile="background")
    )
    actual_pec_facets = _facets_on_pec_segments(
        skmesh, geometry.pec_segments(profile="actual")
    )
    released_pec_facets = np.asarray(
        np.setdiff1d(
            background_pec_facets,
            actual_pec_facets,
            assume_unique=True,
        ),
        dtype=np.int32,
    )
    inserted_pec_facets = np.asarray(
        np.setdiff1d(
            actual_pec_facets,
            background_pec_facets,
            assume_unique=True,
        ),
        dtype=np.int32,
    )
    sheet_by_name = {sheet.name: sheet for sheet in geometry.pec_sheets}
    pec_slot_facets = {
        slot.name: _facets_on_pec_segments(
            skmesh,
            (
                PECSegment(
                    slot.name,
                    sheet_by_name[slot.sheet_name].x,
                    slot.z,
                ),
            ),
        )
        for slot in geometry.pec_slots
    }
    slot_union = np.asarray(
        np.unique(
            np.concatenate(tuple(pec_slot_facets.values()))
            if pec_slot_facets
            else np.empty(0, dtype=np.int32)
        ),
        dtype=np.int32,
    )
    if not np.array_equal(released_pec_facets, slot_union):
        raise MeshError(
            "Actual/background PEC facet subtraction is inconsistent with the "
            "named finite slots."
        )
    return Mesh2D(
        mesh=skmesh,
        element_tags=np.asarray(region_tags, dtype=np.int32),
        physical_names=geometry.physical_names,
        info=MeshInfo(
            nodes=points.shape[0],
            elements=triangles.shape[0],
            minimum_edge=hmin,
            maximum_edge=hmax,
            requested_maximum_edge=float(max_element_size),
        ),
        background_pec_facets=background_pec_facets,
        actual_pec_facets=actual_pec_facets,
        released_pec_facets=released_pec_facets,
        inserted_pec_facets=inserted_pec_facets,
        pec_slot_facets=pec_slot_facets,
    )

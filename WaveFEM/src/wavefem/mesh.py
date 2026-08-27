"""Gmsh-backed conforming triangular mesh generation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from skfem import MeshTri

from .exceptions import MeshError
from .geometry import Circle, GeometryModel, Polygon, Rectangle, Shape


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
    """scikit-fem mesh plus stable actual-material region tags."""

    mesh: MeshTri
    element_tags: NDArray[np.int32]
    physical_names: dict[int, str]
    info: MeshInfo

    def elements_in(self, region: str | int) -> NDArray[np.int64]:
        if isinstance(region, str):
            inverse = {name: tag for tag, name in self.physical_names.items()}
            if region not in inverse:
                raise MeshError(f"No physical region was found for material {region!r}.")
            region = inverse[region]
        return np.flatnonzero(self.element_tags == int(region))


def _add_occ_shape(
    gmsh: object,
    shape: Shape,
    *,
    origin: tuple[float, float],
    scale: float,
) -> int:
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


def generate_mesh(
    geometry: GeometryModel,
    *,
    max_element_size: float,
    x_partitions: tuple[float, ...] = (),
    z_partitions: tuple[float, ...] = (),
) -> Mesh2D:
    """Generate a triangular mesh whose material and PML interfaces conform.

    Gmsh's OCC fragment operation is used for every material shape.  Material
    values are still evaluated from :class:`GeometryModel` at quadrature
    points; physical tags provide stable diagnostics and export metadata.
    """

    if not np.isfinite(max_element_size) or max_element_size <= 0.0:
        raise MeshError("max_element_size must be finite and positive.")
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
        model_name = f"wavefem_{uuid4().hex}"
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
                (*x_partitions, *(value for region in rectangles for value in region.shape.x)),
                xmin,
                xmax,
                "x",
            )
            z_grid = grid_coordinates(
                (*z_partitions, *(value for region in rectangles for value in region.shape.z)),
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

            gmsh.option.setNumber(
                "Mesh.MeshSizeMax", float(max_element_size * cad_scale)
            )
            gmsh.option.setNumber(
                "Mesh.MeshSizeMin", float(max_element_size * cad_scale) * 0.25
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
    )

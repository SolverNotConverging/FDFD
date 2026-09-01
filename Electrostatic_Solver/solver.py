"""Geometry-first 1D/2D finite-element electrostatic solver."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .exceptions import GeometryError, NotDiscretizedError, SolverError
from .geometry import ChargeRegion, Circle, GeometryModel, Interval, MaterialRegion, Permittivity, Polygon, PotentialRegion, Rectangle
from .meshing import FEMMesh, discretize_1d, discretize_2d
from .results import ElectrostaticResult


EPSILON_0 = 8.854_187_812_8e-12
RegionInput: TypeAlias = Interval | Rectangle | Circle | Polygon | slice | tuple[slice, slice] | str


def _dimension(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) not in (1, 2):
        raise GeometryError("dim must be 1 or 2.")
    return int(value)


def _mesh_shape(value: Sequence[int] | int | None, dim: int) -> tuple[int, ...] | None:
    if value is None:
        return None
    entries = (value,) if np.isscalar(value) else tuple(value)
    if len(entries) != dim:
        raise GeometryError(f"mesh_size must contain {dim} positive integer entries.")
    parsed: list[int] = []
    for entry in entries:
        if isinstance(entry, (bool, np.bool_)) or int(entry) != entry or int(entry) < 2:
            raise GeometryError("mesh_size entries must be integers of at least two.")
        parsed.append(int(entry))
    return tuple(parsed)


def _domain_spans(domain: object, dim: int, mesh_size: tuple[int, ...] | None) -> tuple[tuple[float, float], tuple[float, float] | None]:
    if domain is None:
        if mesh_size is None:
            return (0.0, 1.0), None if dim == 1 else (0.0, 1.0)
        return (0.0, float(mesh_size[0])), None if dim == 1 else (0.0, float(mesh_size[1]))
    if dim == 1:
        raw = tuple(domain)  # type: ignore[arg-type]
        if len(raw) == 1 and not np.isscalar(raw[0]):
            raw = tuple(raw[0])
        if len(raw) != 2:
            raise GeometryError("a 1D domain must be (xmin, xmax).")
        return (float(raw[0]), float(raw[1])), None
    raw = tuple(domain)  # type: ignore[arg-type]
    if len(raw) != 2:
        raise GeometryError("a 2D domain must be ((xmin, xmax), (ymin, ymax)).")
    return tuple(float(value) for value in raw[0]), tuple(float(value) for value in raw[1])  # type: ignore[return-value]


class ElectrostaticSolver:
    """Solve ``-div(epsilon_0 epsilon_r grad(phi)) = rho`` with P1 FEM.

    The constructor records only a physical domain and optional target mesh
    counts.  Materials, charge regions, and fixed potentials remain continuous
    geometry until :meth:`discretize` (or an auto-discretizing :meth:`solve`)
    is called.  ``mesh_size`` and slice regions are retained for migration.
    """

    def __init__(
        self,
        mesh_size: Sequence[int] | int | None = None,
        dim: int = 2,
        *,
        domain: object | None = None,
        background_permittivity: float | Sequence[float] | Sequence[Sequence[float]] = 1.0,
        outer_potential: float | None = 0.0,
    ) -> None:
        self.dim = _dimension(dim)
        self.mesh_size = _mesh_shape(mesh_size, self.dim)
        x_span, y_span = _domain_spans(domain, self.dim, self.mesh_size)
        self.geometry = GeometryModel(
            self.dim,
            x_span,
            y_span,
            Permittivity.from_input(background_permittivity, self.dim),
        )
        self.model = self.geometry
        self.domain = (self.geometry.x_span,) if self.dim == 1 else (self.geometry.x_span, self.geometry.y_span)
        if outer_potential is not None and not np.isfinite(float(outer_potential)):
            raise GeometryError("outer_potential must be finite or None.")
        self.outer_potential = None if outer_potential is None else float(outer_potential)
        self.mesh: FEMMesh | None = None
        self.solution: ElectrostaticResult | None = None
        self.potential: NDArray[np.float64] | None = None
        self.fixed_mask: NDArray[np.bool_] | None = None
        self._discretization_settings: dict[str, object] | None = None
        self.geometry.add_change_listener(self._geometry_changed)

    def _geometry_changed(self) -> None:
        self.mesh = None
        self._clear_solution()

    def _clear_solution(self) -> None:
        self.solution = None
        self.potential = None
        self.fixed_mask = None

    @property
    def coordinates(self) -> NDArray[np.float64]:
        if self.mesh is None:
            raise NotDiscretizedError("call discretize() before requesting mesh coordinates.")
        return self.mesh.nodes

    @property
    def elements(self) -> NDArray[np.int64]:
        if self.mesh is None:
            raise NotDiscretizedError("call discretize() before requesting mesh elements.")
        return self.mesh.elements

    def _slice_interval(self, region: slice, axis: int) -> tuple[float, float]:
        if region.step not in (None, 1):
            raise GeometryError("legacy slice regions do not support a step other than one.")
        span = self.geometry.x_span if axis == 0 else self.geometry.y_span
        assert span is not None
        return (
            span[0] if region.start is None else float(region.start),
            span[1] if region.stop is None else float(region.stop),
        )

    def _shape(self, region: RegionInput) -> Interval | Rectangle | Circle | Polygon | str:
        if isinstance(region, str):
            return region.lower()
        if isinstance(region, (Interval, Rectangle, Circle, Polygon)):
            return region
        if self.dim == 1 and isinstance(region, slice):
            return Interval(self._slice_interval(region, 0))
        if self.dim == 2 and isinstance(region, tuple) and len(region) == 2 and all(isinstance(entry, slice) for entry in region):
            return Rectangle(self._slice_interval(region[0], 0), self._slice_interval(region[1], 1))
        raise GeometryError("region must be a geometry primitive, boundary name, or legacy slice region.")

    def set_potential(self, region: RegionInput, potential_value: float, *, name: str | None = None) -> PotentialRegion:
        return self.geometry.add_potential(self._shape(region), potential_value, name=name)

    def add_object(
        self,
        region: RegionInput,
        erxx: float = 1.0,
        eryy: float | None = None,
        *,
        erxy: float = 0.0,
        permittivity: float | Sequence[float] | Sequence[Sequence[float]] | None = None,
        name: str | None = None,
    ) -> MaterialRegion:
        shape = self._shape(region)
        if isinstance(shape, str):
            raise GeometryError("a material object requires an area/interval, not a boundary name.")
        if permittivity is not None:
            if eryy is not None or erxx != 1.0 or erxy != 0.0:
                raise GeometryError("use either permittivity=... or erxx/eryy/erxy, not both.")
            value = permittivity
        elif self.dim == 1:
            if eryy is not None or erxy != 0.0:
                raise GeometryError("eryy and erxy are not applicable to a 1D problem.")
            value = erxx
        else:
            # Historical add_object semantics changed only erxx when eryy was
            # omitted.  Use permittivity=<scalar> for a new isotropic region.
            yy = 1.0 if eryy is None else eryy
            value = ((erxx, erxy), (erxy, yy))
        return self.geometry.add_material(shape, Permittivity.from_input(value, self.dim), name=name)

    def add_charge_density(self, region: RegionInput, density: float, *, name: str | None = None) -> ChargeRegion:
        shape = self._shape(region)
        if isinstance(shape, str):
            raise GeometryError("volume charge requires an area/interval, not a boundary name.")
        return self.geometry.add_charge(shape, density, name=name)

    def remove(self, item: MaterialRegion | PotentialRegion | ChargeRegion) -> None:
        self.geometry.remove(item)

    def _default_maximum_size(self) -> float:
        if self.mesh_size is not None:
            hx = (self.geometry.x_span[1] - self.geometry.x_span[0]) / self.mesh_size[0]
            if self.dim == 1:
                return hx
            assert self.geometry.y_span is not None
            hy = (self.geometry.y_span[1] - self.geometry.y_span[0]) / self.mesh_size[1]
            return min(hx, hy)
        if self.dim == 1:
            return (self.geometry.x_span[1] - self.geometry.x_span[0]) / 80.0
        assert self.geometry.y_span is not None
        return min(self.geometry.x_span[1] - self.geometry.x_span[0], self.geometry.y_span[1] - self.geometry.y_span[0]) / 24.0

    def discretize(
        self,
        *,
        max_element_size: float | None = None,
        material_aware: bool = True,
        interface_refinement: float | None = 0.7,
        boundary_refinement: float | None = 0.5,
        interface_refinement_width: float | None = None,
        boundary_refinement_width: float | None = None,
    ) -> FEMMesh:
        """Discretize after geometry; high-Dk regions and boundaries refine locally."""

        maximum = self._default_maximum_size() if max_element_size is None else max_element_size
        settings = {
            "max_element_size": maximum,
            "material_aware": material_aware,
            "interface_refinement": interface_refinement,
            "boundary_refinement": boundary_refinement,
            "interface_refinement_width": interface_refinement_width,
            "boundary_refinement_width": boundary_refinement_width,
        }
        if self.dim == 1:
            self.mesh = discretize_1d(
                self.geometry,
                max_element_size=float(maximum),
                material_aware=material_aware,
                interface_refinement=interface_refinement,
                boundary_refinement=boundary_refinement,
            )
        else:
            self.mesh = discretize_2d(self.geometry, **settings)
        self._discretization_settings = settings
        self._clear_solution()
        return self.mesh

    def _boundary_mask(self, nodes: NDArray[np.float64], selector: str) -> NDArray[np.bool_]:
        x0, x1 = self.geometry.x_span
        scale = x1 - x0
        if self.dim == 2:
            assert self.geometry.y_span is not None
            scale = max(scale, self.geometry.y_span[1] - self.geometry.y_span[0])
        tolerance = 1e-9 * max(1.0, scale)
        left = np.abs(nodes[:, 0] - x0) <= tolerance
        right = np.abs(nodes[:, 0] - x1) <= tolerance
        if self.dim == 1:
            return left if selector == "left" else right
        assert self.geometry.y_span is not None
        y0, y1 = self.geometry.y_span
        bottom = np.abs(nodes[:, 1] - y0) <= tolerance
        top = np.abs(nodes[:, 1] - y1) <= tolerance
        return {"left": left, "right": right, "bottom": bottom, "top": top, "outer": left | right | bottom | top}[selector]

    def _dirichlet_data(self) -> tuple[NDArray[np.bool_], NDArray[np.float64], dict[str, NDArray[np.bool_]]]:
        assert self.mesh is not None
        nodes = self.mesh.nodes
        fixed = np.zeros(len(nodes), dtype=bool)
        values = np.zeros(len(nodes), dtype=float)
        named_masks: dict[str, NDArray[np.bool_]] = {}
        if self.outer_potential is not None:
            mask = self._boundary_mask(nodes, "outer" if self.dim == 2 else "left")
            if self.dim == 1:
                mask |= self._boundary_mask(nodes, "right")
            fixed[mask] = True
            values[mask] = self.outer_potential
            named_masks["outer"] = mask.copy()
        for region in self.geometry.potentials:
            if isinstance(region.shape, str):
                mask = self._boundary_mask(nodes, region.shape)
            elif self.dim == 1:
                mask = np.asarray(region.shape.contains(nodes[:, 0]), dtype=bool)  # type: ignore[union-attr]
            else:
                mask = np.asarray(region.shape.contains(nodes[:, 0], nodes[:, 1]), dtype=bool)  # type: ignore[union-attr]
            if not np.any(mask):
                raise SolverError(f"fixed-potential region {region.name!r} selected no mesh nodes.")
            # Later geometry wins consistently for both voltage and reaction-
            # charge ownership; no Dirichlet DOF is counted twice.
            for existing in named_masks.values():
                existing &= ~mask
            fixed[mask] = True
            values[mask] = region.value
            named_masks[region.name] = mask.copy()
        if not np.any(fixed):
            raise SolverError("the electrostatic problem needs at least one fixed-potential constraint.")
        return fixed, values, named_masks

    def _assemble(self) -> tuple[object, NDArray[np.float64]]:
        assert self.mesh is not None
        try:
            from skfem import Basis, BilinearForm, LinearForm, asm
            from skfem.element import ElementLineP1, ElementTriP1
            from skfem.helpers import grad
        except Exception as exc:
            raise SolverError("scikit-fem could not be imported; install the package dependencies.") from exc
        basis = Basis(self.mesh.mesh, ElementLineP1() if self.dim == 1 else ElementTriP1(), intorder=3)
        if self.dim == 1:
            @BilinearForm
            def stiffness(u: object, v: object, w: object) -> object:
                return EPSILON_0 * w.e00 * grad(u)[0] * grad(v)[0]
        else:
            @BilinearForm
            def stiffness(u: object, v: object, w: object) -> object:
                du, dv = grad(u), grad(v)
                return EPSILON_0 * (
                    w.e00 * du[0] * dv[0]
                    + w.e01 * (du[0] * dv[1] + du[1] * dv[0])
                    + w.e11 * du[1] * dv[1]
                )
        @LinearForm
        def source(v: object, w: object) -> object:
            return w.rho * v
        matrix = None
        for tag, material in self.geometry.material_table.items():
            indices = np.flatnonzero(self.mesh.element_tags == tag)
            if not len(indices):
                continue
            tensor = material.array
            contribution = asm(
                stiffness,
                basis.with_elements(indices),
                e00=float(tensor[0, 0]),
                e01=0.0 if self.dim == 1 else float(tensor[0, 1]),
                e11=0.0 if self.dim == 1 else float(tensor[1, 1]),
            )
            matrix = contribution if matrix is None else matrix + contribution
        if matrix is None:
            raise SolverError("the FEM mesh contains no material elements.")
        centers = self.mesh.nodes[self.mesh.elements].mean(axis=1)
        densities = self.geometry.charge_at(centers)
        rhs = np.zeros(basis.N, dtype=float)
        for density in np.unique(densities):
            if density != 0.0:
                indices = np.flatnonzero(densities == density)
                rhs += np.asarray(asm(source, basis.with_elements(indices), rho=float(density)), dtype=float)
        return matrix.tocsr(), rhs

    def _fields(self, potential: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        assert self.mesh is not None
        points, elements = self.mesh.nodes, self.mesh.elements
        if self.dim == 1:
            delta = points[elements[:, 1], 0] - points[elements[:, 0], 0]
            measures = np.abs(delta)
            element_e = (-(potential[elements[:, 1]] - potential[elements[:, 0]]) / delta)[:, None]
        else:
            vertices = points[elements]
            systems = np.stack((vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1)
            differences = np.stack((potential[elements[:, 1]] - potential[elements[:, 0]], potential[elements[:, 2]] - potential[elements[:, 0]]), axis=1)
            # Keep the right-hand side explicitly column-shaped for NumPy 2.x
            # batched solve semantics, then remove that final singleton axis.
            element_e = -np.linalg.solve(systems, differences[..., None])[..., 0]
            measures = 0.5 * np.abs(np.linalg.det(systems))
        element_d = np.empty_like(element_e)
        energy = 0.0
        for tag, material in self.geometry.material_table.items():
            mask = self.mesh.element_tags == tag
            element_d[mask] = EPSILON_0 * (element_e[mask] @ material.array.T)
            energy += 0.5 * float(np.sum(measures[mask] * np.einsum("ni,ni->n", element_e[mask], element_d[mask])))
        nodal_e = np.zeros((len(points), self.dim), dtype=float)
        nodal_d = np.zeros_like(nodal_e)
        weights = np.zeros(len(points), dtype=float)
        for local in range(elements.shape[1]):
            nodes = elements[:, local]
            np.add.at(nodal_e, nodes, element_e * measures[:, None])
            np.add.at(nodal_d, nodes, element_d * measures[:, None])
            np.add.at(weights, nodes, measures)
        nodal_e /= weights[:, None]
        nodal_d /= weights[:, None]
        return nodal_e, nodal_d, energy

    def solve(self, tol: float = 1e-10, max_iter: int | None = None) -> ElectrostaticResult:
        """Assemble and solve; legacy iteration arguments remain accepted."""
        tolerance = float(tol)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise SolverError("tol must be finite and positive.")
        if max_iter is not None and (isinstance(max_iter, (bool, np.bool_)) or int(max_iter) != max_iter or int(max_iter) < 1):
            raise SolverError("max_iter must be a positive integer or None.")
        if self.mesh is None:
            self.discretize()
        assert self.mesh is not None
        matrix, rhs = self._assemble()
        fixed, values, named_masks = self._dirichlet_data()
        free = ~fixed
        potential = values.copy()
        if np.any(free):
            try:
                from scipy.sparse.linalg import MatrixRankWarning, spsolve
                import warnings
                reduced_rhs = rhs[free] - matrix[free][:, fixed] @ potential[fixed]
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    potential[free] = spsolve(matrix[free][:, free], reduced_rhs)
            except Exception as exc:
                raise SolverError(f"the sparse FEM system could not be solved: {exc}") from exc
        if not np.isfinite(potential).all():
            raise SolverError("the sparse FEM solve returned non-finite potential values.")
        reaction = np.asarray(matrix @ potential - rhs, dtype=float)
        # Scale by the uncancelled row contributions.  Using ||A u|| alone is
        # meaningless for a homogeneous Laplace solve because the exact free-
        # row value is zero and would turn roundoff/roundoff into a ratio of 1.
        row_scale = np.abs(matrix[free]) @ np.abs(potential)
        denominator = max(
            float(np.linalg.norm(rhs[free])) + float(np.linalg.norm(row_scale)),
            np.finfo(float).tiny,
        )
        residual_norm = float(np.linalg.norm(reaction[free]) / denominator)
        if residual_norm > max(100.0 * tolerance, 1e-9):
            raise SolverError(f"FEM solve residual {residual_norm:.3e} exceeds tolerance.")
        electric, displacement, energy = self._fields(potential)
        result = ElectrostaticResult(
            mesh=self.mesh,
            potential=potential,
            electric_field=electric,
            displacement_field=displacement,
            reaction=reaction,
            conductor_charges={name: float(np.sum(reaction[mask])) for name, mask in named_masks.items()},
            energy=energy,
            residual_norm=residual_norm,
        )
        self.solution = result
        self.potential = result.potential
        self.fixed_mask = fixed
        return result

    def compute_electric_field(self) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.solution is None:
            raise SolverError("call solve() before computing the electric field.")
        if self.dim == 1:
            return self.solution.electric_field[:, 0]
        return self.solution.electric_field[:, 0], self.solution.electric_field[:, 1]

    def visualize(self, *, show: bool = True) -> object:
        if self.solution is None:
            raise SolverError("call solve() before visualize().")
        import matplotlib.pyplot as plt
        if self.dim == 1:
            x = self.coordinates[:, 0]
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
            axes[0].plot(x, self.solution.potential)
            axes[0].set_ylabel("Potential (V)")
            axes[1].plot(x, self.solution.electric_field[:, 0])
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("Electric field (V/m)")
            for axis in axes:
                axis.grid(True)
        else:
            import matplotlib.tri as mtri
            x, y = self.coordinates.T
            triangulation = mtri.Triangulation(x, y, self.elements)
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            image = axes[0].tricontourf(triangulation, self.solution.potential, levels=40)
            fig.colorbar(image, ax=axes[0], label="Potential (V)")
            axes[0].triplot(triangulation, color="k", alpha=0.12, linewidth=0.35)
            field = self.solution.electric_field
            vectors = axes[1].quiver(x, y, field[:, 0], field[:, 1], np.linalg.norm(field, axis=1))
            fig.colorbar(vectors, ax=axes[1], label="|E| (V/m)")
            for axis, title in zip(axes, ("FEM potential", "FEM electric field"), strict=True):
                axis.set_aspect("equal")
                axis.set_xlabel("x")
                axis.set_ylabel("y")
                axis.set_title(title)
        fig.tight_layout()
        if show:
            plt.show()
        return fig


__all__ = ["EPSILON_0", "ElectrostaticSolver"]

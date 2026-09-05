"""Geometry-first 1D/2D finite-element electrostatic solver."""

from __future__ import annotations

from fem_common import FEMSolverMixin
from fem_common.contracts import bounds

from collections.abc import Sequence
from dataclasses import replace
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .exceptions import GeometryError, NotDiscretizedError, SolverError
from .geometry import ChargeRegion, Circle, GeometryModel, Interval, MaterialRegion, Permittivity, Polygon, PotentialRegion, Rectangle
from .meshing import FEMMesh, discretize_1d, discretize_2d
from .results import ElectrostaticResult


EPSILON_0 = 8.854_187_812_8e-12
RegionInput: TypeAlias = Interval | Rectangle | Circle | Polygon | str


def _dimension(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) not in (1, 2):
        raise GeometryError("dim must be 1 or 2.")
    return int(value)






class ElectrostaticSolver(FEMSolverMixin):
    """Solve ``-div(epsilon_0 epsilon_r grad(phi)) = rho`` with P1 FEM.

    The constructor records only a physical domain and optional target mesh
    settings. Materials, charge regions, and fixed potentials remain continuous
    geometry until :meth:`mesh` or an automatically meshing :meth:`solve`.
    """

    def __init__(self, *, dim: int = 2, x_range: float | Sequence[float] = 1.0,
                 y_range: float | Sequence[float] | None = None,
                 background_epsilon: float | Sequence[float] | Sequence[Sequence[float]] = 1.0,
                 outer_potential: float | None = 0.0) -> None:
        self.dim = _dimension(dim)
        if self.dim == 1 and y_range is not None:
            raise GeometryError("y_range is only valid for dim=2.")
        x_span = bounds(x_range, "x_range")
        y_span = None if self.dim == 1 else bounds(1.0 if y_range is None else y_range, "y_range")
        self.geometry = GeometryModel(self.dim, x_span, y_span,
            Permittivity.from_input(background_epsilon, self.dim))
        if outer_potential is not None and not np.isfinite(float(outer_potential)):
            raise GeometryError("outer_potential must be finite or None.")
        self.outer_potential = None if outer_potential is None else float(outer_potential)
        self.mesh_data = None
        self._result = None
        self.potential = self.fixed_mask = None
        self._discretization_settings = None
        self.geometry.add_change_listener(self._geometry_changed)

    @property
    def x_range(self):
        return self.geometry.x_span

    @property
    def y_range(self):
        return self.geometry.y_span

    @property
    def result(self):
        return self._result


    def _geometry_changed(self) -> None:
        self.mesh_data = None
        self._clear_solution()

    def _clear_solution(self) -> None:
        self._result = None
        self.potential = None
        self.fixed_mask = None

    @property
    def coordinates(self) -> NDArray[np.float64]:
        if self.mesh_data is None:
            raise NotDiscretizedError("call mesh() before requesting mesh coordinates.")
        return self.mesh_data.nodes

    @property
    def elements(self) -> NDArray[np.int64]:
        if self.mesh_data is None:
            raise NotDiscretizedError("call mesh() before requesting mesh elements.")
        return self.mesh_data.elements



    def _shape(self, region):
        if isinstance(region, str):
            return region.lower()
        if isinstance(region, (Interval, Rectangle, Circle, Polygon)):
            return region
        raise GeometryError("region must be a geometry primitive or boundary name.")


    def set_potential(self, *, region: RegionInput, potential: float, name: str | None = None) -> PotentialRegion:
        return self.geometry.add_potential(self._shape(region), potential, name=name)

    def add_object(self, *, region: RegionInput, epsilon=1.0, name: str | None = None) -> MaterialRegion:
        shape = self._shape(region)
        if isinstance(shape, str):
            raise GeometryError("A material needs a physical region, not a boundary name.")
        return self.geometry.add_material(shape, Permittivity.from_input(epsilon, self.dim), name=name)

    def add_layer(self, *, x_range, epsilon=1.0, name: str | None = None):
        shape = Interval(bounds(x_range)) if self.dim == 1 else Rectangle(bounds(x_range), self.y_range)
        return self.add_object(region=shape, epsilon=epsilon, name=name)

    def add_rectangle(self, *, x_range, y_range, epsilon=1.0, name: str | None = None):
        if self.dim != 2:
            raise GeometryError("Rectangles require dim=2.")
        return self.add_object(region=Rectangle(bounds(x_range), bounds(y_range)), epsilon=epsilon, name=name)

    def add_circle(self, *, center, radius: float, epsilon=1.0, name: str | None = None):
        if self.dim != 2:
            raise GeometryError("Circles require dim=2.")
        return self.add_object(region=Circle(center, radius), epsilon=epsilon, name=name)

    def add_polygon(self, *, points, epsilon=1.0, name: str | None = None):
        if self.dim != 2:
            raise GeometryError("Polygons require dim=2.")
        return self.add_object(region=Polygon(points), epsilon=epsilon, name=name)


    def add_charge_density(self, *, region: RegionInput, density: float, name: str | None = None) -> ChargeRegion:
        shape = self._shape(region)
        if isinstance(shape, str):
            raise GeometryError("volume charge requires an area/interval, not a boundary name.")
        return self.geometry.add_charge(shape, density, name=name)

    def remove(self, item: MaterialRegion | PotentialRegion | ChargeRegion) -> None:
        self.geometry.remove(item)

    def _default_maximum_size(self) -> float:
        width = self.x_range[1] - self.x_range[0]
        return width / 16 if self.dim == 1 else min(width, self.y_range[1] - self.y_range[0]) / 6


    def mesh(
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
            self.mesh_data = discretize_1d(
                self.geometry,
                max_element_size=float(maximum),
                material_aware=material_aware,
                interface_refinement=interface_refinement,
                boundary_refinement=boundary_refinement,
            )
        else:
            self.mesh_data = discretize_2d(self.geometry, **settings)
        self._discretization_settings = settings
        self._clear_solution()
        return self.mesh_data

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
        assert self.mesh_data is not None
        nodes = self.mesh_data.nodes
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
        assert self.mesh_data is not None
        try:
            from skfem import Basis, BilinearForm, LinearForm, asm
            from skfem.element import ElementLineP1, ElementTriP1
            from skfem.helpers import grad
        except Exception as exc:
            raise SolverError("scikit-fem could not be imported; install the package dependencies.") from exc
        basis = Basis(self.mesh_data.mesh, ElementLineP1() if self.dim == 1 else ElementTriP1(), intorder=3)
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
            indices = np.flatnonzero(self.mesh_data.element_tags == tag)
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
        centers = self.mesh_data.nodes[self.mesh_data.elements].mean(axis=1)
        densities = self.geometry.charge_at(centers)
        rhs = np.zeros(basis.N, dtype=float)
        for density in np.unique(densities):
            if density != 0.0:
                indices = np.flatnonzero(densities == density)
                rhs += np.asarray(asm(source, basis.with_elements(indices), rho=float(density)), dtype=float)
        return matrix.tocsr(), rhs

    def _fields(self, potential: NDArray[np.float64]) -> tuple[
        NDArray[np.float64], NDArray[np.float64], float,
        NDArray[np.float64], NDArray[np.float64],
    ]:
        assert self.mesh_data is not None
        points, elements = self.mesh_data.nodes, self.mesh_data.elements
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
            mask = self.mesh_data.element_tags == tag
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
        # P1 gradients are constant per element and can jump at dielectric
        # interfaces.  Preserve these physical fields before nodal averaging.
        return nodal_e, nodal_d, energy, element_e, element_d

    def solve(
        self, *, linear_solver_tolerance: float = 1e-10, max_refinements: int = 2,
        adaptive_tolerance: float = 0.05, marking_fraction: float = 0.5,
        max_elements: int = 200_000,
    ) -> ElectrostaticResult:
        """Solve with bounded, solution-driven local refinement by default.

        Normal displacement jumps and the Poisson volume residual select
        cells by bulk marking.  ``adaptive_tolerance`` controls that relative
        indicator; ``tol`` independently controls the algebraic solve.  Inspect
        ``result.adaptive_history`` for the stopping reason.  Use
        ``adaptive=False`` to keep the supplied mesh exactly.
        """
        from .adaptive import flux_indicators, refine_marked

        for value, name, minimum in (
            (max_refinements, "max_refinements", 0), (max_elements, "max_elements", 1),
        ):
            try:
                valid = (not isinstance(value, (bool, np.bool_)) and np.isscalar(value)
                         and int(value) == value and int(value) >= minimum)
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise SolverError(f"{name} must be an integer of at least {minimum}.")
        for value, name in ((adaptive_tolerance, "adaptive_tolerance"),
                            (marking_fraction, "marking_fraction")):
            try:
                valid = (not isinstance(value, (bool, np.bool_)) and np.isscalar(value)
                         and np.isfinite(value) and value > 0.0)
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise SolverError(f"{name} must be finite and positive.")
        if marking_fraction > 1.0:
            raise SolverError("marking_fraction must be in (0, 1].")
        result = self._solve_once(linear_solver_tolerance)
        history = []
        for step in range(int(max_refinements) + 1):
            centers = result.coordinates[result.elements].mean(axis=1)
            indicators, error = flux_indicators(
                result.mesh, result.element_displacement_field,
                self.geometry.charge_at(centers), self.fixed_mask,
            )
            if not np.isfinite(error):
                raise SolverError("The adaptive flux indicator is non-finite.")
            record = {"elements": len(result.elements), "relative_indicator": error, "residual": error,
                      "marked_elements": 0, "status": "refined"}
            history.append(record)
            if error <= adaptive_tolerance:
                record["status"] = "tolerance"
                break
            if step == max_refinements:
                record["status"] = "refinement_limit"
                break
            if len(result.elements) >= max_elements:
                record["status"] = "element_limit"
                break
            ranked = np.argsort(-indicators, kind="stable")
            count = int(np.searchsorted(np.cumsum(indicators[ranked]),
                                       marking_fraction * indicators.sum())) + 1
            marked = ranked[:count]
            candidate = refine_marked(result.mesh, marked)
            if len(candidate.elements) > max_elements:
                record["status"] = "element_limit"
                break
            record["marked_elements"] = len(marked)
            previous_mask = self.fixed_mask
            self.mesh_data = candidate
            try:
                result = self._solve_once(linear_solver_tolerance)
            except Exception:
                self.mesh_data = result.mesh
                self._result = result
                self.potential = result.potential
                self.fixed_mask = previous_mask
                raise
        result = replace(result, adaptive_history=tuple(history))
        return self._finish_result(result)

    def _solve_once(self, tol: float) -> ElectrostaticResult:
        """Assemble and solve one linear electrostatic boundary-value problem."""
        tolerance = float(tol)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise SolverError("tol must be finite and positive.")
        if self.mesh_data is None:
            self.mesh(**(self._discretization_settings or {}))
        assert self.mesh_data is not None
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
        if not np.isfinite(residual_norm) or residual_norm > max(100.0 * tolerance, 1e-9):
            raise SolverError(f"FEM solve residual {residual_norm:.3e} exceeds tolerance.")
        electric, displacement, energy, element_e, element_d = self._fields(potential)
        result = ElectrostaticResult(
            mesh=self.mesh_data,
            potential=potential,
            electric_field=electric,
            displacement_field=displacement,
            reaction=reaction,
            conductor_charges={name: float(np.sum(reaction[mask])) for name, mask in named_masks.items()},
            energy=energy,
            residual_norm=residual_norm,
            element_electric_field=element_e,
            element_displacement_field=element_d,
        )
        self._result = result
        self.potential = result.potential
        self.fixed_mask = fixed
        return result

    def compute_electric_field(self) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self._result is None:
            raise SolverError("call solve() before computing the electric field.")
        if self.dim == 1:
            return self._result.electric_field[:, 0]
        return self._result.electric_field[:, 0], self._result.electric_field[:, 1]




__all__ = ["EPSILON_0", "ElectrostaticSolver"]

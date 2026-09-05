"""One-dimensional finite-element waveguide mode solver.

The public phasor convention is

``E(x, z, t) = Re{E(x) exp(+1j*omega*t - 1j*beta*z)}``.

For an x-stratified guide the Maxwell eigenproblem separates into scalar TE
(``E_y``) and TM (``H_y``) problems.  Both use continuous, first-order line
elements on a mesh which conforms to every material, boundary, and PML
interface.  The two spectra are merged before the requested modes are
returned, so ``num_modes`` means the total number of modes rather than a
number per polarization.
"""

from __future__ import annotations

from fem_common.contracts import ElectromagneticSolverMixin

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs, splu

from .constants import C_0, EPSILON_0, MU_0
from .exceptions import (
    BackendCapabilityError,
    ConfigurationError,
    GeometryError,
    NotDiscretizedError,
    SolverError,
    StaleDiscretizationError,
)
from .geometry import (
    BoundaryRegion,
    GeometryModel1D,
    Interval,
    PMLSpec,
    Region,
)
from .materials import Material, MaterialInput
from .meshing import FEMMesh1D, discretize_1d
from .results import Mode, ModeSet, SampledFields


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Polarization = Literal["TE", "TM"]


@dataclass(frozen=True, slots=True)
class _PolarizationSystem:
    """Reduced generalized eigenproblem ``A u = neff**2 B u``."""

    polarization: Polarization
    A: csr_matrix
    B: csr_matrix
    free_nodes: IntArray
    active_elements: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class _Candidate:
    polarization: Polarization
    eigenvalue: complex
    neff: complex
    vector: ComplexArray
    residual: float


def _positive_integer(value: int, name: str, *, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or not np.isscalar(value)
    ):
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(
            f"{name} must be an integer of at least {minimum}."
        ) from exc
    if result != value or result < minimum:
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
    return result


def _finite_positive(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
        raise ConfigurationError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be finite and positive.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ConfigurationError(f"{name} must be finite and positive.")
    return result


def _passive_material(material: Material, context: str) -> None:
    """Reject gain while documenting the package's constitutive sign."""

    values = (*material.eps_r, *material.mu_r)
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, *(abs(item) for item in values))
    if any(item.imag > tolerance for item in values):
        raise ConfigurationError(
            f"{context} is active under the +j*omega*t convention. "
            "Passive epsilon and mu must have non-positive imaginary parts."
        )
    if any(abs(item) <= np.finfo(float).tiny for item in material.eps_r):
        raise ConfigurationError("epsilon entries must be nonzero in the 1D FEM backend.")


class ModeSolver1D(ElectromagneticSolverMixin):
    """FEM-native mode solver for an x-stratified cross-section.

    Geometry is recorded continuously by the ``add_*`` methods.  Call
    :meth:`mesh` only after object placement, then call :meth:`solve`.

    Parameters
    ----------
    frequency:
        Frequency in hertz.
    x_range:
        Either a positive physical width (domain ``0..width``) or an explicit
        ``(minimum, maximum)`` pair in metres.
    """

    def __init__(self, *, frequency: float, x_range: float | Sequence[float], background_epsilon: MaterialInput=1.0, background_mu: MaterialInput=1.0) -> None:
        self.frequency = _finite_positive(frequency, "frequency")
        self.omega = 2.0 * np.pi * self.frequency
        self.k0 = self.omega / C_0

        background = Material(background_epsilon, background_mu)
        _passive_material(background, "background material")
        self.geometry = GeometryModel1D(x_range, background)
        self.model = self.geometry
        self.x_span = self.geometry.x_span
        self.x_range = self.x_span

        self.mesh_data: FEMMesh1D | None = None
        self._result: ModeSet | None = None
        self._quadrature_order = 4
        self._discretization_settings: dict[str, object] | None = None
        self._clear_result_views()
        self.geometry.add_change_listener(self._geometry_changed)

    def _clear_result_views(self) -> None:
        self._result = None
        self.modes = None
        self.neff = None
        self.beta = None
        self.eigenvalues = None
        self.propagation_constant = None
        self.attenuation_constant = None
        self.neff_TE = None
        self.neff_TM = None
        self.propagation_constant_TE = None
        self.propagation_constant_TM = None
        self.attenuation_constant_TE = None
        self.attenuation_constant_TM = None
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            if hasattr(self, name):
                delattr(self, name)

    def _geometry_changed(self) -> None:
        self.mesh_data = None
        self._clear_result_views()

    def add_layer(self, *, epsilon: MaterialInput, mu: MaterialInput, x_range: Sequence[float], name: str | None=None) -> Region:
        """Place a material interval and return its geometry handle.

        Later regions take precedence in overlap areas.  Material interfaces
        are inserted into the mesh exactly during :meth:`mesh`.
        """

        material = Material(epsilon, mu)
        _passive_material(material, "layer material")
        handle = self.geometry.add_region(Interval(tuple(x_range)), material, name=name)
        self._geometry_changed()
        return handle

    @staticmethod
    def _validate_components(components: object | None) -> None:
        if components is None:
            return
        if isinstance(components, str):
            selected = {components.lower()}
        else:
            try:
                selected = {str(item).lower() for item in components}  # type: ignore[union-attr]
            except TypeError as exc:
                raise ConfigurationError("components must be a component sequence.") from exc
        if selected != {"xx", "yy", "zz"}:
            raise BackendCapabilityError(
                "Component-selective PEC/PMC regions are not part of the scalar 1D FEM formulation."
            )

    def _add_boundary(
        self,
        kind: Literal["pec", "pmc"],
        x_range: Sequence[float] | None,
        *,
        components: object | None,
        name: str | None,
    ) -> BoundaryRegion | None:
        self._validate_components(components)
        if x_range is None:
            self.geometry.set_outer_boundary(kind)
            self._geometry_changed()
            return None
        interval = Interval(tuple(x_range))
        if interval.x[0] < self.x_span[0] or interval.x[1] > self.x_span[1]:
            raise GeometryError("Boundary region lies outside the solver domain.")
        handle = self.geometry.add_boundary(interval, kind, name=name)
        self._geometry_changed()
        return handle

    def add_pec(self, *, x_range: Sequence[float] | None=None, components: object | None=None, name: str | None=None) -> BoundaryRegion | None:
        """Add an opaque PEC interval, or set both outer walls to PEC."""

        return self._add_boundary("pec", x_range, components=components, name=name)

    def add_pmc(self, *, x_range: Sequence[float] | None=None, components: object | None=None, name: str | None=None) -> BoundaryRegion | None:
        """Add an opaque PMC interval, or set both outer walls to PMC."""

        return self._add_boundary("pmc", x_range, components=components, name=name)

    def add_impedance_surface(self, *, Zs: complex | None=None, preset: str | None=None, x_range: Sequence[float], name: str | None=None) -> BoundaryRegion:
        """Record a scalar impedance object.

        Geometry and meshing support the object now; assembly intentionally
        raises a capability error until the orientation-dependent 1D Robin
        form is validated for both polarizations.
        """

        if preset is not None:
            raise BackendCapabilityError(
                "Named surface-impedance presets are not yet available in fem_waveguide_modes."
            )
        if Zs is None:
            raise ConfigurationError("Supply Zs when preset is not specified.")
        impedance = complex(Zs)
        if not np.isfinite((impedance.real, impedance.imag)).all() or impedance.real < 0.0:
            raise ConfigurationError("Zs must be finite with nonnegative real part.")
        interval = Interval(tuple(x_range))
        if interval.x[0] < self.x_span[0] or interval.x[1] > self.x_span[1]:
            raise GeometryError("Impedance region lies outside the solver domain.")
        handle = self.geometry.add_boundary(
            interval,
            "impedance",
            impedance=impedance,
            name=name,
        )
        self._geometry_changed()
        return handle

    def add_pml(self, *, thickness: float, order: int=3, sigma_max: float=5.0, direction: str='all') -> PMLSpec:
        """Place a physical transformation-optics PML at selected x ends."""

        canonical = str(direction).lower()
        aliases = {"both": "all", "left": "x-", "right": "x+"}
        canonical = aliases.get(canonical, canonical)
        if canonical not in {"all", "x", "x-", "x+"}:
            raise ConfigurationError("1D PML direction must be 'all', 'x', 'x-', or 'x+'.")
        spec = PMLSpec(float(thickness), int(order), float(sigma_max), canonical)
        width = self.x_span[1] - self.x_span[0]
        occupied = 2.0 * spec.thickness if canonical in {"all", "x"} else spec.thickness
        if occupied >= width:
            raise ConfigurationError("The requested PML leaves no non-PML domain interior.")
        self.geometry.add_pml(spec)
        self._geometry_changed()
        return spec




    def set_outer_boundary(self, *, kind: str) -> None:
        """Set both transverse truncation walls to ``'pec'`` or ``'pmc'``."""

        self.geometry.set_outer_boundary(kind)
        self._geometry_changed()

    def remove(self, handle: Region | BoundaryRegion | PMLSpec) -> None:
        """Remove a previously returned geometry handle."""

        if isinstance(handle, PMLSpec):
            try:
                self.geometry.pmls.remove(handle)
            except ValueError as exc:
                raise GeometryError("The PML handle does not belong to this solver.") from exc
            self.geometry._changed()
        else:
            self.geometry.remove(handle)
        self._geometry_changed()

    @property
    def discretized(self) -> bool:
        """Whether a current, geometry-matching mesh is available."""

        return self.mesh_data is not None and self.mesh_data.geometry_revision == self.geometry.revision

    @property
    def result(self):
        return self._result




    @property
    def native_mesh(self) -> object:
        """Return the underlying scikit-fem line mesh."""

        return self._require_mesh().mesh

    def _mesh_impl(
        self,
        max_element_size: float | None = None,
        *,
        resolution: int | None = None,
        wavelength_elements: int = 4,
        material_aware: bool = True,
        element_order: int = 1,
        quadrature_order: int = 4,
    ) -> FEMMesh1D:
        """Create an interface-conforming first-order line mesh.

        By default, element density follows the local material wavenumber:
        intervals with a larger conservative material-index estimate receive
        smaller elements.  ``wavelength_elements`` additionally sets the minimum
        number of elements per shortest local material wavelength.  Set
        ``material_aware=False`` to use a spatially uniform target size.
        """

        quadrature = _positive_integer(
            quadrature_order, "quadrature_order", minimum=2
        )
        wavelength_count = _positive_integer(
            wavelength_elements, "wavelength_elements", minimum=4
        )
        if not isinstance(material_aware, (bool, np.bool_)):
            raise ConfigurationError("material_aware must be a boolean.")
        mesh = discretize_1d(
            self.geometry,
            resolution=resolution,
            max_element_size=max_element_size,
            element_order=element_order,
            vacuum_wavenumber=self.k0,
            wavelength_elements=wavelength_count,
            material_aware=bool(material_aware),
        )
        self.mesh_data = mesh
        self._quadrature_order = quadrature
        self._discretization_settings = {
            "max_element_size": max_element_size,
            "resolution": resolution,
            "wavelength_elements": wavelength_count,
            "material_aware": bool(material_aware),
            "element_order": element_order,
            "quadrature_order": quadrature,
        }
        self._clear_result_views()
        return mesh

    def refine(self, factor: float = 2.0) -> FEMMesh1D:
        """Remesh the current geometry with ``factor`` times the density.

        Refinement scales every active size control, preserves material-aware
        grading and exact interfaces, and invalidates any previously solved
        modes.  Repeated calls refine relative to the most recent mesh.
        """

        density_factor = _finite_positive(factor, "refinement factor")
        if not np.isfinite(density_factor) or density_factor <= 1.0:
            raise ConfigurationError(
                "refinement factor must be finite and greater than one."
            )
        self._require_mesh()
        if self._discretization_settings is None:  # defensive invariant
            raise NotDiscretizedError("Call mesh() before refine().")

        settings = dict(self._discretization_settings)
        resolution = settings["resolution"]
        maximum = settings["max_element_size"]
        if resolution is None and maximum is None:
            resolution = int(np.ceil(24 * density_factor))
        elif resolution is not None:
            resolution = int(np.ceil(int(resolution) * density_factor))
        if maximum is not None:
            maximum = float(maximum) / density_factor

        wavelength_elements = int(
            np.ceil(int(settings["wavelength_elements"]) * density_factor)
        )
        return self._mesh_impl(
            max_element_size=maximum,
            resolution=resolution,
            wavelength_elements=wavelength_elements,
            material_aware=bool(settings["material_aware"]),
            element_order=int(settings["element_order"]),
            quadrature_order=int(settings["quadrature_order"]),
        )

    def _require_mesh(self) -> FEMMesh1D:
        if self.mesh_data is None:
            raise NotDiscretizedError(
                "Call mesh() after placing all geometry and before solve()."
            )
        if self.mesh_data.geometry_revision != self.geometry.revision:
            raise StaleDiscretizationError(
                "Geometry changed after discretization; call mesh() again."
            )
        return self.mesh_data

    def _transformed_material_at(
        self, x: FloatArray
    ) -> tuple[ComplexArray, ComplexArray]:
        return self.geometry.transformed_material_at(x)

    def _boundary_masks(
        self, nodes: FloatArray
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
        centres = 0.5 * (nodes[:-1] + nodes[1:])
        blocked = np.zeros(centres.size, dtype=bool)
        pec_nodes = np.zeros(nodes.size, dtype=bool)
        pmc_nodes = np.zeros(nodes.size, dtype=bool)
        tolerance = 128.0 * np.finfo(float).eps * max(
            1.0, abs(self.x_span[0]), abs(self.x_span[1])
        )

        for boundary in self.geometry.boundaries:
            if boundary.kind == "impedance":
                raise BackendCapabilityError(
                    "1D FEM impedance-surface assembly is not implemented yet."
                )
            if boundary.kind not in {"pec", "pmc"}:
                raise ConfigurationError(f"Unknown boundary kind {boundary.kind!r}.")
            interval = boundary.shape
            if not isinstance(interval, Interval):
                raise ConfigurationError("1D boundaries must use Interval geometry.")
            mask = (centres > interval.x[0] - tolerance) & (
                centres < interval.x[1] + tolerance
            )
            blocked |= mask
            endpoint_mask = np.isclose(nodes, interval.x[0], rtol=0.0, atol=tolerance)
            endpoint_mask |= np.isclose(nodes, interval.x[1], rtol=0.0, atol=tolerance)
            if boundary.kind == "pec":
                pec_nodes |= endpoint_mask
            else:
                pmc_nodes |= endpoint_mask

        if np.all(blocked):
            raise SolverError("Boundary objects remove the complete 1D solve domain.")
        return blocked, pec_nodes, pmc_nodes

    def _assemble_systems(self) -> tuple[_PolarizationSystem, _PolarizationSystem]:
        mesh = self._require_mesh()
        nodes = np.asarray(mesh.nodes, dtype=float)
        widths_xi = self.k0 * np.diff(nodes)
        # Gauss points integrate the varying transformation-optics coefficients
        # inside PML elements.  Midpoint material sampling is insufficient even
        # for the P1 mass form, whose shape-function products are quadratic.
        points, weights = np.polynomial.legendre.leggauss(
            (self._quadrature_order + 2) // 2
        )
        shapes = np.stack(((1.0 - points) / 2.0, (1.0 + points) / 2.0))
        coordinates = nodes[:-1, None] + np.diff(nodes)[:, None] * shapes[1]
        eps, mu = self._transformed_material_at(coordinates)
        if np.any(np.abs(eps) <= np.finfo(float).tiny) or np.any(
            np.abs(mu) <= np.finfo(float).tiny
        ):
            raise ConfigurationError("epsilon and mu must be nonzero throughout the mesh.")

        blocked, pec_nodes, pmc_nodes = self._boundary_masks(nodes)
        active_elements = ~blocked
        active_nodes = np.zeros(nodes.size, dtype=bool)
        active_indices = np.flatnonzero(active_elements)
        active_nodes[active_indices] = True
        active_nodes[active_indices + 1] = True

        dofs = np.column_stack((active_indices, active_indices + 1))
        rows = np.repeat(dofs, 2, axis=1).ravel()
        columns = np.tile(dofs, (1, 2)).ravel()
        h = widths_xi[active_indices]
        mass_weights = np.einsum("q,iq,jq->qij", weights / 2.0, shapes, shapes)
        gradient_product = np.asarray(((1.0, -1.0), (-1.0, 1.0)))

        systems: list[_PolarizationSystem] = []
        for polarization in ("TE", "TM"):
            if polarization == "TE":
                mass_a = eps[1]
                stiffness = 1.0 / mu[2]
                mass_b = 1.0 / mu[0]
                constrained = pec_nodes.copy()
                if self.geometry.outer_boundary == "pec":
                    constrained[[0, -1]] = True
            else:
                mass_a = mu[1]
                stiffness = 1.0 / eps[2]
                mass_b = 1.0 / eps[0]
                constrained = pmc_nodes.copy()
                if self.geometry.outer_boundary == "pmc":
                    constrained[[0, -1]] = True

            local_a = h[:, None, None] * np.einsum(
                "eq,qij->eij", mass_a[active_indices], mass_weights
            )
            local_a -= (
                (stiffness[active_indices] @ (weights / 2.0)) / h
            )[:, None, None] * gradient_product
            local_b = h[:, None, None] * np.einsum(
                "eq,qij->eij", mass_b[active_indices], mass_weights
            )

            free = np.flatnonzero(active_nodes & ~constrained).astype(np.int64)
            if free.size < 2:
                raise SolverError(
                    f"The {polarization} problem has fewer than two unconstrained nodal DOFs."
                )
            shape = (nodes.size, nodes.size)
            full_a = coo_matrix((local_a.ravel(), (rows, columns)), shape=shape).tocsr()
            full_b = coo_matrix((local_b.ravel(), (rows, columns)), shape=shape).tocsr()
            systems.append(
                _PolarizationSystem(
                    polarization=polarization,
                    A=full_a[free][:, free].tocsr(),
                    B=full_b[free][:, free].tocsr(),
                    free_nodes=free,
                    active_elements=np.array(active_elements, copy=True),
                )
            )
        return systems[0], systems[1]

    @staticmethod
    def _select_neff(eigenvalue: complex) -> complex:
        root = complex(np.sqrt(complex(eigenvalue)))
        tolerance = 1e-12 * max(1.0, abs(root))
        if root.real < -tolerance or (
            abs(root.real) <= tolerance and root.imag > tolerance
        ):
            root = -root
        real = 0.0 if abs(root.real) <= tolerance else root.real
        imag = 0.0 if abs(root.imag) <= tolerance else root.imag
        return complex(real, imag)

    @staticmethod
    def _eigen_residual(
        system: _PolarizationSystem,
        eigenvalue: complex,
        vector: ComplexArray,
    ) -> float:
        left = system.A @ vector
        right = complex(eigenvalue) * (system.B @ vector)
        denominator = float(np.linalg.norm(left) + np.linalg.norm(right))
        residual = left - right
        return float(np.linalg.norm(residual) if denominator == 0.0 else np.linalg.norm(residual) / denominator)

    def _eigenpairs(
        self,
        system: _PolarizationSystem,
        sigma: complex,
        requested: int,
        tolerance: float,
        dense_limit: int,
    ) -> tuple[ComplexArray, ComplexArray, str]:
        size = system.A.shape[0]
        candidate_count = min(size, max(2 * requested + 6, 10))
        use_dense = size <= dense_limit or candidate_count >= size - 1
        if use_dense:
            try:
                values, vectors = linalg.eig(
                    system.A.toarray(),
                    system.B.toarray(),
                    check_finite=False,
                )
                method = "dense-generalized"
            except linalg.LinAlgError as exc:
                # Some LAPACK GGEV builds fail to converge for the complex-
                # symmetric mass matrices produced by transformation-optics
                # PMLs even though shift-invert remains well conditioned.
                if size <= 3:
                    raise SolverError(
                        f"Dense {system.polarization} eigenproblem failed and "
                        "is too small for sparse fallback."
                    ) from exc
                use_dense = False
        if not use_dense:
            count = min(candidate_count, size - 2)
            # Avoid an exactly singular factorization when the requested shift
            # lands on a mode (the homogeneous TEM case does this routinely).
            # An offset near machine precision leaves the factorization so ill
            # conditioned that otherwise accurate neighbouring eigenvectors
            # can fail the residual filter; 1e-6 is still negligible on the
            # dimensionless neff**2 scale while keeping shift-invert stable.
            shift = complex(sigma) + 1e-6j * max(1.0, abs(sigma))
            rng = np.random.default_rng(20260828)
            initial = rng.standard_normal(size) + 1j * rng.standard_normal(size)
            try:
                factor = splu(csc_matrix(system.A - shift * system.B))

                def apply_shift_invert(vector: ComplexArray) -> ComplexArray:
                    return np.asarray(
                        factor.solve(system.B @ vector), dtype=np.complex128
                    )

                operator = LinearOperator(
                    (size, size), matvec=apply_shift_invert, dtype=np.complex128
                )
                theta, vectors = eigs(
                    operator,
                    k=count,
                    which="LM",
                    v0=initial,
                    tol=tolerance,
                    maxiter=max(3000, 20 * size),
                )
            except ArpackNoConvergence as exc:
                if exc.eigenvalues is None or exc.eigenvectors is None:
                    raise SolverError(
                        f"Sparse {system.polarization} mode iteration did not converge."
                    ) from exc
                theta = exc.eigenvalues
                vectors = exc.eigenvectors
            except (RuntimeError, ValueError) as exc:
                raise SolverError(
                    f"Sparse {system.polarization} eigenproblem failed near neff**2={sigma!r}."
                ) from exc
            valid = np.abs(theta) > np.finfo(float).eps
            values = shift + 1.0 / theta[valid]
            vectors = vectors[:, valid]
            method = "sparse-shift-invert"

        finite = np.isfinite(values)
        values = np.asarray(values[finite], dtype=np.complex128)
        vectors = np.asarray(vectors[:, finite], dtype=np.complex128)
        order = np.argsort(np.abs(values - sigma))
        return values[order], vectors[:, order], method

    def _default_neff_guess(self, active_elements: NDArray[np.bool_]) -> complex:
        mesh = self._require_mesh()
        centres = 0.5 * (mesh.nodes[:-1] + mesh.nodes[1:])
        eps, mu = self.geometry.material_at(centres)
        estimates = np.concatenate(
            (
                eps[1, active_elements] * mu[0, active_elements],
                mu[1, active_elements] * eps[0, active_elements],
            )
        )
        if estimates.size == 0:
            raise SolverError("No active material elements remain.")
        roots = np.asarray([self._select_neff(value) for value in estimates])
        return complex(roots[int(np.argmax(np.abs(roots)))])

    def _is_lossless_closed_model(self) -> bool:
        """Whether exact generalized eigenvalues should be real."""

        if self.geometry.pmls:
            return False
        materials = [
            self.geometry.background,
            *(region.material for region in self.geometry.regions),
        ]
        return all(
            value.imag == 0.0
            for material in materials
            for value in (*material.eps_r, *material.mu_r)
        )

    def _mode_fields(
        self,
        candidate: _Candidate,
        system: _PolarizationSystem,
        index: int,
    ) -> Mode:
        mesh = self._require_mesh()
        nodes = np.asarray(mesh.nodes, dtype=float)
        cells = np.column_stack(
            (np.arange(nodes.size - 1, dtype=np.int64), np.arange(1, nodes.size, dtype=np.int64))
        )
        centres = 0.5 * (nodes[:-1] + nodes[1:])
        widths = np.diff(nodes)
        eps_transformed, mu_transformed = self._transformed_material_at(centres)
        eps_physical, mu_physical = self.geometry.material_at(centres)
        material_index = np.sqrt(
            np.max(np.abs(eps_physical * mu_physical), axis=0)
        )

        nodal = np.zeros(nodes.size, dtype=np.complex128)
        nodal[system.free_nodes] = candidate.vector
        primary = 0.5 * (nodal[:-1] + nodal[1:])
        derivative = np.diff(nodal) / widths
        primary[~system.active_elements] = 0.0
        derivative[~system.active_elements] = 0.0

        zeros = np.zeros(centres.size, dtype=np.complex128)
        beta = self.k0 * candidate.neff
        if candidate.polarization == "TE":
            ex, ey, ez = zeros.copy(), primary, zeros.copy()
            hx = -beta * ey / (self.omega * MU_0 * mu_transformed[0])
            hy = zeros.copy()
            hz = 1j * derivative / (self.omega * MU_0 * mu_transformed[2])
            longitudinal_flux = -ey * np.conj(hx)
        else:
            hx, hy, hz = zeros.copy(), primary, zeros.copy()
            ex = beta * hy / (self.omega * EPSILON_0 * eps_transformed[0])
            ey = zeros.copy()
            ez = -1j * derivative / (self.omega * EPSILON_0 * eps_transformed[2])
            longitudinal_flux = ex * np.conj(hy)

        physical_cells = system.active_elements & self._non_pml_elements(centres)
        complex_power = 0.5 * np.sum(widths[physical_cells] * longitudinal_flux[physical_cells])
        real_power = float(np.real(complex_power))
        power_scale = max(float(np.max(np.abs(primary))), np.finfo(float).tiny)
        if real_power > 1e-12 * power_scale**2 * max(self.x_range[1] - self.x_range[0], np.finfo(float).tiny):
            scale = 1.0 / np.sqrt(real_power)
            normalization = "unit-power"
        else:
            norm = float(np.sqrt(np.sum(widths[system.active_elements] * np.abs(primary[system.active_elements]) ** 2)))
            if norm <= np.finfo(float).tiny:
                raise SolverError("An eigenvector has zero physical norm.")
            scale = 1.0 / norm
            normalization = "l2"

        phase_source = primary
        pivot = phase_source[int(np.argmax(np.abs(phase_source)))]
        phase = 1.0 + 0.0j if abs(pivot) == 0.0 else np.exp(-1j * np.angle(pivot))
        factor = scale * phase
        values = {
            "Ex": np.asarray(ex * factor, dtype=np.complex128),
            "Ey": np.asarray(ey * factor, dtype=np.complex128),
            "Ez": np.asarray(ez * factor, dtype=np.complex128),
            "Hx": np.asarray(hx * factor, dtype=np.complex128),
            "Hy": np.asarray(hy * factor, dtype=np.complex128),
            "Hz": np.asarray(hz * factor, dtype=np.complex128),
        }
        normalized_power = float(real_power * abs(scale) ** 2)
        sampled = SampledFields(
            coordinates=centres[:, np.newaxis],
            values=values,
            dimension=1,
            mesh_points=nodes[:, np.newaxis],
            mesh_cells=cells,
            material=np.asarray(eps_physical[1], dtype=np.complex128),
            metadata={
                "x_nodes": np.array(nodes, copy=True),
                "active_elements": np.array(system.active_elements, copy=True),
                "material_index": np.asarray(material_index, dtype=float),
                "phasor_convention": "exp(+1j*omega*t - 1j*beta*z)",
            },
        )
        return Mode(
            neff=candidate.neff,
            beta=complex(beta),
            fields=sampled,
            index=index,
            polarization=candidate.polarization,
            eigenvalue=candidate.eigenvalue,
            power=normalized_power,
            normalization=normalization,
            residual=candidate.residual,
            divergence_residual=0.0 if candidate.polarization == "TE" else None,
            metadata={
                "nodal_primary": np.asarray(nodal * factor),
                "attenuation": float(-np.imag(beta)),
                "complex_power_before_normalization": complex(complex_power),
            },
        )

    def _non_pml_elements(self, centres: FloatArray) -> NDArray[np.bool_]:
        result = np.ones(centres.size, dtype=bool)
        xmin, xmax = self.x_span
        for pml in self.geometry.pmls:
            if pml.direction in {"x-", "x", "all"}:
                result &= centres >= xmin + pml.thickness
            if pml.direction in {"x+", "x", "all"}:
                result &= centres <= xmax - pml.thickness
        return result

    def solve(self, *, neff_guess: complex | None=None, num_modes: int=4, eigensolver_tolerance: float=1e-10, residual_tolerance: float=1e-07, dense_limit: int=450, max_refinements: int=2, adaptive_tolerance: float=0.05):
        """Solve and return modes with separate algebraic and adaptive controls."""
        from .adaptive import solve_1d
        options = {'neff_guess': neff_guess, 'num_modes': num_modes, 'tol': eigensolver_tolerance, 'residual_tolerance': residual_tolerance, 'dense_limit': dense_limit}
        return self._finish_result(solve_1d(self, (), options, max_refinements, adaptive_tolerance))


    def _solve_once(
        self,
        neff_guess: complex | None = None,
        num_modes: int | None = None,
        *,
        tol: float = 1e-10,
        residual_tolerance: float = 1e-7,
        dense_limit: int = 450,
    ) -> ModeSet:
        """Solve, merge, normalize, and return the requested TE/TM modes."""

        requested = _positive_integer(num_modes, "num_modes")
        tolerance = _finite_positive(tol, "tol")
        residual_limit = _finite_positive(residual_tolerance, "residual_tolerance")
        dense_threshold = _positive_integer(dense_limit, "dense_limit", minimum=2)
        te_system, tm_system = self._assemble_systems()

        supplied_guess = None if neff_guess is None else complex(neff_guess)
        if supplied_guess is None:
            guess = self._default_neff_guess(te_system.active_elements)
        else:
            guess = complex(supplied_guess)
            if not np.isfinite((guess.real, guess.imag)).all():
                raise ConfigurationError("neff_guess must be finite.")
        sigma = guess**2

        candidates: list[_Candidate] = []
        methods: dict[str, str] = {}
        rejected = 0
        for system in (te_system, tm_system):
            values, vectors, method = self._eigenpairs(
                system,
                sigma,
                requested,
                tolerance,
                dense_threshold,
            )
            methods[system.polarization] = method
            for eigenvalue, vector in zip(values, vectors.T, strict=True):
                eigenvalue = complex(eigenvalue)
                if self._is_lossless_closed_model() and abs(eigenvalue.imag) <= (
                    1e-8 * max(1.0, abs(eigenvalue.real))
                ):
                    eigenvalue = complex(eigenvalue.real)
                norm = float(np.linalg.norm(vector))
                if norm <= np.finfo(float).tiny:
                    continue
                normalized = np.asarray(vector / norm, dtype=np.complex128)
                residual = self._eigen_residual(system, eigenvalue, normalized)
                if not np.isfinite(residual) or residual > residual_limit:
                    rejected += 1
                    continue
                candidates.append(
                    _Candidate(
                        polarization=system.polarization,
                        eigenvalue=eigenvalue,
                        neff=self._select_neff(eigenvalue),
                        vector=normalized,
                        residual=residual,
                    )
                )

        candidates.sort(
            key=lambda item: (
                abs(item.neff - guess),
                0 if item.polarization == "TM" else 1,
                -item.neff.real,
            )
        )
        selected = candidates[:requested]
        if len(selected) < requested:
            raise SolverError(
                f"Requested {requested} modes near neff={guess!r}, but only "
                f"{len(selected)} passed the residual tolerance; rejected={rejected}."
            )

        system_by_polarization = {"TE": te_system, "TM": tm_system}
        mode_objects = tuple(
            self._mode_fields(
                candidate,
                system_by_polarization[candidate.polarization],
                index,
            )
            for index, candidate in enumerate(selected)
        )
        solution = ModeSet(
            mode_objects,
            frequency=self.frequency,
            k0=self.k0,
            dimension=1,
            backend="fem",
            metadata={
                "methods": methods,
                "neff_guess": guess,
                "mesh_info": self._require_mesh().info,
                "quadrature_order": self._quadrature_order,
                "phasor_convention": "exp(+1j*omega*t - 1j*beta*z)",
            },
        )
        self._result = solution
        self.modes = solution
        self.neff = np.asarray([mode.neff for mode in solution], dtype=np.complex128)
        self.beta = np.asarray([mode.beta for mode in solution], dtype=np.complex128)
        self.eigenvalues = np.asarray(
            [mode.eigenvalue for mode in solution], dtype=np.complex128
        )
        self.propagation_constant = np.real(self.neff)
        self.attenuation_constant = -np.imag(self.neff)
        for polarization in ("TE", "TM"):
            subset = [mode for mode in solution if mode.polarization == polarization]
            setattr(
                self,
                f"neff_{polarization}",
                np.asarray([mode.neff for mode in subset], dtype=np.complex128),
            )
            setattr(
                self,
                f"propagation_constant_{polarization}",
                np.asarray([mode.neff.real for mode in subset], dtype=float),
            )
            setattr(
                self,
                f"attenuation_constant_{polarization}",
                np.asarray([-mode.neff.imag for mode in subset], dtype=float),
            )
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            setattr(
                self,
                component,
                np.column_stack([mode.fields.values[component] for mode in solution]),
            )
        return solution

    def mesh(self, *, max_element_size: float | None=None, resolution: int | None=None, wavelength_elements: int=4, material_aware: bool=True, element_order: int=1, quadrature_order: int=4):
        """Build the initial mesh; solve() may subsequently refine it."""
        settings = {'max_element_size': max_element_size, 'resolution': resolution, 'wavelength_elements': wavelength_elements, 'material_aware': material_aware, 'element_order': element_order, 'quadrature_order': quadrature_order}
        mesh = self._mesh_impl(**settings)
        self._mesh_settings = settings
        self._result = None
        return mesh

    _physical_axes = ('x',)






__all__ = ["ModeSolver1D"]

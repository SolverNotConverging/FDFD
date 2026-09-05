"""Full-vector fixed-frequency modes of an x-stratified 2.5D waveguide.

The solver follows WaveFEM's convention

``E(x, y, z, t) = Re{e(x) exp(i ky y + i beta z - i omega t)}``.

It uses the dimensionless coordinates ``xi = k0*x``, ``eta = ky/k0``, and
``lambda = beta/k0``.  The electric field is discretized with the one-
dimensional trace of the scattering solver's Nedelec--H1 space:

* ``E_x`` is piecewise constant and may jump across an x-normal interface;
* ``E_y`` and ``E_z`` are continuous piecewise linear fields.

The resulting analytic quadratic pencil is

``(A0 + lambda*A1 + lambda**2*A2) e = 0``.

The coefficient matrices are assembled separately.  In particular, the code
never conjugates the eigenvalue while constructing an energy form; doing so
would create a non-analytic dependence on ``conj(lambda)``.

The outer transverse truncation remains explicit PEC or PMC.  An optional
transformation-optics PML may be placed in front of a PEC wall; its diagonal
constitutive tensors are included directly in the same full-vector pencil.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import Literal, TypeAlias, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from scipy.sparse import bmat, csc_matrix, csr_matrix, diags, eye
from scipy.sparse.linalg import (
    ArpackNoConvergence,
    LinearOperator,
    eigs,
    norm as sparse_norm,
    splu,
)
from skfem import Basis, BilinearForm, MeshLine, asm
from skfem.element import ElementLineP0, ElementLineP1

from .constants import ETA_0
from .exceptions import ConfigurationError, ModeSolverError
from .frequency import Frequency, resolve_frequency
from .materials import Material
from .pml import PML


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoundaryKind: TypeAlias = Literal["pec", "pmc"]
RequestedDirection: TypeAlias = Literal["forward", "backward", "all"]
ModeDirection: TypeAlias = Literal[
    "forward",
    "backward",
    "right-decaying",
    "left-decaying",
    "indeterminate",
]
ModeClassification: TypeAlias = Literal["propagating", "evanescent"]


def _finite_span(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ConfigurationError(f"{name} must contain exactly two values.")
    lo, hi = float(value[0]), float(value[1])
    if not np.isfinite((lo, hi)).all() or not lo < hi:
        raise ConfigurationError(
            f"{name} must be finite and strictly increasing; got {value!r}."
        )
    return lo, hi


def _positive_integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ConfigurationError(
            f"{name} must be an integer of at least {minimum}."
        )
    return int(value)


@dataclass(frozen=True, slots=True)
class Layer:
    """One non-overlapping material interval in a straight cross-section."""

    x: tuple[float, float]
    material: Material
    name: str


@dataclass(frozen=True, slots=True)
class PECBoundary:
    """One zero-thickness, z-invariant internal PEC sheet."""

    x: float
    name: str


@dataclass(slots=True)
class CrossSection:
    """A one-dimensional material profile for a z-uniform lead.

    Parameters
    ----------
    x_span:
        Finite transverse computational interval in metres.
    background:
        Isotropic material outside all explicitly added layers.
    boundary:
        Explicit transverse truncation.  ``None`` is rejected by
        :class:`ModeSolver` to prevent an open guide from silently becoming a
        PEC box.
    pml:
        Optional symmetric transverse PML placed inside both ends of
        ``x_span``.  The PML is terminated by the explicit outer PEC wall.
    pec_boundaries:
        Zero-thickness internal PEC sheets at physical x coordinates.  Each
        sheet constrains tangential ``E_y`` and ``E_z`` while retaining the
        normal ``E_x`` trace on both sides.
    """

    x_span: tuple[float, float]
    background: Material = field(default_factory=Material)
    boundary: BoundaryKind | None = None
    layers: list[Layer] = field(default_factory=list)
    pml: PML | None = None
    pec_boundaries: list[PECBoundary] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x_span = _finite_span(self.x_span, "x_span")
        if not isinstance(self.background, Material):
            raise ConfigurationError("background must be a Material instance.")
        if self.boundary not in (None, "pec", "pmc"):
            raise ConfigurationError("boundary must be None, 'pec', or 'pmc'.")
        if self.pml is not None:
            if not isinstance(self.pml, PML):
                raise ConfigurationError("pml must be a PML instance or None.")
            if self.boundary not in (None, "pec"):
                raise ConfigurationError(
                    "A transverse mode PML is terminated by an outer PEC; "
                    "set boundary='pec'."
                )
            if 2.0 * self.pml.thickness >= self.x_span[1] - self.x_span[0]:
                raise ConfigurationError(
                    "Two transverse PMLs leave no non-PML cross-section interior."
                )
        raw_pec_boundaries = tuple(self.pec_boundaries)
        self.pec_boundaries = []
        for item in raw_pec_boundaries:
            if not isinstance(item, PECBoundary):
                raise ConfigurationError(
                    "pec_boundaries must contain PECBoundary instances; "
                    "prefer CrossSection.add_pec()."
                )
            self.add_pec(x=item.x, name=item.name)

    def add_pec(self, *, x: float, name: str | None = None) -> PECBoundary:
        """Add and return a zero-thickness internal PEC sheet.

        The coordinate must lie strictly inside :attr:`x_span`.  Mode
        assembly removes the nodal ``E_y`` and ``E_z`` degrees of freedom at
        the conforming sheet node.  Cellwise ``E_x`` remains free because it
        is normal to the sheet and may be discontinuous across it.
        """

        if isinstance(x, (bool, np.bool_)):
            raise ConfigurationError("PEC x must be a finite real coordinate.")
        try:
            coordinate = float(x)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError("PEC x must be a finite real coordinate.") from exc
        if not np.isfinite(coordinate):
            raise ConfigurationError("PEC x must be a finite real coordinate.")
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, *(abs(value) for value in self.x_span)
        )
        if not self.x_span[0] + tolerance < coordinate < self.x_span[1] - tolerance:
            raise ConfigurationError(
                "An internal PEC x coordinate must lie strictly inside x_span; "
                f"got x={coordinate!r} for x_span={self.x_span}."
            )
        if any(
            abs(coordinate - existing.x) <= tolerance
            for existing in self.pec_boundaries
        ):
            raise ConfigurationError(
                f"An internal PEC boundary already exists at x={coordinate!r}."
            )
        boundary_name = name or f"pec_{len(self.pec_boundaries) + 1}"
        if not boundary_name or any(
            existing.name == boundary_name for existing in self.pec_boundaries
        ):
            raise ConfigurationError(f"PEC boundary name {boundary_name!r} is not unique.")
        result = PECBoundary(coordinate, boundary_name)
        self.pec_boundaries.append(result)
        self.pec_boundaries.sort(key=lambda item: item.x)
        return result

    def add_layer(
        self,
        *,
        x: Sequence[float],
        material: Material,
        name: str | None = None,
    ) -> Layer:
        """Add a mesh-conforming material interval and return it."""

        interval = _finite_span(x, "layer x")
        if not isinstance(material, Material):
            raise ConfigurationError("layer material must be a Material instance.")
        if interval[0] < self.x_span[0] or interval[1] > self.x_span[1]:
            raise ConfigurationError(
                f"Layer x={interval} lies outside cross-section x={self.x_span}."
            )
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, *(abs(value) for value in self.x_span)
        )
        for existing in self.layers:
            overlap = min(interval[1], existing.x[1]) - max(
                interval[0], existing.x[0]
            )
            if overlap > tolerance:
                raise ConfigurationError(
                    f"Layer x={interval} overlaps existing layer {existing.name!r}."
                )
        layer_name = name or f"layer_{len(self.layers) + 1}"
        if not layer_name or any(layer.name == layer_name for layer in self.layers):
            raise ConfigurationError(f"Layer name {layer_name!r} is not unique.")
        layer = Layer(interval, material, layer_name)
        self.layers.append(layer)
        return layer

    @property
    def interfaces(self) -> tuple[float, ...]:
        """Sorted material, PEC, and PML interfaces, including outer bounds."""

        points = {
            self.x_span[0],
            self.x_span[1],
            *(boundary.x for boundary in self.pec_boundaries),
        }
        for layer in self.layers:
            points.update(layer.x)
        if self.pml is not None:
            points.update(
                (
                    self.x_span[0] + self.pml.thickness,
                    self.x_span[1] - self.pml.thickness,
                )
            )
        return tuple(sorted(points))

    def material_at(self, x: ArrayLike) -> tuple[ComplexArray, ComplexArray]:
        """Evaluate scalar relative ``(epsilon, mu)`` at physical x points."""

        coordinates = np.asarray(x, dtype=float)
        eps_r = np.full(coordinates.shape, self.background.eps_r, dtype=np.complex128)
        mu_r = np.full(coordinates.shape, self.background.mu_r, dtype=np.complex128)
        for layer in self.layers:
            mask = (coordinates >= layer.x[0]) & (coordinates <= layer.x[1])
            eps_r[mask] = layer.material.eps_r
            mu_r[mask] = layer.material.mu_r
        return eps_r, mu_r

    def diagonal_material_at(
        self, x: ArrayLike, *, k_reference: float
    ) -> tuple[ComplexArray, ComplexArray]:
        """Evaluate diagonal relative ``(epsilon, mu)`` tensors.

        Arrays are returned with leading component order ``(x, y, z)``.  For
        a transverse stretch ``S = diag(sx, 1, 1)``, transformation optics
        gives ``material * (1/sx, sx, sx)``.  With no PML this is exactly the
        scalar material expanded onto three equal diagonal components.
        """

        coordinates = np.asarray(x, dtype=float)
        eps_r, mu_r = self.material_at(coordinates)
        sx = np.ones(coordinates.shape, dtype=np.complex128)
        if self.pml is not None:
            left_depth = self.x_span[0] + self.pml.thickness - coordinates
            right_depth = coordinates - (self.x_span[1] - self.pml.thickness)
            depth = np.maximum(np.maximum(left_depth, right_depth), 0.0)
            sx = self.pml.stretch(depth, k_reference)
        factors = np.stack((1.0 / sx, sx, sx), axis=0)
        return factors * eps_r[np.newaxis, ...], factors * mu_r[np.newaxis, ...]


@dataclass(frozen=True, slots=True)
class ModeFEMSystem:
    """Assembled dimensionless quadratic pencil and diagnostic operators."""

    x_nodes: FloatArray
    xi_nodes: FloatArray
    A0: csr_matrix
    A1: csr_matrix
    A2: csr_matrix
    free_dofs: IntArray
    full_size: int
    ex_slice: slice
    ey_slice: slice
    ez_slice: slice
    divergence_x: csr_matrix
    epsilon_mass: csr_matrix
    epsilon_mass_z: csr_matrix
    divergence_test_dofs: IntArray
    frequency: Frequency
    ky: float
    eta: float
    boundary: BoundaryKind

    @property
    def ndofs(self) -> int:
        """Number of unconstrained electric-field degrees of freedom."""

        return int(self.A0.shape[0])

    @property
    def elements(self) -> int:
        return int(self.x_nodes.size - 1)

    def polynomial(self, neff: complex) -> csr_matrix:
        """Return ``A0 + neff*A1 + neff**2*A2``."""

        value = complex(neff)
        return self.A0 + value * self.A1 + value**2 * self.A2

    def expand(self, vector: ArrayLike) -> ComplexArray:
        """Expand a reduced PEC/PMC vector into physical component blocks."""

        reduced = np.asarray(vector, dtype=np.complex128)
        if reduced.shape != (self.ndofs,):
            raise ValueError(
                f"mode vector must have shape ({self.ndofs},), got {reduced.shape}."
            )
        full = np.zeros(self.full_size, dtype=np.complex128)
        full[self.free_dofs] = reduced
        return full

    def relative_hermiticity_errors(self) -> tuple[float, float, float]:
        """Return coefficient Hermiticity errors for lossless validation."""

        errors: list[float] = []
        for matrix in (self.A0, self.A1, self.A2):
            denominator = float(sparse_norm(matrix))
            numerator = float(sparse_norm(matrix - matrix.getH()))
            errors.append(numerator if denominator == 0.0 else numerator / denominator)
        return tuple(errors)  # type: ignore[return-value]

    def divergence_residual(self, full_vector: ArrayLike, neff: complex) -> float:
        r"""Return the normalized weak residual of ``div(eps_r E) = 0``.

        Test functions vanish at the two outer endpoints, so conductor surface
        charge does not contaminate this bulk Gauss-law diagnostic.
        """

        vector = np.asarray(full_vector, dtype=np.complex128)
        if vector.shape != (self.full_size,):
            raise ValueError(
                f"full_vector must have shape ({self.full_size},), got {vector.shape}."
            )
        ex = vector[self.ex_slice]
        ey = vector[self.ey_slice]
        ez = vector[self.ez_slice]
        terms = (
            -(self.divergence_x @ ex),
            1j * self.eta * (self.epsilon_mass @ ey),
            1j * complex(neff) * (self.epsilon_mass_z @ ez),
        )
        rows = self.divergence_test_dofs
        residual = sum(terms)[rows]
        # Scale by operator and field norms rather than by the evaluated terms.
        # For an exact TEM mode every term is individually zero; dividing by
        # their roundoff-level values would incorrectly report an O(1) error.
        divergence_rows = self.divergence_x[rows]
        mass_y_rows = self.epsilon_mass[rows]
        mass_z_rows = self.epsilon_mass_z[rows]
        denominator = (
            float(sparse_norm(divergence_rows)) * float(np.linalg.norm(ex))
            + abs(self.eta)
            * float(sparse_norm(mass_y_rows))
            * float(np.linalg.norm(ey))
            + abs(complex(neff))
            * float(sparse_norm(mass_z_rows))
            * float(np.linalg.norm(ez))
        )
        if denominator == 0.0:
            return float(np.linalg.norm(residual))
        return float(np.linalg.norm(residual) / denominator)


@dataclass(frozen=True, slots=True)
class Mode:
    """One normalized full-vector lead mode.

    Electric coefficients retain their conforming FEM representation:
    ``E_x`` is cellwise and ``E_y,E_z`` are nodal.  The ``E`` and ``H``
    properties provide all three physical components sampled at cell centres.
    """

    beta: complex
    neff: complex
    E_x: ComplexArray
    E_y: ComplexArray
    E_z: ComplexArray
    H_x: ComplexArray
    H_y: ComplexArray
    H_z: ComplexArray
    x_nodes: FloatArray
    power: float
    complex_power: complex
    ky: float
    omega: float
    direction: ModeDirection
    classification: ModeClassification
    normalization: Literal["unit-power", "energy-like"]
    residual: float
    divergence_residual: float
    H_x_left: ComplexArray | None = None
    H_x_right: ComplexArray | None = None

    @property
    def x(self) -> FloatArray:
        """Cell-centre physical x coordinates."""

        return 0.5 * (self.x_nodes[:-1] + self.x_nodes[1:])

    @property
    def E(self) -> ComplexArray:
        """Cell-centred electric field in physical ``(x, y, z)`` order."""

        return np.vstack(
            (
                self.E_x,
                0.5 * (self.E_y[:-1] + self.E_y[1:]),
                0.5 * (self.E_z[:-1] + self.E_z[1:]),
            )
        )

    @property
    def H(self) -> ComplexArray:
        """Cell-centred magnetic field in physical ``(x, y, z)`` order."""

        return np.vstack((self.H_x, self.H_y, self.H_z))

    @property
    def is_propagating(self) -> bool:
        return self.classification == "propagating"

    def _sampling_coordinates(self, x: ArrayLike) -> tuple[FloatArray, tuple[int, ...]]:
        """Validate and flatten physical transverse sampling coordinates."""

        values = np.asarray(x)
        if np.iscomplexobj(values) and np.any(np.imag(values) != 0.0):
            raise ValueError("x sampling coordinates must be real.")
        try:
            coordinates = np.asarray(np.real(values), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("x sampling coordinates must be real numbers.") from exc
        if not np.isfinite(coordinates).all():
            raise ValueError("x sampling coordinates must be finite.")

        nodes = np.asarray(self.x_nodes, dtype=np.float64)
        if nodes.ndim != 1 or nodes.size < 2 or np.any(np.diff(nodes) <= 0.0):
            raise ValueError("mode x_nodes must be a strictly increasing 1D array.")
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, abs(float(nodes[0])), abs(float(nodes[-1]))
        )
        if np.any(coordinates < nodes[0] - tolerance) or np.any(
            coordinates > nodes[-1] + tolerance
        ):
            raise ValueError(
                "x sampling coordinates lie outside the mode cross-section "
                f"[{nodes[0]:.16g}, {nodes[-1]:.16g}]."
            )
        clipped = np.clip(coordinates, nodes[0], nodes[-1])
        return np.asarray(clipped.ravel(), dtype=np.float64), coordinates.shape

    def _cell_indices(self, flat_x: FloatArray) -> IntArray:
        """Return deterministic right-sided P0 element indices."""

        indices = np.searchsorted(self.x_nodes, flat_x, side="right") - 1
        return np.asarray(np.clip(indices, 0, self.x_nodes.size - 2), dtype=np.int64)

    def _sample_cellwise(
        self, values: ArrayLike, flat_x: FloatArray, shape: tuple[int, ...], name: str
    ) -> ComplexArray:
        coefficients = np.asarray(values, dtype=np.complex128)
        expected = self.x_nodes.size - 1
        if coefficients.shape != (expected,):
            raise ValueError(
                f"mode {name} must have one value per cell, shape ({expected},); "
                f"got {coefficients.shape}."
            )
        sampled = coefficients[self._cell_indices(flat_x)]
        return np.asarray(sampled.reshape(shape), dtype=np.complex128)

    def _sample_nodal(
        self, values: ArrayLike, flat_x: FloatArray, shape: tuple[int, ...], name: str
    ) -> ComplexArray:
        coefficients = np.asarray(values, dtype=np.complex128)
        expected = self.x_nodes.size
        if coefficients.shape != (expected,):
            raise ValueError(
                f"mode {name} must have one value per node, shape ({expected},); "
                f"got {coefficients.shape}."
            )
        cells = self._cell_indices(flat_x)
        left = self.x_nodes[cells]
        fractions = (flat_x - left) / (self.x_nodes[cells + 1] - left)
        sampled = (1.0 - fractions) * coefficients[cells]
        sampled += fractions * coefficients[cells + 1]
        return np.asarray(sampled.reshape(shape), dtype=np.complex128)

    def sample_E(self, x: ArrayLike) -> ComplexArray:
        """Evaluate the transverse FEM electric trace at physical ``x``.

        The result has shape ``(3, *np.shape(x))`` in physical ``(x, y, z)``
        component order.  ``E_x`` retains its piecewise-constant Nedelec trace,
        while ``E_y`` and ``E_z`` are interpolated linearly from their nodal
        coefficients.  At an internal element boundary the P0 component uses
        the element immediately to the right; the final endpoint uses the last
        element.
        """

        flat_x, shape = self._sampling_coordinates(x)
        return np.stack(
            (
                self._sample_cellwise(self.E_x, flat_x, shape, "E_x"),
                self._sample_nodal(self.E_y, flat_x, shape, "E_y"),
                self._sample_nodal(self.E_z, flat_x, shape, "E_z"),
            ),
            axis=0,
        )

    def sample_H(self, x: ArrayLike) -> ComplexArray:
        """Evaluate the reconstructed magnetic trace at physical ``x``.

        ``H_x`` is piecewise linear within each cell (and can jump across a
        material interface), while ``H_y,H_z`` are cellwise.  Older manually
        constructed Mode objects without endpoint data retain the midpoint-P0
        fallback.  The output shape and component order match :meth:`sample_E`.
        """

        flat_x, shape = self._sampling_coordinates(x)
        if self.H_x_left is None or self.H_x_right is None:
            sampled_hx = self._sample_cellwise(self.H_x, flat_x, shape, "H_x")
        else:
            left_values = np.asarray(self.H_x_left, dtype=np.complex128)
            right_values = np.asarray(self.H_x_right, dtype=np.complex128)
            expected = self.x_nodes.size - 1
            if left_values.shape != (expected,) or right_values.shape != (expected,):
                raise ValueError(
                    "mode H_x endpoint arrays must each have one value per cell."
                )
            cells = self._cell_indices(flat_x)
            left_coordinates = self.x_nodes[cells]
            fractions = (flat_x - left_coordinates) / (
                self.x_nodes[cells + 1] - left_coordinates
            )
            sampled_hx = (
                (1.0 - fractions) * left_values[cells]
                + fractions * right_values[cells]
            ).reshape(shape)
        return np.stack(
            (
                np.asarray(sampled_hx, dtype=np.complex128),
                self._sample_cellwise(self.H_y, flat_x, shape, "H_y"),
                self._sample_cellwise(self.H_z, flat_x, shape, "H_z"),
            ),
            axis=0,
        )

    def fields(
        self,
        x: ArrayLike,
        z: ArrayLike,
        reference_plane: float = 0.0,
    ) -> tuple[ComplexArray, ComplexArray]:
        """Evaluate ``(E, H)`` with ``exp(i*beta*(z-reference_plane))``.

        ``x`` and ``z`` follow NumPy broadcasting rules.  At the reference
        plane the returned values are exactly the sampled modal traces.
        """

        reference = float(reference_plane)
        if not np.isfinite(reference):
            raise ValueError("reference_plane must be finite.")
        x_values = np.asarray(x)
        z_values = np.asarray(z)
        if np.iscomplexobj(z_values) and np.any(np.imag(z_values) != 0.0):
            raise ValueError("z sampling coordinates must be real.")
        try:
            x_broadcast, z_broadcast = np.broadcast_arrays(
                x_values, np.asarray(np.real(z_values), dtype=np.float64)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("x and z sampling coordinates must be broadcastable.") from exc
        if not np.isfinite(z_broadcast).all():
            raise ValueError("z sampling coordinates must be finite.")
        electric = self.sample_E(x_broadcast)
        magnetic = self.sample_H(x_broadcast)
        phase = self.phase_factor(z_broadcast, reference_plane=reference)
        if not np.isfinite(phase).all():
            raise ValueError("modal phase overflowed at the requested z coordinates.")
        return electric * phase[np.newaxis, ...], magnetic * phase[np.newaxis, ...]

    def counterpropagating(self) -> Mode:
        r"""Return the exact z-mirrored mode at the same ``omega`` and ``ky``.

        For scalar z-invariant media, reflecting ``z -> -z`` changes

        ``beta -> -beta``, ``E -> (E_x, E_y, -E_z)``, and
        ``H -> (-H_x, -H_y, H_z)``.

        This is a spatial reflection, not complex conjugation.  It therefore
        remains correct for loss under the ``exp(-i*omega*t)`` convention and
        reverses both the signed real power and complex longitudinal flux.
        """

        opposite_direction: dict[ModeDirection, ModeDirection] = {
            "forward": "backward",
            "backward": "forward",
            "right-decaying": "left-decaying",
            "left-decaying": "right-decaying",
            "indeterminate": "indeterminate",
        }
        return Mode(
            beta=-self.beta,
            neff=-self.neff,
            E_x=np.array(self.E_x, copy=True),
            E_y=np.array(self.E_y, copy=True),
            E_z=-np.array(self.E_z, copy=True),
            H_x=-np.array(self.H_x, copy=True),
            H_y=-np.array(self.H_y, copy=True),
            H_z=np.array(self.H_z, copy=True),
            x_nodes=np.array(self.x_nodes, copy=True),
            power=-self.power,
            complex_power=-self.complex_power,
            ky=self.ky,
            omega=self.omega,
            direction=opposite_direction[self.direction],
            classification=self.classification,
            normalization=self.normalization,
            residual=self.residual,
            divergence_residual=self.divergence_residual,
            H_x_left=(
                None
                if self.H_x_left is None
                else -np.array(self.H_x_left, copy=True)
            ),
            H_x_right=(
                None
                if self.H_x_right is None
                else -np.array(self.H_x_right, copy=True)
            ),
        )

    def backward(self) -> Mode:
        """Return this modal family member propagating or decaying toward -z."""

        if self.direction in ("backward", "left-decaying"):
            return self
        return self.counterpropagating()

    def phase_factor(self, z: ArrayLike, *, reference_plane: float = 0.0) -> ComplexArray:
        """Return ``exp(i*beta*(z-reference_plane))`` for incident fields."""

        return np.asarray(
            np.exp(1j * self.beta * (np.asarray(z, dtype=float) - reference_plane)),
            dtype=np.complex128,
        )


@dataclass(frozen=True, slots=True)
class ModeSet(Sequence[Mode]):
    """Sequence of modes plus the matrices and eigensolver metadata."""

    modes: tuple[Mode, ...]
    system: ModeFEMSystem
    solve_info: dict[str, object]

    @overload
    def __getitem__(self, index: int) -> Mode: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Mode, ...]: ...

    def __getitem__(self, index: int | slice) -> Mode | tuple[Mode, ...]:
        return self.modes[index]

    def __len__(self) -> int:
        return len(self.modes)

    def __iter__(self) -> Iterator[Mode]:
        return iter(self.modes)


def _mesh_nodes(cross_section: CrossSection, requested_elements: int) -> FloatArray:
    target_size = (cross_section.x_span[1] - cross_section.x_span[0]) / requested_elements
    nodes: list[float] = []
    interfaces = cross_section.interfaces
    for left, right in zip(interfaces[:-1], interfaces[1:], strict=True):
        count = max(1, int(ceil((right - left) / target_size)))
        segment = np.linspace(left, right, count + 1)
        nodes.extend(segment[:-1])
    nodes.append(interfaces[-1])
    result = np.asarray(nodes, dtype=float)
    if np.any(np.diff(result) <= 0.0):
        raise ModeSolverError("Failed to create a strictly increasing mode mesh.")
    return result


def _zero(rows: int, columns: int) -> csr_matrix:
    return csr_matrix((rows, columns), dtype=np.complex128)


def _p1_inner_product(
    a_left: ComplexArray,
    a_right: ComplexArray,
    b_left: ComplexArray,
    b_right: ComplexArray,
    widths: FloatArray,
) -> complex:
    """Exactly integrate a piecewise-linear ``a*conj(b)`` product."""

    values = widths / 6.0 * (
        2.0 * a_left * np.conj(b_left)
        + a_left * np.conj(b_right)
        + a_right * np.conj(b_left)
        + 2.0 * a_right * np.conj(b_right)
    )
    return complex(np.sum(values))


class ModeSolver:
    """Solve the fixed-omega, fixed-ky full-vector 1D Maxwell pencil."""

    def __init__(
        self,
        cross_section: CrossSection,
        *,
        frequency: float | None = None,
        omega: float | None = None,
        wavelength: float | None = None,
        ky: float = 0.0,
        num_elements: int = 24,
        quadrature_order: int = 4,
        dense_linearization_limit: int = 420,
    ) -> None:
        if not isinstance(cross_section, CrossSection):
            raise ConfigurationError("cross_section must be a CrossSection instance.")
        self.cross_section = cross_section
        self.frequency = resolve_frequency(
            wavelength=wavelength, frequency=frequency, omega=omega
        )
        ky_value = complex(ky)
        if not np.isfinite((ky_value.real, ky_value.imag)).all():
            raise ConfigurationError("ky must be finite.")
        if ky_value.imag != 0.0:
            raise ConfigurationError(
                "Complex ky is not yet validated by the mode solver; use a real ky."
            )
        self.ky = float(ky_value.real)
        self.num_elements = _positive_integer(
            num_elements, "num_elements", minimum=2
        )
        self.quadrature_order = _positive_integer(
            quadrature_order, "quadrature_order", minimum=2
        )
        self.dense_linearization_limit = _positive_integer(
            dense_linearization_limit, "dense_linearization_limit", minimum=2
        )

    def assemble(self) -> ModeFEMSystem:
        """Assemble and return the dimensionless QEP coefficient matrices."""

        boundary = self.cross_section.boundary
        if boundary is None:
            raise ConfigurationError(
                "A mode cross-section requires an explicit transverse boundary; "
                "set boundary='pec' or boundary='pmc'. A PML attenuates the "
                "field before the boundary but does not replace its outer "
                "termination."
            )
        if getattr(self, "_adaptive_interfaces", None) == self.cross_section.interfaces:
            x_nodes = self._adaptive_nodes
        else:
            x_nodes = _mesh_nodes(self.cross_section, self.num_elements)
        xi_nodes = self.frequency.k0 * x_nodes
        mesh = MeshLine(xi_nodes)
        basis_x = Basis(
            mesh, ElementLineP0(), intorder=self.quadrature_order
        )
        basis_t = Basis(
            mesh, ElementLineP1(), intorder=self.quadrature_order
        )
        k0 = self.frequency.k0

        def coefficients(w: object) -> tuple[ComplexArray, ComplexArray]:
            eps_r, mu_r = self.cross_section.diagonal_material_at(
                w.x[0] / k0, k_reference=k0
            )
            if np.any(np.abs(mu_r) == 0.0):
                raise ConfigurationError(
                    "mu_r must be nonzero throughout the cross-section."
                )
            return eps_r, 1.0 / mu_r

        @BilinearForm(dtype=np.complex128)
        def mass_qx(u: object, v: object, w: object) -> object:
            _, q = coefficients(w)
            return q[0] * u * np.conj(v)

        @BilinearForm(dtype=np.complex128)
        def mass_qy(u: object, v: object, w: object) -> object:
            _, q = coefficients(w)
            return q[1] * u * np.conj(v)

        @BilinearForm(dtype=np.complex128)
        def mass_qz(u: object, v: object, w: object) -> object:
            _, q = coefficients(w)
            return q[2] * u * np.conj(v)

        @BilinearForm(dtype=np.complex128)
        def mass_px(u: object, v: object, w: object) -> object:
            p, _ = coefficients(w)
            return p[0] * u * np.conj(v)

        @BilinearForm(dtype=np.complex128)
        def mass_py(u: object, v: object, w: object) -> object:
            p, _ = coefficients(w)
            return p[1] * u * np.conj(v)

        @BilinearForm(dtype=np.complex128)
        def mass_pz(u: object, v: object, w: object) -> object:
            p, _ = coefficients(w)
            return p[2] * u * np.conj(v)

        @BilinearForm(dtype=np.complex128)
        def stiffness_qy(u: object, v: object, w: object) -> object:
            _, q = coefficients(w)
            return q[1] * u.grad[0] * np.conj(v.grad[0])

        @BilinearForm(dtype=np.complex128)
        def stiffness_qz(u: object, v: object, w: object) -> object:
            _, q = coefficients(w)
            return q[2] * u.grad[0] * np.conj(v.grad[0])

        @BilinearForm(dtype=np.complex128)
        def trial_derivative_qy(u: object, v: object, w: object) -> object:
            _, q = coefficients(w)
            return q[1] * u.grad[0] * np.conj(v)

        @BilinearForm(dtype=np.complex128)
        def test_derivative_qy(u: object, v: object, w: object) -> object:
            _, q = coefficients(w)
            return q[1] * u * np.conj(v.grad[0])

        @BilinearForm(dtype=np.complex128)
        def trial_derivative_qz(u: object, v: object, w: object) -> object:
            _, q = coefficients(w)
            return q[2] * u.grad[0] * np.conj(v)

        @BilinearForm(dtype=np.complex128)
        def test_derivative_qz(u: object, v: object, w: object) -> object:
            _, q = coefficients(w)
            return q[2] * u * np.conj(v.grad[0])

        @BilinearForm(dtype=np.complex128)
        def test_derivative_px(u: object, v: object, w: object) -> object:
            p, _ = coefficients(w)
            return p[0] * u * np.conj(v.grad[0])

        mqy_x = asm(mass_qy, basis_x).astype(np.complex128)
        mqz_x = asm(mass_qz, basis_x).astype(np.complex128)
        mpx_x = asm(mass_px, basis_x).astype(np.complex128)
        mqx_t = asm(mass_qx, basis_t).astype(np.complex128)
        mpy_t = asm(mass_py, basis_t).astype(np.complex128)
        mpz_t = asm(mass_pz, basis_t).astype(np.complex128)
        kqy_t = asm(stiffness_qy, basis_t).astype(np.complex128)
        kqz_t = asm(stiffness_qz, basis_t).astype(np.complex128)
        dqy_xt = asm(trial_derivative_qy, basis_t, basis_x).astype(np.complex128)
        dqy_tx = asm(test_derivative_qy, basis_x, basis_t).astype(np.complex128)
        dqz_xt = asm(trial_derivative_qz, basis_t, basis_x).astype(np.complex128)
        dqz_tx = asm(test_derivative_qz, basis_x, basis_t).astype(np.complex128)
        dpx_tx = asm(test_derivative_px, basis_x, basis_t).astype(np.complex128)

        nx, nt = basis_x.N, basis_t.N
        eta = self.ky / k0
        z_xx, z_xt, z_tx, z_tt = (
            _zero(nx, nx),
            _zero(nx, nt),
            _zero(nt, nx),
            _zero(nt, nt),
        )
        # With C(E) = (i eta Ez - i lambda Ey,
        #              i lambda Ex - Dx Ez,
        #              Dx Ey - i eta Ex),
        # the x/y/z curl components carry qx/qy/qz respectively.  Keeping
        # those weights explicit is essential in a transverse PML, where
        # qx = sx/mu while qy = qz = 1/(mu*sx).
        a0 = bmat(
            (
                (eta**2 * mqz_x - mpx_x, 1j * eta * dqz_xt, z_xt),
                (-1j * eta * dqz_tx, kqz_t - mpy_t, z_tt),
                (z_tx, z_tt, kqy_t + eta**2 * mqx_t - mpz_t),
            ),
            format="csr",
            dtype=np.complex128,
        )
        a1 = bmat(
            (
                (z_xx, z_xt, 1j * dqy_xt),
                (z_tx, z_tt, -eta * mqx_t),
                (-1j * dqy_tx, -eta * mqx_t, z_tt),
            ),
            format="csr",
            dtype=np.complex128,
        )
        a2 = bmat(
            (
                (mqy_x, z_xt, z_xt),
                (z_tx, mqx_t, z_tt),
                (z_tx, z_tt, z_tt),
            ),
            format="csr",
            dtype=np.complex128,
        )

        ex_slice = slice(0, nx)
        ey_slice = slice(nx, nx + nt)
        ez_slice = slice(nx + nt, nx + 2 * nt)
        full_size = nx + 2 * nt
        interior = np.arange(1, nt - 1, dtype=np.int64)
        pec_coordinates = np.asarray(
            [item.x for item in self.cross_section.pec_boundaries], dtype=float
        )
        pec_nodes = np.asarray(
            [int(np.searchsorted(x_nodes, coordinate)) for coordinate in pec_coordinates],
            dtype=np.int64,
        )
        coordinate_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, float(np.max(np.abs(x_nodes))))
        )
        if pec_nodes.size and not np.allclose(
            x_nodes[pec_nodes],
            pec_coordinates,
            rtol=0.0,
            atol=coordinate_tolerance,
        ):
            raise ModeSolverError(
                "The mode mesh does not conform to an internal PEC boundary."
            )

        constrained_tangential = pec_nodes
        if boundary == "pec":
            constrained_tangential = np.unique(
                np.concatenate(
                    (
                        np.asarray((0, nt - 1), dtype=np.int64),
                        constrained_tangential,
                    )
                )
            )
        free_tangential = np.setdiff1d(
            np.arange(nt, dtype=np.int64),
            constrained_tangential,
            assume_unique=True,
        )
        free_dofs = np.concatenate(
            (
                # E_x is normal to x=constant PEC sheets, so every one-sided
                # P0 trace remains an unconstrained unknown.
                np.arange(nx, dtype=np.int64),
                nx + free_tangential,
                nx + nt + free_tangential,
            )
        )
        divergence_test_dofs = np.setdiff1d(
            interior,
            pec_nodes,
            assume_unique=True,
        )

        reduced = tuple(
            matrix[free_dofs][:, free_dofs].tocsr() for matrix in (a0, a1, a2)
        )
        return ModeFEMSystem(
            x_nodes=x_nodes,
            xi_nodes=xi_nodes,
            A0=reduced[0],
            A1=reduced[1],
            A2=reduced[2],
            free_dofs=free_dofs,
            full_size=full_size,
            ex_slice=ex_slice,
            ey_slice=ey_slice,
            ez_slice=ez_slice,
            divergence_x=dpx_tx.tocsr(),
            epsilon_mass=mpy_t.tocsr(),
            epsilon_mass_z=mpz_t.tocsr(),
            divergence_test_dofs=divergence_test_dofs,
            frequency=self.frequency,
            ky=self.ky,
            eta=eta,
            boundary=boundary,
        )

    @staticmethod
    def _qep_residual(
        system: ModeFEMSystem, vector: ComplexArray, neff: complex
    ) -> float:
        terms = (
            system.A0 @ vector,
            complex(neff) * (system.A1 @ vector),
            complex(neff) ** 2 * (system.A2 @ vector),
        )
        denominator = sum(float(np.linalg.norm(term)) for term in terms)
        residual = sum(terms)
        if denominator == 0.0:
            return float(np.linalg.norm(residual))
        return float(np.linalg.norm(residual) / denominator)

    def _linearized_candidates(
        self,
        system: ModeFEMSystem,
        sigma: complex,
        count: int,
        tolerance: float,
    ) -> tuple[ComplexArray, ComplexArray, str]:
        n = system.ndofs
        size = 2 * n

        if size <= self.dense_linearization_limit or size <= 3:
            identity = eye(n, format="csr", dtype=np.complex128)
            zero = _zero(n, n)
            pencil_a = bmat(
                ((zero, identity), (-system.A0, -system.A1)),
                format="csc", dtype=np.complex128,
            )
            pencil_b = bmat(
                ((identity, zero), (zero, system.A2)),
                format="csc", dtype=np.complex128,
            )
            homogeneous, eigenvectors = linalg.eig(
                pencil_a.toarray(),
                pencil_b.toarray(),
                right=True,
                check_finite=False,
                homogeneous_eigvals=True,
            )
            alpha, denominator = homogeneous
            scale = np.maximum(np.abs(alpha), np.abs(denominator))
            finite = np.abs(denominator) > 256.0 * np.finfo(float).eps * np.maximum(
                scale, 1.0
            )
            values = alpha[finite] / denominator[finite]
            vectors = eigenvectors[:n, finite]
            order = np.argsort(np.abs(values - sigma))
            keep = order[: min(order.size, max(count * 3, count))]
            return (
                np.asarray(values[keep], dtype=np.complex128),
                np.asarray(vectors[:, keep], dtype=np.complex128),
                "dense-qz",
            )

        requested = min(max(count, 2), size - 2)
        shift = complex(sigma)
        perturbation = 1e-6j * max(1.0, abs(shift))
        shifted = shift + perturbation
        try:
            factor = splu(csc_matrix(system.polynomial(shifted)))
        except RuntimeError as exc:
            raise ModeSolverError(
                f"Shifted mode pencil could not be factorized near neff={sigma!r}."
            ) from exc

        # Eliminate the companion block analytically: only the original
        # n-by-n quadratic polynomial needs a sparse LU factorization.
        coupling = system.A1 + shifted * system.A2

        def apply_shift_invert(vector: ComplexArray) -> ComplexArray:
            first = factor.solve(-(coupling @ vector[:n] + system.A2 @ vector[n:]))
            return np.concatenate((first, vector[:n] + shifted * first))

        operator = LinearOperator(
            (size, size), matvec=apply_shift_invert, dtype=np.complex128
        )
        rng = np.random.default_rng(20260827)
        initial = rng.standard_normal(size) + 1j * rng.standard_normal(size)
        try:
            theta, eigenvectors = eigs(
                operator,
                k=requested,
                which="LM",
                v0=initial,
                tol=tolerance,
                maxiter=max(3000, 20 * size),
            )
        except ArpackNoConvergence as exc:
            if exc.eigenvalues is None or exc.eigenvectors is None:
                raise ModeSolverError("Sparse mode iteration did not converge.") from exc
            theta = exc.eigenvalues
            eigenvectors = exc.eigenvectors
        valid = np.abs(theta) > np.finfo(float).eps
        values = shift + perturbation + 1.0 / theta[valid]
        vectors = eigenvectors[:n, valid]
        order = np.argsort(np.abs(values - sigma))
        return (
            np.asarray(values[order], dtype=np.complex128),
            np.asarray(vectors[:, order], dtype=np.complex128),
            "sparse-shift-invert",
        )

    def _default_guess(self, system: ModeFEMSystem) -> complex:
        sample = 0.5 * (system.x_nodes[:-1] + system.x_nodes[1:])
        eps_r, mu_r = self.cross_section.material_at(sample)
        index_squared = eps_r * mu_r - system.eta**2
        candidate = index_squared[np.argmax(np.abs(index_squared))]
        root = complex(np.sqrt(candidate))
        if root.real < 0.0 or (root.real == 0.0 and root.imag < 0.0):
            root = -root
        return root

    def _field_data(
        self,
        system: ModeFEMSystem,
        ex: ComplexArray,
        ey: ComplexArray,
        ez: ComplexArray,
        neff: complex,
    ) -> tuple[
        ComplexArray,
        ComplexArray,
        ComplexArray,
        ComplexArray,
        ComplexArray,
        complex,
        float,
    ]:
        widths = np.diff(system.xi_nodes)
        centres = 0.5 * (system.x_nodes[:-1] + system.x_nodes[1:])
        integration_widths = widths
        if self.cross_section.pml is not None:
            physical_left = (
                self.cross_section.x_span[0] + self.cross_section.pml.thickness
            )
            physical_right = (
                self.cross_section.x_span[1] - self.cross_section.pml.thickness
            )
            # PML coordinates are numerical, not part of the physical port
            # cross-section used for modal power and normalization.
            integration_widths = widths * (
                (centres >= physical_left) & (centres <= physical_right)
            )
        eps_r, mu_r = self.cross_section.diagonal_material_at(
            centres, k_reference=system.frequency.k0
        )
        qx, qy, qz = 1.0 / mu_r
        dey = np.diff(ey) / widths
        dez = np.diff(ez) / widths
        # curl(E) = i*omega*mu*H for exp(-i*omega*t).  Each reconstructed
        # component therefore uses the matching diagonal inverse-mu entry.
        hx_left = qx * (system.eta * ez[:-1] - neff * ey[:-1])
        hx_right = qx * (system.eta * ez[1:] - neff * ey[1:])
        hx = 0.5 * (hx_left + hx_right)
        hy = qy * (neff * ex + 1j * dez)
        hz = qz * (-system.eta * ex - 1j * dey)

        ex_hy = complex(np.sum(integration_widths * ex * np.conj(hy)))
        ey_hx = _p1_inner_product(
            ey[:-1], ey[1:], hx_left, hx_right, integration_widths
        )
        complex_power = (ex_hy - ey_hx) / (
            2.0 * ETA_0 * system.frequency.k0
        )

        eps_weight = np.abs(eps_r)
        mu_weight = np.abs(mu_r)
        e_norm = np.sum(integration_widths * eps_weight[0] * np.abs(ex) ** 2)
        e_norm += _p1_inner_product(
            ey[:-1] * eps_weight[1],
            ey[1:] * eps_weight[1],
            ey[:-1],
            ey[1:],
            integration_widths,
        ).real
        e_norm += _p1_inner_product(
            ez[:-1] * eps_weight[2],
            ez[1:] * eps_weight[2],
            ez[:-1],
            ez[1:],
            integration_widths,
        ).real
        h_norm = np.sum(
            integration_widths
            * (
                mu_weight[1] * np.abs(hy) ** 2
                + mu_weight[2] * np.abs(hz) ** 2
            )
        )
        h_norm += _p1_inner_product(
            hx_left * mu_weight[0],
            hx_right * mu_weight[0],
            hx_left,
            hx_right,
            integration_widths,
        ).real
        energy_like = float(max((e_norm + h_norm).real / system.frequency.k0, 0.0))
        return (
            hx,
            hx_left,
            hx_right,
            hy,
            hz,
            complex(complex_power),
            energy_like,
        )

    def _make_mode(
        self,
        system: ModeFEMSystem,
        reduced_vector: ComplexArray,
        neff: complex,
        residual: float,
        divergence_residual: float,
        *,
        propagation_ratio_tolerance: float,
    ) -> Mode:
        full = system.expand(reduced_vector)
        ex = np.array(full[system.ex_slice], copy=True)
        ey = np.array(full[system.ey_slice], copy=True)
        ez = np.array(full[system.ez_slice], copy=True)
        hx, hx_left, hx_right, hy, hz, complex_power, energy_like = self._field_data(
            system, ex, ey, ez, neff
        )
        real_power = float(complex_power.real)
        propagating = (
            abs(complex_power) > np.finfo(float).tiny
            and abs(real_power) / abs(complex_power) >= propagation_ratio_tolerance
        )
        if propagating:
            scale = 1.0 / np.sqrt(abs(real_power))
            normalization: Literal["unit-power", "energy-like"] = "unit-power"
            direction: ModeDirection = "forward" if real_power > 0.0 else "backward"
            classification: ModeClassification = "propagating"
        else:
            if energy_like <= np.finfo(float).tiny:
                raise ModeSolverError("A mode has zero flux and zero energy-like norm.")
            scale = 1.0 / np.sqrt(energy_like)
            normalization = "energy-like"
            classification = "evanescent"
            decay_tolerance = 1e-10 * max(1.0, abs(neff))
            if neff.imag > decay_tolerance:
                direction = "right-decaying"
            elif neff.imag < -decay_tolerance:
                direction = "left-decaying"
            else:
                direction = "indeterminate"

        ex *= scale
        ey *= scale
        ez *= scale
        gauge_vector = np.concatenate((ex, ey, ez))
        pivot = gauge_vector[int(np.argmax(np.abs(gauge_vector)))]
        if abs(pivot) > 0.0:
            phase = np.exp(-1j * np.angle(pivot))
            ex *= phase
            ey *= phase
            ez *= phase

        hx, hx_left, hx_right, hy, hz, complex_power, _ = self._field_data(
            system, ex, ey, ez, neff
        )
        return Mode(
            beta=complex(system.frequency.k0 * neff),
            neff=complex(neff),
            E_x=ex,
            E_y=ey,
            E_z=ez,
            H_x=hx / ETA_0,
            H_y=hy / ETA_0,
            H_z=hz / ETA_0,
            x_nodes=np.array(system.x_nodes, copy=True),
            power=float(complex_power.real),
            complex_power=complex(complex_power),
            ky=system.ky,
            omega=system.frequency.omega,
            direction=direction,
            classification=classification,
            normalization=normalization,
            residual=residual,
            divergence_residual=divergence_residual,
            H_x_left=hx_left / ETA_0,
            H_x_right=hx_right / ETA_0,
        )

    @staticmethod
    def _direction_matches(mode: Mode, requested: RequestedDirection) -> bool:
        if requested == "all":
            return True
        if requested == "forward":
            return mode.direction in ("forward", "right-decaying")
        return mode.direction in ("backward", "left-decaying")

    def solve(self, *, max_refinements: int = 2,
              adaptive_tolerance: float = 0.05, **options) -> ModeSet:
        """Adapt the mixed line mesh using normal-D/tangential-H residuals."""
        from .adaptive import solve_modes
        return solve_modes(self, options, max_refinements, adaptive_tolerance)

    def _solve_once(
        self,
        *,
        num_modes: int = 4,
        neff_guess: complex | None = None,
        direction: RequestedDirection = "forward",
        eigensolver_tolerance: float = 1e-10,
        residual_tolerance: float = 1e-8,
        divergence_tolerance: float = 1e-7,
        propagation_ratio_tolerance: float = 1e-3,
    ) -> ModeSet:
        """Solve for modes nearest ``neff_guess`` and reject invalid roots."""

        requested = _positive_integer(num_modes, "num_modes")
        if direction not in ("forward", "backward", "all"):
            raise ConfigurationError("direction must be 'forward', 'backward', or 'all'.")
        for value, name in (
            (eigensolver_tolerance, "eigensolver_tolerance"),
            (residual_tolerance, "residual_tolerance"),
            (divergence_tolerance, "divergence_tolerance"),
            (propagation_ratio_tolerance, "propagation_ratio_tolerance"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ConfigurationError(f"{name} must be finite and positive.")
        system = self.assemble()
        sigma = self._default_guess(system) if neff_guess is None else complex(neff_guess)
        if not np.isfinite((sigma.real, sigma.imag)).all():
            raise ConfigurationError("neff_guess must be finite.")

        candidate_count = max(4 * requested, requested + 12)
        values, vectors, method = self._linearized_candidates(
            system,
            sigma,
            candidate_count,
            eigensolver_tolerance,
        )
        accepted: list[Mode] = []
        rejected_residual = 0
        rejected_divergence = 0
        rejected_direction = 0
        for neff, vector in zip(values, vectors.T, strict=True):
            vector_norm = float(np.linalg.norm(vector))
            if not np.isfinite(neff) or vector_norm <= np.finfo(float).tiny:
                continue
            vector = np.asarray(vector / vector_norm, dtype=np.complex128)
            residual = self._qep_residual(system, vector, complex(neff))
            if not np.isfinite(residual) or residual > residual_tolerance:
                rejected_residual += 1
                continue
            full = system.expand(vector)
            divergence_residual = system.divergence_residual(full, complex(neff))
            if (
                not np.isfinite(divergence_residual)
                or divergence_residual > divergence_tolerance
            ):
                rejected_divergence += 1
                continue
            mode = self._make_mode(
                system,
                vector,
                complex(neff),
                residual,
                divergence_residual,
                propagation_ratio_tolerance=propagation_ratio_tolerance,
            )
            if not self._direction_matches(mode, direction):
                rejected_direction += 1
                continue
            duplicate = False
            for existing in accepted:
                if abs(existing.neff - mode.neff) > 1e-8 * max(1.0, abs(mode.neff)):
                    continue
                old = np.concatenate((existing.E_x, existing.E_y, existing.E_z))
                new = np.concatenate((mode.E_x, mode.E_y, mode.E_z))
                overlap = abs(np.vdot(old, new)) / (
                    np.linalg.norm(old) * np.linalg.norm(new)
                )
                if overlap > 1.0 - 1e-8:
                    duplicate = True
                    break
            if duplicate:
                continue
            accepted.append(mode)
            if len(accepted) == requested:
                break

        if len(accepted) < requested:
            raise ModeSolverError(
                f"Requested {requested} {direction} mode(s) near neff={sigma!r}, "
                f"but only {len(accepted)} passed validation. Rejected: "
                f"eigen-residual={rejected_residual}, "
                f"Gauss-law={rejected_divergence}, direction={rejected_direction}. "
                "Try a closer neff_guess, more elements, or direction='all'."
            )
        return ModeSet(
            modes=tuple(accepted),
            system=system,
            solve_info={
                "method": method,
                "candidate_count": int(values.size),
                "neff_guess": sigma,
                "requested_direction": direction,
                "eigen_residual_tolerance": residual_tolerance,
                "divergence_residual_tolerance": divergence_tolerance,
            },
        )


__all__ = [
    "CrossSection",
    "Layer",
    "Mode",
    "ModeFEMSystem",
    "ModeSet",
    "ModeSolver",
    "PECBoundary",
]

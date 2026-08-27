"""Power-inner-product modal projection on uniform-lead monitor lines.

The projector uses both electric and magnetic fields and constructs the
small dense Gram system from the symmetrized complex Poynting inner product.
This avoids assuming that numerically computed modes are exactly orthogonal.
The initial implementation targets reciprocal propagating modes; genuinely
lossy guides require adjoint/biorthogonal modes and are rejected upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import ModeProjectionError


ComplexArray = NDArray[np.complex128]


def modal_power_from_gram(
    amplitudes: ArrayLike,
    gram_matrix: ArrayLike,
    *,
    indices: ArrayLike | None = None,
    normalize_diagonal: bool = False,
) -> float:
    """Return signed real modal flux from a power-Gram block.

    WaveFEM's overlap is linear in its first field and conjugate-linear in its
    second, so coefficients ``a`` give ``Re(a.T @ G @ conj(a))``. Restricting
    ``indices`` to propagating families excludes energy-normalized evanescent
    traces from ordinary port power.
    """

    coefficients = np.asarray(amplitudes, dtype=np.complex128)
    gram = np.asarray(gram_matrix, dtype=np.complex128)
    if coefficients.ndim != 1 or gram.shape != (
        coefficients.size,
        coefficients.size,
    ):
        raise ModeProjectionError(
            "amplitudes must be one-dimensional and gram_matrix must be square "
            "with the same size."
        )
    if not np.isfinite(coefficients).all() or not np.isfinite(gram).all():
        raise ModeProjectionError("Modal amplitudes and power Gram must be finite.")
    if indices is not None:
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "iu":
            raise ModeProjectionError("indices must be a one-dimensional integer array.")
        if np.any(selected < 0) or np.any(selected >= coefficients.size):
            raise ModeProjectionError("A modal-power index is outside the Gram block.")
        coefficients = coefficients[selected]
        gram = gram[np.ix_(selected, selected)]
    if not isinstance(normalize_diagonal, bool):
        raise ModeProjectionError("normalize_diagonal must be a boolean.")
    hermitian_gram = 0.5 * (gram + gram.conj().T)
    if normalize_diagonal and coefficients.size:
        diagonal_power = np.abs(np.real(np.diag(hermitian_gram)))
        if np.any(diagonal_power <= np.finfo(float).tiny):
            raise ModeProjectionError(
                "A selected propagating mode has zero sampled diagonal power."
            )
        scales = np.sqrt(diagonal_power)
        hermitian_gram = hermitian_gram / (scales[:, None] * scales[None, :])
    return float(np.real(coefficients @ hermitian_gram @ np.conj(coefficients)))


def _field_array(value: ArrayLike, name: str) -> ComplexArray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != 3:
        raise ModeProjectionError(
            f"{name} must have shape (3, npoints); received {array.shape}."
        )
    if array.shape[1] == 0 or not np.isfinite(array).all():
        raise ModeProjectionError(f"{name} must contain finite monitor samples.")
    return array


@dataclass(frozen=True, slots=True)
class ModalTrace:
    """Electric and magnetic fields of one mode on a monitor quadrature."""

    E: ComplexArray
    H: ComplexArray
    label: str = "mode"

    def __post_init__(self) -> None:
        electric = _field_array(self.E, "ModalTrace.E")
        magnetic = _field_array(self.H, "ModalTrace.H")
        if electric.shape != magnetic.shape:
            raise ModeProjectionError("ModalTrace E and H samples must have equal shape.")
        object.__setattr__(self, "E", electric)
        object.__setattr__(self, "H", magnetic)


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Recovered modal amplitudes and numerical quality diagnostics."""

    amplitudes: ComplexArray
    gram_matrix: ComplexArray
    condition_number: float
    relative_residual: float
    labels: tuple[str, ...]


class ElectromagneticProjector:
    """Project sampled fields onto a forward/backward modal trace basis."""

    def __init__(
        self,
        traces: Iterable[ModalTrace],
        weights: ArrayLike,
        *,
        impedance: float | None = None,
        condition_limit: float = 1e12,
    ) -> None:
        self.traces = tuple(traces)
        if not self.traces:
            raise ModeProjectionError("At least one candidate modal trace is required.")
        npoints = self.traces[0].E.shape[1]
        if any(trace.E.shape[1] != npoints for trace in self.traces):
            raise ModeProjectionError("All modal traces must share the same quadrature.")
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (npoints,):
            raise ModeProjectionError(
                f"weights must have shape ({npoints},); received {self.weights.shape}."
            )
        if not np.isfinite(self.weights).all() or np.any(self.weights <= 0.0):
            raise ModeProjectionError("Monitor quadrature weights must be finite and positive.")
        # Kept as a deprecated compatibility argument for the residual norm;
        # modal amplitudes themselves are obtained from the power Gram system.
        self.impedance = 1.0 if impedance is None else float(impedance)
        self.condition_limit = float(condition_limit)
        if not np.isfinite(self.impedance) or self.impedance <= 0.0:
            raise ModeProjectionError("Projection impedance must be finite and positive.")
        if not np.isfinite(self.condition_limit) or self.condition_limit <= 1.0:
            raise ModeProjectionError("condition_limit must be finite and greater than one.")

    def _weighted_state(self, E: ArrayLike, H: ArrayLike) -> ComplexArray:
        electric = _field_array(E, "monitor E")
        magnetic = _field_array(H, "monitor H")
        if electric.shape != self.traces[0].E.shape or magnetic.shape != electric.shape:
            raise ModeProjectionError(
                "Monitor fields must use the same (3, npoints) quadrature as the modes."
            )
        root_w = np.sqrt(self.weights)[None, :]
        return np.concatenate(
            ((root_w * electric).ravel(), (root_w * self.impedance * magnetic).ravel())
        )

    def _power_overlap(
        self,
        first_E: ComplexArray,
        first_H: ComplexArray,
        second_E: ComplexArray,
        second_H: ComplexArray,
    ) -> complex:
        """Return the symmetrized Poynting product ``<first, second>_P``.

        The monitor normal is +z.  For equal arguments this is the signed
        time-averaged z power, ``0.5 Re integral(E x H*)_z dx``.
        """

        first_cross_second = (
            first_E[0] * np.conj(second_H[1])
            - first_E[1] * np.conj(second_H[0])
        )
        second_star_cross_first = (
            np.conj(second_E[0]) * first_H[1]
            - np.conj(second_E[1]) * first_H[0]
        )
        return complex(
            0.25
            * np.sum(self.weights * (first_cross_second + second_star_cross_first))
        )

    def project(self, E: ArrayLike, H: ArrayLike) -> ProjectionResult:
        """Solve the small dense electromagnetic Gram system for amplitudes."""

        columns = np.column_stack(
            [self._weighted_state(trace.E, trace.H) for trace in self.traces]
        )
        data = self._weighted_state(E, H)
        scales = np.linalg.norm(columns, axis=0)
        if np.any(scales == 0.0):
            raise ModeProjectionError("A candidate mode has a zero electromagnetic trace.")
        target_E = _field_array(E, "monitor E")
        target_H = _field_array(H, "monitor H")
        gram = np.asarray(
            [
                [
                    self._power_overlap(a.E, a.H, b.E, b.H)
                    for b in self.traces
                ]
                for a in self.traces
            ],
            dtype=np.complex128,
        )
        rhs = np.asarray(
            [
                self._power_overlap(trace.E, trace.H, target_E, target_H)
                for trace in self.traces
            ],
            dtype=np.complex128,
        )
        normalized_gram = gram / (scales[:, None] * scales[None, :])
        normalized_rhs = rhs / scales
        condition = float(np.linalg.cond(normalized_gram))
        if not np.isfinite(condition) or condition > self.condition_limit:
            raise ModeProjectionError(
                "The modal projection Gram matrix is near singular "
                f"(condition number {condition:.3e}, limit {self.condition_limit:.3e})."
            )
        # <mode_i, field> is conjugate-linear in ``field``.  Therefore the
        # Gram solve recovers conjugated modal amplitudes.
        scaled_conjugate_amplitudes = np.linalg.solve(
            normalized_gram, normalized_rhs
        )
        amplitudes = np.conj(scaled_conjugate_amplitudes / scales)
        residual = np.linalg.norm(columns @ amplitudes - data)
        denominator = np.linalg.norm(data)
        relative_residual = float(residual / denominator) if denominator else float(residual)
        return ProjectionResult(
            amplitudes=np.asarray(amplitudes, dtype=np.complex128),
            gram_matrix=gram,
            condition_number=condition,
            relative_residual=relative_residual,
            labels=tuple(trace.label for trace in self.traces),
        )


__all__ = [
    "ElectromagneticProjector",
    "ModalTrace",
    "ProjectionResult",
    "modal_power_from_gram",
]

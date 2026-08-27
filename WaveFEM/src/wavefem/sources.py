"""Scattered-field equivalent-source assembly.

For unchanged permeability, the relative-unit equation is

``L_actual E_sc = k0**2 (eps_actual - eps_background) E_inc``.

The material difference is evaluated directly at quadrature points so its
support is exactly the user-defined perturbation rather than a projected or
smoothed approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .fem import (
    ConstitutiveCoefficient,
    MixedFEMSystem,
    MixedFieldSolution,
    assemble_load_vector,
    evaluate_diagonal_coefficient,
    solve_homogeneous_pec,
)
from .exceptions import ConfigurationError


IncidentField: TypeAlias = Callable[
    [NDArray[np.floating], NDArray[np.floating]], object
]


def _incident_values(
    incident: IncidentField,
    x: NDArray[np.floating],
    z: NDArray[np.floating],
) -> NDArray[np.complex128]:
    values = np.asarray(incident(x, z), dtype=np.complex128)
    target = (3, *x.shape)
    try:
        if values.shape == (3,):
            values = np.broadcast_to(
                values.reshape((3,) + (1,) * x.ndim), target
            )
        elif values.shape == (*x.shape, 3):
            values = np.moveaxis(values, -1, 0)
        else:
            values = np.broadcast_to(values, target)
    except ValueError as exc:
        raise ValueError(
            "incident field must return three components in (x, y, z) order; "
            f"received {values.shape}, expected compatibility with {target}."
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError("incident field contains a non-finite value.")
    return np.asarray(values, dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class EquivalentSource:
    """Assembled RHS and quadrature support diagnostics."""

    load: NDArray[np.complex128]
    active_quadrature_fraction: float
    maximum_delta_eps: float

    @property
    def is_zero(self) -> bool:
        return not np.any(self.load)


@dataclass(frozen=True, slots=True)
class ScatteredFieldSolution:
    """Low-level scattered FEM field and the source that produced it."""

    field: MixedFieldSolution
    source: EquivalentSource


def assemble_equivalent_source(
    system: MixedFEMSystem,
    *,
    eps_background: ConstitutiveCoefficient,
    mu_background: ConstitutiveCoefficient = 1.0,
    incident: IncidentField,
) -> EquivalentSource:
    """Assemble ``k0^2 Delta-eps E_inc`` for unchanged permeability."""

    coordinates = system.physical_coordinates()
    x, z = coordinates[0], coordinates[1]
    actual = evaluate_diagonal_coefficient(
        system.parameters.eps_r, x, z, name="actual eps_r"
    )
    background = evaluate_diagonal_coefficient(
        eps_background, x, z, name="background eps_r"
    )
    actual_mu = evaluate_diagonal_coefficient(
        system.parameters.mu_r, x, z, name="actual mu_r"
    )
    background_mu = evaluate_diagonal_coefficient(
        mu_background, x, z, name="background mu_r"
    )
    if not np.allclose(actual_mu, background_mu, rtol=1e-12, atol=1e-14):
        raise ConfigurationError(
            "The MVP scattered-field source supports permittivity "
            "perturbations only; actual and background mu_r differ."
        )
    delta = actual - background
    incident_values = _incident_values(incident, x, z)
    def source_at_quadrature(
        x_computational: NDArray[np.floating],
        z_computational: NDArray[np.floating],
    ) -> NDArray[np.complex128]:
        # The FEM basis lives on the dimensionless computational mesh; public
        # material and incident callbacks continue to receive SI metres.
        x_request = system.length_scale * x_computational
        z_request = system.length_scale * z_computational
        actual_request = evaluate_diagonal_coefficient(
            system.parameters.eps_r, x_request, z_request, name="actual eps_r"
        )
        background_request = evaluate_diagonal_coefficient(
            eps_background, x_request, z_request, name="background eps_r"
        )
        return (
            system.dimensionless_k0**2
            * (actual_request - background_request)
            * _incident_values(incident, x_request, z_request)
        )

    load = assemble_load_vector(system.basis, source_at_quadrature)
    active = np.any(np.abs(delta) > 0.0, axis=0)
    return EquivalentSource(
        load=np.asarray(load, dtype=np.complex128),
        active_quadrature_fraction=float(np.count_nonzero(active) / active.size),
        maximum_delta_eps=float(np.max(np.abs(delta))),
    )


def solve_scattered_pec(
    system: MixedFEMSystem,
    *,
    eps_background: ConstitutiveCoefficient,
    mu_background: ConstitutiveCoefficient = 1.0,
    incident: IncidentField,
    residual_tolerance: float = 1e-7,
) -> ScatteredFieldSolution:
    """Solve the scattered field with homogeneous outer PEC truncation."""

    source = assemble_equivalent_source(
        system,
        eps_background=eps_background,
        mu_background=mu_background,
        incident=incident,
    )
    field = solve_homogeneous_pec(
        system,
        source.load,
        residual_tolerance=residual_tolerance,
    )
    return ScatteredFieldSolution(field=field, source=source)


__all__ = [
    "EquivalentSource",
    "IncidentField",
    "ScatteredFieldSolution",
    "assemble_equivalent_source",
    "solve_scattered_pec",
]

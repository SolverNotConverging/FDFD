"""Conforming monitor-line sampling for mixed 2.5D Maxwell fields.

Monitor planes used for modal projection are horizontal lines in the
computational ``(x, z)`` plane, i.e. ``z = constant`` and parameterized by
``x``.  This module samples only lines made from mesh facets; it never probes
through element interiors.  The convention is ``exp(-i omega t)`` and hence

``H = curl_ky(E) / (i omega mu_0 mu_r)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skfem import Basis, InteriorFacetBasis

from .constants import MU_0
from .exceptions import ModeProjectionError
from .fem import (
    ConstitutiveCoefficient,
    _validate_mixed_basis,
    evaluate_diagonal_coefficient,
)
from .operators import electric_field_vector, modified_curl


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MonitorSamples:
    """Electric and magnetic fields on one sorted monitor quadrature.

    ``E`` and ``H`` have shape ``(3, npoints)`` in physical ``(x, y, z)``
    order.  ``weights`` are positive line-integration weights paired with the
    monotonically increasing ``x`` coordinates.
    """

    x: RealArray
    weights: RealArray
    E: ComplexArray
    H: ComplexArray
    z: float

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        electric = np.asarray(self.E, dtype=np.complex128)
        magnetic = np.asarray(self.H, dtype=np.complex128)
        if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
            raise ModeProjectionError("Monitor x coordinates must be a finite 1D array.")
        if weights.shape != x.shape or not np.isfinite(weights).all():
            raise ModeProjectionError("Monitor weights must match the x coordinates.")
        if np.any(weights <= 0.0):
            raise ModeProjectionError("Monitor quadrature weights must be positive.")
        expected = (3, x.size)
        if electric.shape != expected or magnetic.shape != expected:
            raise ModeProjectionError(
                f"Monitor E and H must have shape {expected}; received "
                f"{electric.shape} and {magnetic.shape}."
            )
        if not np.isfinite(electric).all() or not np.isfinite(magnetic).all():
            raise ModeProjectionError("Monitor fields must contain only finite values.")
        if not np.isfinite(self.z):
            raise ModeProjectionError("Monitor z coordinate must be finite.")
        object.__setattr__(self, "x", np.array(x, copy=True))
        object.__setattr__(self, "weights", np.array(weights, copy=True))
        object.__setattr__(self, "E", np.array(electric, copy=True))
        object.__setattr__(self, "H", np.array(magnetic, copy=True))
        object.__setattr__(self, "z", float(self.z))


@dataclass(frozen=True, slots=True)
class HorizontalMonitorSamples:
    """Fields on a sorted physical ``x = constant`` side monitor."""

    z: RealArray
    weights: RealArray
    E: ComplexArray
    H: ComplexArray
    x: float

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        electric = np.asarray(self.E, dtype=np.complex128)
        magnetic = np.asarray(self.H, dtype=np.complex128)
        if z.ndim != 1 or z.size == 0 or not np.isfinite(z).all():
            raise ModeProjectionError("Side-monitor z coordinates must be a finite 1D array.")
        if weights.shape != z.shape or np.any(weights <= 0.0):
            raise ModeProjectionError("Side-monitor weights must be finite, positive, and match z.")
        expected = (3, z.size)
        if electric.shape != expected or magnetic.shape != expected:
            raise ModeProjectionError(f"Side-monitor E and H must have shape {expected}.")
        if not np.isfinite(electric).all() or not np.isfinite(magnetic).all():
            raise ModeProjectionError("Side-monitor fields must contain only finite values.")
        if not np.isfinite(self.x):
            raise ModeProjectionError("Side-monitor x coordinate must be finite.")
        object.__setattr__(self, "z", np.array(z, copy=True))
        object.__setattr__(self, "weights", np.array(weights, copy=True))
        object.__setattr__(self, "E", np.array(electric, copy=True))
        object.__setattr__(self, "H", np.array(magnetic, copy=True))
        object.__setattr__(self, "x", float(self.x))


def _finite_real(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
        qualifier = "positive " if positive else ""
        raise ModeProjectionError(f"{name} must be a {qualifier}finite real number.")
    converted = complex(value)
    if converted.imag != 0.0 or not np.isfinite(converted.real):
        qualifier = "positive " if positive else ""
        raise ModeProjectionError(f"{name} must be a {qualifier}finite real number.")
    result = float(converted.real)
    if positive and result <= 0.0:
        raise ModeProjectionError(f"{name} must be a positive finite real number.")
    return result


def _default_coordinate_tolerance(coordinates: RealArray) -> float:
    spans = np.ptp(coordinates, axis=1)
    span = float(np.max(spans))
    magnitude = float(np.max(np.abs(coordinates)))
    scale = max(span, magnitude, np.finfo(float).tiny)
    return max(1e-12 * span, 64.0 * np.finfo(float).eps * scale)


def _monitor_facets(
    basis: Basis,
    z: float,
    tolerance: float,
    *,
    length_scale: float,
) -> NDArray[np.int32]:
    mesh = basis.mesh
    z_nodes = np.asarray(mesh.p[1], dtype=float)
    z_min = float(np.min(z_nodes))
    z_max = float(np.max(z_nodes))
    if not (z_min + tolerance < z < z_max - tolerance):
        raise ModeProjectionError(
            f"Monitor z={z * length_scale:g} must lie strictly inside the "
            f"physical mesh domain ({z_min * length_scale:g}, "
            f"{z_max * length_scale:g})."
        )

    facet_nodes = mesh.facets
    facet_z = mesh.p[1, facet_nodes]
    facet_x = mesh.p[0, facet_nodes]
    on_monitor = np.all(np.abs(facet_z - z) <= tolerance, axis=0)
    nondegenerate = np.abs(facet_x[1] - facet_x[0]) > tolerance
    interior = mesh.f2t[1] >= 0
    facets = np.flatnonzero(on_monitor & nondegenerate & interior).astype(np.int32)
    if facets.size == 0:
        raise ModeProjectionError(
            f"No interior mesh facets were found at physical z={z * length_scale:g}; "
            "the monitor must coincide with a mesh-conforming constant-z line."
        )
    return facets


def _side_monitor_facets(
    basis: Basis,
    x: float,
    tolerance: float,
    *,
    length_scale: float,
) -> NDArray[np.int32]:
    mesh = basis.mesh
    x_nodes = np.asarray(mesh.p[0], dtype=float)
    x_min = float(np.min(x_nodes))
    x_max = float(np.max(x_nodes))
    if not (x_min + tolerance < x < x_max - tolerance):
        raise ModeProjectionError(
            f"Side monitor x={x * length_scale:g} must lie strictly inside the "
            f"physical mesh domain ({x_min * length_scale:g}, {x_max * length_scale:g})."
        )
    facet_nodes = mesh.facets
    facet_x = mesh.p[0, facet_nodes]
    facet_z = mesh.p[1, facet_nodes]
    on_monitor = np.all(np.abs(facet_x - x) <= tolerance, axis=0)
    nondegenerate = np.abs(facet_z[1] - facet_z[0]) > tolerance
    interior = mesh.f2t[1] >= 0
    facets = np.flatnonzero(on_monitor & nondegenerate & interior).astype(np.int32)
    if facets.size == 0:
        raise ModeProjectionError(
            f"No interior mesh facets were found at physical x={x * length_scale:g}; "
            "the side monitor must coincide with a mesh-conforming constant-x line."
        )
    return facets


def _facet_fields(
    facet_basis: InteriorFacetBasis,
    coefficients: ComplexArray,
    ky: complex,
) -> tuple[ComplexArray, ComplexArray]:
    fields = facet_basis.interpolate(coefficients)
    if not isinstance(fields, tuple) or len(fields) != 2:
        raise ModeProjectionError("Expected an H(curl)-H1 composite monitor field.")
    tangential, invariant = fields
    electric = electric_field_vector(tangential, invariant)
    curl = modified_curl(tangential, invariant, ky)
    return (
        np.asarray(electric, dtype=np.complex128),
        np.asarray(curl, dtype=np.complex128),
    )


def sample_vertical_monitor(
    basis: Basis,
    coefficients: ArrayLike,
    *,
    z: float,
    ky: complex = 0.0,
    omega: float,
    mu_r: ConstitutiveCoefficient = 1.0,
    length_scale: float = 1.0,
    intorder: int = 4,
    tolerance: float | None = None,
) -> MonitorSamples:
    """Sample a mesh-conforming ``z = constant`` monitor line.

    ``basis`` coordinates are interpreted as nondimensional coordinates
    ``(x, z) / length_scale``; ``z``, ``ky``, the constitutive callback
    coordinates, returned coordinates, and returned weights are all physical
    SI quantities.  Both adjacent elements are evaluated with
    :class:`InteriorFacetBasis` and
    their traces are averaged.  Tangential H(curl) values and H1 values agree
    exactly across the facet; averaging additionally gives a deterministic
    centered trace for elementwise-discontinuous normal components and
    derivatives, independent of triangle numbering or facet orientation.
    Samples and their positive geometric quadrature weights are finally sorted
    by physical ``x`` coordinate.
    """

    _validate_mixed_basis(basis)
    z_value = _finite_real(z, name="z")
    omega_value = _finite_real(omega, name="omega", positive=True)
    length_scale_value = _finite_real(
        length_scale, name="length_scale", positive=True
    )
    ky_value = complex(ky)
    if not np.isfinite(ky_value.real) or not np.isfinite(ky_value.imag):
        raise ModeProjectionError("ky must be finite.")
    if isinstance(intorder, bool) or int(intorder) != intorder or intorder < 1:
        raise ModeProjectionError("intorder must be a positive integer.")

    mesh_coordinates = np.asarray(basis.mesh.p, dtype=float)
    if tolerance is None:
        mesh_tolerance = _default_coordinate_tolerance(mesh_coordinates)
    else:
        physical_tolerance = _finite_real(
            tolerance, name="tolerance", positive=True
        )
        mesh_tolerance = physical_tolerance / length_scale_value

    values = np.asarray(coefficients, dtype=np.complex128)
    if values.shape != (basis.N,):
        raise ModeProjectionError(
            f"coefficients must have shape ({basis.N},); received {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ModeProjectionError("coefficients contain a non-finite value.")

    mesh_z = z_value / length_scale_value
    facets = _monitor_facets(
        basis,
        mesh_z,
        mesh_tolerance,
        length_scale=length_scale_value,
    )
    facet_kwargs = dict(
        facets=facets,
        dofs=basis.dofs,
        intorder=int(intorder),
    )
    side_zero = InteriorFacetBasis(basis.mesh, basis.elem, side=0, **facet_kwargs)
    side_one = InteriorFacetBasis(basis.mesh, basis.elem, side=1, **facet_kwargs)

    coordinates_zero = np.asarray(side_zero.global_coordinates(), dtype=float)
    coordinates_one = np.asarray(side_one.global_coordinates(), dtype=float)
    if not np.allclose(
        coordinates_zero,
        coordinates_one,
        rtol=0.0,
        atol=mesh_tolerance,
    ):
        raise ModeProjectionError("Interior-facet sides produced inconsistent quadrature.")
    if not np.all(np.abs(coordinates_zero[1] - mesh_z) <= mesh_tolerance):
        raise ModeProjectionError("Selected monitor quadrature is not on the requested z line.")

    scaled_ky = ky_value * length_scale_value
    electric_zero, curl_zero = _facet_fields(side_zero, values, scaled_ky)
    electric_one, curl_one = _facet_fields(side_one, values, scaled_ky)
    electric = 0.5 * (electric_zero + electric_one)
    curl = 0.5 * (curl_zero + curl_one) / length_scale_value

    physical_coordinates = coordinates_zero * length_scale_value

    mu_diagonal = evaluate_diagonal_coefficient(
        mu_r,
        physical_coordinates[0],
        physical_coordinates[1],
        name="mu_r",
    )
    if np.any(np.abs(mu_diagonal) == 0.0):
        raise ModeProjectionError("mu_r must be nonzero at every monitor point.")
    magnetic = curl / (1j * omega_value * MU_0 * mu_diagonal)

    weights = np.asarray(side_zero.dx, dtype=float) * length_scale_value
    if weights.shape != coordinates_zero[0].shape or np.any(weights <= 0.0):
        raise ModeProjectionError("Facet quadrature produced non-positive monitor weights.")

    x_flat = physical_coordinates[0].reshape(-1)
    order = np.argsort(x_flat, kind="stable")
    return MonitorSamples(
        x=x_flat[order],
        weights=weights.reshape(-1)[order],
        E=electric.reshape(3, -1)[:, order],
        H=magnetic.reshape(3, -1)[:, order],
        z=z_value,
    )


def sample_horizontal_monitor(
    basis: Basis,
    coefficients: ArrayLike,
    *,
    x: float,
    ky: complex = 0.0,
    omega: float,
    mu_r: ConstitutiveCoefficient = 1.0,
    length_scale: float = 1.0,
    intorder: int = 4,
    tolerance: float | None = None,
) -> HorizontalMonitorSamples:
    """Sample a mesh-conforming physical ``x = constant`` side monitor."""

    _validate_mixed_basis(basis)
    x_value = _finite_real(x, name="x")
    omega_value = _finite_real(omega, name="omega", positive=True)
    length_scale_value = _finite_real(length_scale, name="length_scale", positive=True)
    ky_value = complex(ky)
    if not np.isfinite((ky_value.real, ky_value.imag)).all():
        raise ModeProjectionError("ky must be finite.")
    if isinstance(intorder, bool) or int(intorder) != intorder or intorder < 1:
        raise ModeProjectionError("intorder must be a positive integer.")
    coordinates = np.asarray(basis.mesh.p, dtype=float)
    mesh_tolerance = (
        _default_coordinate_tolerance(coordinates)
        if tolerance is None
        else _finite_real(tolerance, name="tolerance", positive=True) / length_scale_value
    )
    values = np.asarray(coefficients, dtype=np.complex128)
    if values.shape != (basis.N,) or not np.isfinite(values).all():
        raise ModeProjectionError(
            f"coefficients must be finite with shape ({basis.N},); received {values.shape}."
        )
    mesh_x = x_value / length_scale_value
    facets = _side_monitor_facets(
        basis, mesh_x, mesh_tolerance, length_scale=length_scale_value
    )
    kwargs = dict(facets=facets, dofs=basis.dofs, intorder=int(intorder))
    side_zero = InteriorFacetBasis(basis.mesh, basis.elem, side=0, **kwargs)
    side_one = InteriorFacetBasis(basis.mesh, basis.elem, side=1, **kwargs)
    coordinates_zero = np.asarray(side_zero.global_coordinates(), dtype=float)
    coordinates_one = np.asarray(side_one.global_coordinates(), dtype=float)
    if not np.allclose(coordinates_zero, coordinates_one, rtol=0.0, atol=mesh_tolerance):
        raise ModeProjectionError("Interior side-facet quadratures are inconsistent.")
    if not np.all(np.abs(coordinates_zero[0] - mesh_x) <= mesh_tolerance):
        raise ModeProjectionError("Selected side quadrature is not on the requested x line.")
    scaled_ky = ky_value * length_scale_value
    electric_zero, curl_zero = _facet_fields(side_zero, values, scaled_ky)
    electric_one, curl_one = _facet_fields(side_one, values, scaled_ky)
    electric = 0.5 * (electric_zero + electric_one)
    curl = 0.5 * (curl_zero + curl_one) / length_scale_value
    physical_coordinates = coordinates_zero * length_scale_value
    mu_diagonal = evaluate_diagonal_coefficient(
        mu_r,
        physical_coordinates[0],
        physical_coordinates[1],
        name="mu_r",
    )
    if np.any(np.abs(mu_diagonal) == 0.0):
        raise ModeProjectionError("mu_r must be nonzero at every side-monitor point.")
    magnetic = curl / (1j * omega_value * MU_0 * mu_diagonal)
    weights = np.asarray(side_zero.dx, dtype=float) * length_scale_value
    if weights.shape != coordinates_zero[1].shape or np.any(weights <= 0.0):
        raise ModeProjectionError("Facet quadrature produced non-positive side weights.")
    z_flat = physical_coordinates[1].reshape(-1)
    order = np.argsort(z_flat, kind="stable")
    return HorizontalMonitorSamples(
        z=z_flat[order],
        weights=weights.reshape(-1)[order],
        E=electric.reshape(3, -1)[:, order],
        H=magnetic.reshape(3, -1)[:, order],
        x=x_value,
    )


__all__ = [
    "HorizontalMonitorSamples",
    "MonitorSamples",
    "sample_horizontal_monitor",
    "sample_vertical_monitor",
]

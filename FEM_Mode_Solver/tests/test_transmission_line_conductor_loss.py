"""Focused checks for quasi-TEM perturbative conductor surface loss."""

from __future__ import annotations

from math import pi, sqrt

import numpy as np
import pytest

from FEM_Mode_Solver import TransmissionLineCalculator
from FEM_Mode_Solver.constants import MU_0
from FEM_Mode_Solver.transmission_lines.electrostatics import solve_quasi_tem


FREQUENCY = 10.0e9
COPPER_CONDUCTIVITY = 5.96e7


def _coax_solution(conductivity: float | None):
    calculator = TransmissionLineCalculator.coaxial(
        frequency=FREQUENCY,
        inner_radius=0.50e-3,
        outer_radius=1.67e-3,
        outer_conductor_thickness=0.15e-3,
        epsilon_r=2.25,
        loss_tangent=0.0,
        metal_conductivity=conductivity,
    )
    calculator.discretize(max_element_size=0.30e-3)
    return calculator, solve_quasi_tem(
        calculator._built,
        frequency=calculator.frequency,
    )


@pytest.mark.gmsh
def test_none_conductivity_preserves_the_pec_extraction_exactly() -> None:
    _, solution = _coax_solution(None)

    expected_neff = np.sqrt(
        solution.capacitance_per_length
        / solution.vacuum_capacitance_per_length
    )
    expected_impedance = np.sqrt(
        solution.external_inductance_per_length
        / solution.capacitance_per_length
    )

    assert solution.resistance_per_length == 0.0
    assert solution.surface_resistance == 0.0
    assert solution.inductance_per_length == solution.external_inductance_per_length
    assert solution.neff == expected_neff
    assert solution.characteristic_impedance == expected_impedance
    assert solution.series_impedance_per_length == complex(
        0.0,
        2.0 * pi * FREQUENCY * solution.inductance_per_length,
    )
    assert solution.metadata["conductor_geometry_factor_per_length"] == 0.0
    assert all(
        factor == 0.0
        for factor in solution.metadata["conductor_geometry_factors"].values()
    )


@pytest.mark.gmsh
def test_coaxial_projected_surface_loss_matches_the_exact_geometry_factor() -> None:
    calculator, solution = _coax_solution(COPPER_CONDUCTIVITY)
    spec = calculator.spec
    surface_resistance = sqrt(
        pi * FREQUENCY * MU_0 / COPPER_CONDUCTIVITY
    )
    exact_geometry_factor = (1.0 / (2.0 * pi)) * (
        1.0 / spec.inner_radius + 1.0 / spec.outer_radius
    )
    exact_resistance = surface_resistance * exact_geometry_factor

    assert solution.surface_resistance == pytest.approx(
        surface_resistance, rel=2e-15
    )
    assert solution.metadata["conductor_geometry_factor_per_length"] == pytest.approx(
        exact_geometry_factor,
        rel=5e-3,
    )
    assert solution.resistance_per_length == pytest.approx(
        exact_resistance,
        rel=5e-3,
    )
    assert set(solution.metadata["conductor_geometry_factors"]) == {
        "signal",
        "outer_conductor",
    }


@pytest.mark.gmsh
def test_finite_conductivity_uses_the_full_passive_rlgc_relations() -> None:
    _, solution = _coax_solution(COPPER_CONDUCTIVITY)
    omega = 2.0 * pi * FREQUENCY
    series = complex(
        solution.resistance_per_length,
        omega * solution.inductance_per_length,
    )
    shunt = complex(
        solution.conductance_per_length,
        omega * solution.capacitance_per_length.real,
    )
    expected_beta = np.sqrt(-series * shunt)
    if expected_beta.real < 0.0:
        expected_beta = -expected_beta
    expected_neff = expected_beta * 299_792_458.0 / omega
    expected_impedance = np.sqrt(series / shunt)
    if expected_impedance.real < 0.0:
        expected_impedance = -expected_impedance

    assert solution.series_impedance_per_length == pytest.approx(series)
    assert solution.shunt_admittance_per_length == pytest.approx(shunt)
    assert solution.inductance_per_length == pytest.approx(
        solution.external_inductance_per_length
        + solution.resistance_per_length / omega,
        rel=2e-15,
    )
    assert solution.neff == pytest.approx(expected_neff, rel=2e-15)
    assert solution.characteristic_impedance == pytest.approx(
        expected_impedance, rel=2e-15
    )
    assert solution.neff.imag < 0.0
    assert solution.metadata["conductor_loss_per_length"] > 0.0
    assert solution.metadata["power_balance"]["relative_residual"] < 1e-10


@pytest.mark.gmsh
def test_microstrip_return_ground_is_lossy_but_outer_pec_is_not_metal() -> None:
    calculator = TransmissionLineCalculator.microstrip(
        frequency=FREQUENCY,
        trace_width=3.0e-3,
        substrate_height=1.524e-3,
        conductor_thickness=35.0e-6,
        epsilon_r=3.55,
        loss_tangent=0.0027,
        metal_conductivity=COPPER_CONDUCTIVITY,
    )
    calculator.discretize(max_element_size=0.55e-3)
    solution = solve_quasi_tem(
        calculator._built,
        frequency=calculator.frequency,
    )
    factors = solution.metadata["conductor_geometry_factors"]

    assert set(factors) == {"signal", "ground"}
    assert factors["signal"] > 0.0
    assert factors["ground"] > 0.0
    assert "outer_pec" not in factors
    assert solution.resistance_per_length > 0.0
    assert solution.conductance_per_length > 0.0


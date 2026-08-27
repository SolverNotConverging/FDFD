import numpy as np
import pytest
from skfem import MeshTri

from wavefem.exceptions import ConfigurationError
from wavefem.fem import MaxwellParameters, assemble_mixed_system
from wavefem.sources import assemble_equivalent_source, solve_scattered_pec


def incident(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(x, dtype=complex)
    return np.asarray((zeros, np.sin(np.pi * x) * np.sin(np.pi * z), zeros))


def test_identical_actual_and_background_material_has_exactly_zero_source() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8))
    system = assemble_mixed_system(mesh, MaxwellParameters(k0=2.0, ky=0.4, eps_r=2.0))
    source = assemble_equivalent_source(system, eps_background=2.0, incident=incident)
    assert source.is_zero
    assert source.active_quadrature_fraction == 0.0
    assert source.maximum_delta_eps == 0.0


def _solve(delta: float) -> tuple[float, float]:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 13), np.linspace(0.0, 1.0, 13))

    def eps_actual(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        perturbation = (x > 0.35) & (x < 0.65) & (z > 0.35) & (z < 0.65)
        return 2.0 + delta * perturbation

    system = assemble_mixed_system(
        mesh,
        MaxwellParameters(k0=1.7, ky=0.3, eps_r=eps_actual),
        intorder=4,
    )
    solution = solve_scattered_pec(
        system, eps_background=2.0, incident=incident
    )
    et, ey = solution.field.interpolate()
    field_norm = float(
        np.sqrt(
            np.sum(
                system.basis.dx
                * (np.abs(et[0]) ** 2 + np.abs(ey) ** 2 + np.abs(et[1]) ** 2)
            )
        )
    )
    return field_norm, field_norm**2


@pytest.mark.slow
def test_weak_perturbation_field_and_squared_norm_scaling() -> None:
    field_1, squared_norm_1 = _solve(1e-4)
    field_2, squared_norm_2 = _solve(2e-4)
    assert field_2 / field_1 == pytest.approx(2.0, rel=5e-3)
    assert squared_norm_2 / squared_norm_1 == pytest.approx(4.0, rel=1e-2)


def test_permeability_perturbation_is_rejected_explicitly() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    system = assemble_mixed_system(
        mesh, MaxwellParameters(k0=2.0, ky=0.2, eps_r=2.0, mu_r=1.1)
    )
    with pytest.raises(ConfigurationError, match="permittivity perturbations only"):
        assemble_equivalent_source(
            system,
            eps_background=2.0,
            mu_background=1.0,
            incident=incident,
        )

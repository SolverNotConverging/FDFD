from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import norm as sparse_norm
from skfem import Functional, MeshTri, asm

from wavefem.fem import (
    MaxwellParameters,
    assemble_load_vector,
    assemble_maxwell_matrix,
    assemble_mixed_system,
    create_mixed_basis,
    relative_hermiticity_error,
    solve_homogeneous_pec,
)
from wavefem.operators import electric_field_vector, modified_curl


_PI = np.pi
_KY = 0.7
_K0 = 1.3
_EPS_R = 2.0
_KAPPA = _K0**2 * _EPS_R


def _mesh(cells_per_axis: int) -> MeshTri:
    coordinates = np.linspace(0.0, 1.0, cells_per_axis + 1)
    return MeshTri.init_tensor(coordinates, coordinates)


def _exact_field(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.cos(_PI * x) * np.sin(_PI * z),
            np.sin(_PI * x) * np.sin(_PI * z),
            2.0 * np.sin(_PI * x) * np.cos(_PI * z),
        )
    )


def _exact_modified_curl(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    sin_x = np.sin(_PI * x)
    cos_x = np.cos(_PI * x)
    sin_z = np.sin(_PI * z)
    cos_z = np.cos(_PI * z)
    return np.stack(
        (
            (2j * _KY - _PI) * sin_x * cos_z,
            -_PI * cos_x * cos_z,
            (_PI - 1j * _KY) * cos_x * sin_z,
        )
    )


def _manufactured_source(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    sin_x = np.sin(_PI * x)
    cos_x = np.cos(_PI * x)
    sin_z = np.sin(_PI * z)
    cos_z = np.cos(_PI * z)
    return np.stack(
        (
            (_KY**2 - _PI**2 + 1j * _PI * _KY - _KAPPA)
            * cos_x
            * sin_z,
            (2.0 * _PI**2 - 3j * _PI * _KY - _KAPPA)
            * sin_x
            * sin_z,
            (_PI**2 + 2.0 * _KY**2 + 1j * _PI * _KY - 2.0 * _KAPPA)
            * sin_x
            * cos_z,
        )
    )


@Functional
def _combined_l2_error(w: object) -> object:
    et, ey = w.uh
    difference = electric_field_vector(et, ey) - _exact_field(w.x[0], w.x[1])
    return np.sum(np.abs(difference) ** 2, axis=0)


@Functional
def _ey_l2_error(w: object) -> object:
    _, ey = w.uh
    exact_ey = np.sin(_PI * w.x[0]) * np.sin(_PI * w.x[1])
    return np.abs(ey - exact_ey) ** 2


@Functional
def _modified_curl_error(w: object) -> object:
    et, ey = w.uh
    difference = modified_curl(et, ey, _KY) - _exact_modified_curl(
        w.x[0], w.x[1]
    )
    return np.sum(np.abs(difference) ** 2, axis=0)


def test_lossless_nonzero_ky_matrix_is_complex_and_hermitian() -> None:
    parameters = MaxwellParameters(k0=_K0, ky=_KY, eps_r=_EPS_R, mu_r=1.0)
    system = assemble_mixed_system(_mesh(8), parameters, intorder=4)

    assert system.matrix.dtype == np.dtype(np.complex128)
    assert system.matrix.shape == (system.ndofs, system.ndofs)
    assert np.max(np.abs(system.matrix.data.imag)) > 0.0
    assert relative_hermiticity_error(system.matrix) < 1e-13

    split_indices = system.basis.split_indices()
    assert len(split_indices) == 2
    assert len(split_indices[0]) + len(split_indices[1]) == system.ndofs
    # Composite DOFs are grouped topologically, not by element declaration.
    assert not np.array_equal(
        split_indices[0], np.arange(len(split_indices[0]), dtype=np.int32)
    )


def test_scalar_diagonal_and_callback_materials_agree() -> None:
    basis = create_mixed_basis(_mesh(4), intorder=4)
    scalar = MaxwellParameters(k0=_K0, ky=_KY, eps_r=2.0, mu_r=1.5)
    diagonal = MaxwellParameters(
        k0=_K0,
        ky=_KY,
        eps_r=(2.0, 2.0, 2.0),
        mu_r=np.array([1.5, 1.5, 1.5]),
    )
    callback = MaxwellParameters(
        k0=_K0,
        ky=_KY,
        eps_r=lambda x, z: 2.0 + 0.0 * x,
        mu_r=lambda x, z: np.stack(
            (1.5 + 0.0 * x, 1.5 + 0.0 * x, 1.5 + 0.0 * x)
        ),
    )

    matrices = [
        assemble_maxwell_matrix(basis, parameters)
        for parameters in (scalar, diagonal, callback)
    ]
    reference_norm = sparse_norm(matrices[0])
    assert sparse_norm(matrices[0] - matrices[1]) / reference_norm < 1e-14
    assert sparse_norm(matrices[0] - matrices[2]) / reference_norm < 1e-14

    anisotropic = assemble_maxwell_matrix(
        basis,
        MaxwellParameters(
            k0=_K0,
            ky=_KY,
            eps_r=(2.0, 2.2, 2.4),
            mu_r=(1.0, 1.1, 1.2),
        ),
    )
    assert sparse_norm(anisotropic - matrices[0]) > 0.0
    assert relative_hermiticity_error(anisotropic) < 1e-13


def test_si_mesh_nondimensionalization_reproduces_unit_problem() -> None:
    unit_mesh = _mesh(6)
    unit_parameters = MaxwellParameters(k0=_K0, ky=_KY, eps_r=2.0, mu_r=1.5)
    unit = assemble_mixed_system(unit_mesh, unit_parameters, intorder=4)

    metre_scale = 1e-6
    physical_mesh = unit_mesh.scaled(float(metre_scale))
    physical_parameters = MaxwellParameters(
        k0=_K0 / metre_scale,
        ky=_KY / metre_scale,
        eps_r=lambda x, z: 2.0 + 0.0 * x,
        mu_r=1.5,
    )
    scaled = assemble_mixed_system(
        physical_mesh,
        physical_parameters,
        intorder=4,
        length_scale=metre_scale,
    )

    assert scaled.dimensionless_k0 == pytest.approx(_K0)
    assert scaled.dimensionless_ky == pytest.approx(_KY)
    np.testing.assert_allclose(scaled.physical_coordinates(), metre_scale * unit.basis.global_coordinates())
    relative = sparse_norm(scaled.matrix - unit.matrix) / sparse_norm(unit.matrix)
    assert relative < 1e-13


def test_manufactured_solution_converges_in_mixed_space() -> None:
    parameters = MaxwellParameters(k0=_K0, ky=_KY, eps_r=_EPS_R, mu_r=1.0)
    combined_errors: list[float] = []
    ey_errors: list[float] = []
    curl_errors: list[float] = []

    for cells_per_axis in (4, 8, 16, 32):
        system = assemble_mixed_system(
            _mesh(cells_per_axis),
            parameters,
            intorder=8,
        )
        load = assemble_load_vector(system.basis, _manufactured_source)
        solution = solve_homogeneous_pec(system, load)

        combined_errors.append(
            float(
                np.sqrt(
                    asm(
                        _combined_l2_error,
                        system.basis,
                        uh=solution.coefficients,
                    )
                )
            )
        )
        ey_errors.append(
            float(
                np.sqrt(
                    asm(_ey_l2_error, system.basis, uh=solution.coefficients)
                )
            )
        )
        curl_errors.append(
            float(
                np.sqrt(
                    asm(
                        _modified_curl_error,
                        system.basis,
                        uh=solution.coefficients,
                    )
                )
            )
        )

    combined = np.asarray(combined_errors)
    ey = np.asarray(ey_errors)
    curl = np.asarray(curl_errors)
    assert np.all(np.diff(combined) < 0.0)
    assert np.all(np.diff(ey) < 0.0)
    assert np.all(np.diff(curl) < 0.0)

    combined_rates = np.log2(combined[:-1] / combined[1:])
    ey_rates = np.log2(ey[:-1] / ey[1:])
    curl_rates = np.log2(curl[:-1] / curl[1:])
    assert combined_rates[-1] > 0.9
    assert ey_rates[-1] > 1.8
    assert curl_rates[-1] > 0.9
    assert combined[-1] < 0.06
    assert ey[-1] < 0.002
    assert curl[-1] < 0.14

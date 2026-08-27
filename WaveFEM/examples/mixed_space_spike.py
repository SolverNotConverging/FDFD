"""Stage-A feasibility spike for the conforming 2.5D Maxwell mixed space.

Run from the project directory with::

    PYTHONPATH=src python examples/mixed_space_spike.py
"""

from __future__ import annotations

import numpy as np
from skfem import Functional, MeshTri, asm

from wavefem.fem import (
    MaxwellParameters,
    assemble_load_vector,
    assemble_mixed_system,
    relative_hermiticity_error,
    solve_homogeneous_pec,
)
from wavefem.operators import electric_field_vector, modified_curl


PI = np.pi
KY = 0.7
K0 = 1.3
EPS_R = 2.0
KAPPA = K0**2 * EPS_R


def exact_field(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Manufactured electric field in physical ``(x, y, z)`` order."""

    return np.stack(
        (
            np.cos(PI * x) * np.sin(PI * z),
            np.sin(PI * x) * np.sin(PI * z),
            2.0 * np.sin(PI * x) * np.cos(PI * z),
        )
    )


def exact_curl(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    sin_x, cos_x = np.sin(PI * x), np.cos(PI * x)
    sin_z, cos_z = np.sin(PI * z), np.cos(PI * z)
    return np.stack(
        (
            (2j * KY - PI) * sin_x * cos_z,
            -PI * cos_x * cos_z,
            (PI - 1j * KY) * cos_x * sin_z,
        )
    )


def manufactured_source(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Strong forcing ``curl_ky curl_ky E - k0^2 eps_r E``."""

    sin_x, cos_x = np.sin(PI * x), np.cos(PI * x)
    sin_z, cos_z = np.sin(PI * z), np.cos(PI * z)
    return np.stack(
        (
            (KY**2 - PI**2 + 1j * PI * KY - KAPPA) * cos_x * sin_z,
            (2.0 * PI**2 - 3j * PI * KY - KAPPA) * sin_x * sin_z,
            (PI**2 + 2.0 * KY**2 + 1j * PI * KY - 2.0 * KAPPA)
            * sin_x
            * cos_z,
        )
    )


@Functional
def l2_error(w: object) -> object:
    et, ey = w.uh
    difference = electric_field_vector(et, ey) - exact_field(w.x[0], w.x[1])
    return np.sum(np.abs(difference) ** 2, axis=0)


@Functional
def curl_error(w: object) -> object:
    et, ey = w.uh
    difference = modified_curl(et, ey, KY) - exact_curl(w.x[0], w.x[1])
    return np.sum(np.abs(difference) ** 2, axis=0)


def main() -> None:
    parameters = MaxwellParameters(k0=K0, ky=KY, eps_r=EPS_R, mu_r=1.0)
    errors: list[tuple[int, int, float, float, float]] = []

    for cells in (4, 8, 16, 32):
        points = np.linspace(0.0, 1.0, cells + 1)
        mesh = MeshTri.init_tensor(points, points)
        system = assemble_mixed_system(mesh, parameters, intorder=8)
        load = assemble_load_vector(system.basis, manufactured_source)
        solution = solve_homogeneous_pec(system, load)
        field_error = float(
            np.sqrt(asm(l2_error, system.basis, uh=solution.coefficients))
        )
        derivative_error = float(
            np.sqrt(asm(curl_error, system.basis, uh=solution.coefficients))
        )
        errors.append(
            (
                cells,
                system.ndofs,
                field_error,
                derivative_error,
                relative_hermiticity_error(system.matrix),
            )
        )

    print(" cells    ndofs       ||E-Eh||       ||curl-curlh||    Hermitian residual")
    for cells, ndofs, field_error, derivative_error, hermitian_error in errors:
        print(
            f" {cells:5d} {ndofs:8d}  {field_error:14.6e}  "
            f"{derivative_error:14.6e}  {hermitian_error:14.3e}"
        )

    field_rate = np.log2(errors[-2][2] / errors[-1][2])
    curl_rate = np.log2(errors[-2][3] / errors[-1][3])
    print(f"Final refinement rates: field={field_rate:.3f}, curl={curl_rate:.3f}")


if __name__ == "__main__":
    main()

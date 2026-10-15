"""Periodic solvers: Maxwell-consistent fields, exact pencil eigenvalues, PML, FEM reference."""

import numpy as np
import pytest
from scipy import linalg
from scipy.sparse import csr_matrix

import fdfd
from fdfd import Material, materials
from fdfd.core import eigs
from fdfd.core.constants import EPSILON_0, MU_0
from fdfd.periodic._kernel_2d import PeriodicKernel2D
from fdfd.periodic._kernel_3d import PeriodicKernel3D

LOSSY = Material(name="fill", epsilon=2.25 - 0.02j)
OPTIONS = dict(device="cpu", max_restarts=12, random_seed=0)


def flat(values):
    return np.column_stack([values[..., i].ravel(order="F") for i in range(values.shape[-1])])


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_2d_fields_obey_the_longitudinal_maxwell_equation(polarization, tmp_path):
    solver = fdfd.PeriodicModeSolver2D(
        frequency=30e9, x_range=(0.0, 8e-3), z_range=(0.0, 6e-3), background_material=LOSSY
    )
    solver.mesh(resolution=(10, 8))
    result = solver.solve(num_modes=2, polarization=polarization, guess=1.4)
    kernel = solver._kernel
    assert result.metadata["polarizations"] == (polarization,) * 2
    assert set(result.fields) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    for name, values in result.fields.items():
        assert values.shape == (*kernel.field_shapes[name], 2)
        assert tuple(map(len, result.field_coordinates[name])) == values.shape[:-1]
        assert np.isfinite(values).all()
    materials_, masks = kernel.materials_and_masks()
    if polarization == "TM":
        derivative = kernel.DHX_HY_TO_EZ @ flat(result.fields["Hy"])
        field = (
            1j
            * kernel.omega
            * EPSILON_0
            * materials_["eps_zz"].ravel(order="F")[:, None]
            * flat(result.fields["Ez"])
        )
        # PEC carries surface current: the bulk law holds on free Ez samples.
        constrained = masks["Ez"].ravel(order="F")
        np.testing.assert_array_equal(field[constrained], 0.0)
        derivative, field = derivative[~constrained], field[~constrained]
        inactive = ("Ey", "Hx", "Hz")
    else:
        derivative = kernel.DEX_EY_TO_HZ @ flat(result.fields["Ey"])
        field = (
            -1j
            * kernel.omega
            * MU_0
            * materials_["mu_zz"].ravel(order="F")[:, None]
            * flat(result.fields["Hz"])
        )
        inactive = ("Ex", "Ez", "Hy")
    assert np.linalg.norm(derivative) > 0
    np.testing.assert_allclose(field, derivative, rtol=1e-12, atol=1e-12)
    for name in inactive:
        assert not np.any(result.fields[name])
    restored = fdfd.load(result.save(tmp_path / "modes.h5"))
    for name in result.fields:
        np.testing.assert_array_equal(restored.fields[name], result.fields[name])
    for name in ("epsilon", "mu", "conductor"):
        np.testing.assert_array_equal(
            restored.metadata["material_background"][name], result.metadata["material_background"][name]
        )


@pytest.mark.parametrize(
    ("polarization", "boundary", "name"), [("TM", materials.PEC, "Ez"), ("TE", materials.PMC, "Hz")]
)
def test_longitudinal_field_vanishes_on_conductor_constraints(polarization, boundary, name):
    solver = fdfd.PeriodicModeSolver2D(frequency=30e9, x_range=(0.0, 8e-3), z_range=(0.0, 6e-3))
    solver.add_rectangle(x_range=(0.0, 800e-6), z_range=(0.0, 6e-3), material=boundary)
    solver.mesh(resolution=(10, 8), subpixels=1)
    result = solver.solve(num_modes=1, polarization=polarization, guess=0.8)
    constrained = solver._kernel.materials_and_masks()[1][name]
    assert np.any(constrained)
    np.testing.assert_array_equal(result.fields[name][constrained], 0.0)


def test_3d_returns_all_six_fields_obeying_maxwell():
    solver = fdfd.PeriodicModeSolver3D(
        frequency=30e9,
        x_range=(0.0, 8e-3),
        y_range=(0.0, 7e-3),
        z_range=(0.0, 6e-3),
        background_material=LOSSY,
    )
    solver.mesh(resolution=(4, 4, 4))
    result = solver.solve(num_modes=2, guess=1.3)
    kernel = solver._kernel
    for name, values in result.fields.items():
        assert values.shape == (*kernel.field_shapes[name], 2) and np.isfinite(values).all()
    f = {name: flat(values) for name, values in result.fields.items()}
    materials_, _ = kernel.materials_and_masks()
    electric_curl = kernel.DHX_HY_TO_EZ @ f["Hy"] - kernel.DHY_HX_TO_EZ @ f["Hx"]
    magnetic_curl = kernel.DEX_EY_TO_HZ @ f["Ey"] - kernel.DEY_EX_TO_HZ @ f["Ex"]
    np.testing.assert_allclose(
        1j * kernel.omega * EPSILON_0 * materials_["eps_zz"].ravel(order="F")[:, None] * f["Ez"],
        electric_curl,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        -1j * kernel.omega * MU_0 * materials_["mu_zz"].ravel(order="F")[:, None] * f["Hz"],
        magnetic_curl,
        atol=1e-12,
    )
    assert result.metadata["material_background"]["epsilon"].shape == (3, 4, 4, 4)


def test_both_polarizations_are_merged_nearest_to_the_guess():
    solver = fdfd.PeriodicModeSolver2D(
        frequency=30e9, x_range=(0.0, 8e-3), z_range=(0.0, 6e-3), background_material=LOSSY
    )
    solver.mesh(resolution=(10, 8))
    te = solver.solve(num_modes=3, polarization="TE", guess=1.4)
    tm = solver.solve(num_modes=3, polarization="TM", guess=1.4)
    both = solver.solve(num_modes=3, polarization="both", guess=1.4)
    pool = np.r_[te.neff, tm.neff]
    expected = pool[np.argsort(abs(pool - 1.4), kind="stable")][:3]
    np.testing.assert_allclose(np.sort_complex(both.neff), np.sort_complex(expected), rtol=1e-10)
    assert set(both.metadata["polarizations"]) <= {"TE", "TM"}


def two_d_kernel():
    kernel = PeriodicKernel2D(20e9, 10e-3, 8e-3, 8, 7, boundary="pmc")
    kernel.apply_fractional_material(10.2, 1.0, np.ones((2, 7)), slice(0, 2), slice(0, 7))
    patch = np.zeros((8, 7), dtype=bool)
    patch[2, 1:3] = True
    kernel.add_pec(patch)
    return kernel


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_2d_arpack_and_refined_match_dense_qz_with_a_conductor(polarization):
    kernel = two_d_kernel()
    materials_, masks = kernel.materials_and_masks()
    A, B, free = kernel._system(polarization, materials_, masks)
    A, B = A[free][:, free], B[free][:, free]
    exact = linalg.eigvals(A.toarray(), B.toarray())
    shift = 0.3j * kernel.k0
    for method in ("arpack", "refined"):
        gamma, _, residuals = kernel.solve(
            4, shift, polarization=polarization, tol=1e-9, ncv=40, method=method, **OPTIONS
        )
        assert np.max(residuals) < 1e-7
        for value in gamma:
            assert np.min(np.abs(exact - value)) < 1e-6 * kernel.k0
    if polarization == "TM":
        # Reciprocity survives a finite PEC patch with different E/H masks.
        for value in exact[np.isfinite(exact)]:
            assert np.min(np.abs(exact + value)) < 1e-7 * max(1.0, abs(value))


def test_unified_eigensolver_handles_an_arbitrary_singular_mass_matrix():
    rng = np.random.default_rng(7)
    A = csr_matrix(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    B = csr_matrix(np.diag(np.r_[np.arange(1.0, 12.0), 0.0])) @ csr_matrix(rng.normal(size=(12, 12)))
    exact = linalg.eigvals(A.toarray(), B.toarray())
    pairs = eigs.solve(A, B, k=3, sigma=0.2j, tol=1e-10, ncv=12)
    for value in pairs.values:
        assert np.min(np.abs(exact - value)) < 1e-8
    assert np.max(pairs.residuals) < 1e-8


def test_3d_constrained_pencil_matches_dense_qz():
    kernel = PeriodicKernel3D(20e9, 10e-3, 8e-3, 6e-3, 3, 2, 3)
    block = np.zeros((3, 2, 3), dtype=bool)
    block[1, 0, 0] = True
    kernel.add_pec(block)
    A, B, free = kernel.system(*kernel.materials_and_masks())
    exact = linalg.eigvals(A[free][:, free].toarray(), B[free][:, free].toarray())
    for method in ("arpack", "refined"):
        gamma, _, residuals = kernel.solve(2, 0.3j * kernel.k0, tol=1e-9, ncv=40, method=method, **OPTIONS)
        assert np.max(residuals) < 1e-7
        for value in gamma:
            assert np.min(np.abs(exact - value)) < 1e-6 * kernel.k0


def test_leaky_wave_cell_matches_refined_fem_reference():
    solver = fdfd.PeriodicModeSolver2D(frequency=20e9, x_range=(0.0, 10e-3), z_range=(0.0, 8e-3))
    solver.add_rectangle(
        x_range=(0.0, 1.27e-3), z_range=(0.0, 8e-3), material=Material(name="substrate", epsilon=10.2)
    )
    solver.add_rectangle(x_range=(1.27e-3, 1.32e-3), z_range=(1e-3, 2e-3), material=materials.PEC)
    solver.add_pml(thickness=2.5e-3, sides="x+")
    solver.mesh(resolution=(200, 80))
    result = solver.solve(num_modes=4, polarization="TM", guess=0.0, tol=1e-9, ncv=36)
    assert result.info.converged
    # FEM P1 (100 um edges, no refinement); the benchmark records convergence.
    expected = np.array(
        [
            0.0143483929443405 + 0.0926042962539239j,
            -0.0143483929443405 - 0.0926042962539239j,
            0.5791833804852682 - 0.5261466284273147j,
            -0.5791833804852682 + 0.5261466284273147j,
        ]
    )
    for value in expected:
        assert np.min(np.abs(result.neff - value)) < 0.035
    for value in result.neff:
        assert np.min(np.abs(result.neff + value)) < 1e-8


def test_uniform_tem_mode_and_dimensionless_pml():
    solver = fdfd.PeriodicModeSolver2D(
        frequency=20e9,
        x_range=(0.0, 10e-3),
        z_range=(0.0, 8e-3),
        background_material=Material(name="fill", epsilon=2.25),
    )
    solver.mesh(resolution=(12, 9))
    np.testing.assert_allclose(
        solver.solve(num_modes=1, polarization="TM", guess=1.48).neff, [1.5], atol=1e-9
    )
    for frequency in (10e9, 30e9):
        solver = fdfd.PeriodicModeSolver2D(frequency=frequency, x_range=(0.0, 10e-3), z_range=(0.0, 8e-3))
        solver.add_pml(thickness=2.5e-3, sides="x+", strength=5.0)
        solver.mesh(resolution=(20, 8))
        np.testing.assert_allclose(solver._kernel.cell_eps_r_yy[-1], 1.0 - 5j * 0.9**3)


def test_periodic_axis_cannot_carry_a_pml():
    solver = fdfd.PeriodicModeSolver3D(
        frequency=20e9, x_range=(0.0, 1.0), y_range=(0.0, 1.0), z_range=(0.0, 1.0)
    )
    with pytest.raises(fdfd.BackendCapabilityError):
        solver.add_pml(thickness=0.1, sides="z")


def test_failed_residual_check_raises_solver_error(monkeypatch):
    solver = fdfd.PeriodicModeSolver2D(frequency=20e9, x_range=(0.0, 10e-3), z_range=(0.0, 8e-3))
    solver.mesh(resolution=(8, 6))
    real_solve = eigs.solve

    def inaccurate(*args, **kwargs):
        pairs = real_solve(*args, **kwargs)
        return eigs.Eigenpairs(pairs.values, pairs.vectors, pairs.residuals + 1e-3)

    monkeypatch.setattr(eigs, "solve", inaccurate)
    with pytest.raises(fdfd.SolverError, match="backward-error"):
        solver.solve(num_modes=1, guess=0.5)


@pytest.mark.parametrize("dimension", [2, 3])
def test_materials_and_coordinates_share_the_z_lattice(dimension):
    settings = dict(frequency=20e9, x_range=(0.0, 10e-3), z_range=(0.0, 8e-3))
    if dimension == 2:
        solver = fdfd.PeriodicModeSolver2D(**settings)
        solver.mesh(resolution=(6, 8))
    else:
        solver = fdfd.PeriodicModeSolver3D(y_range=(0.0, 6e-3), **settings)
        solver.mesh(resolution=(6, 5, 8))
    kernel = solver._kernel
    z_cells = (np.arange(8) + 0.5) * 8e-3 / 8
    for tensor in kernel.cell_tensors():
        for values in tensor:
            values[:] = 3 + np.sin(2 * np.pi * z_cells / 8e-3)
    result = solver.solve(num_modes=1, guess=0.8, tol=1e-9)
    materials_, _ = kernel.materials_and_masks()
    for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        z_node = field in ("Ex", "Ey", "Hz")
        expected_z = (np.arange(8) + (0.0 if z_node else 0.5)) * 8e-3 / 8
        np.testing.assert_allclose(result.field_coordinates[field][-1], expected_z, atol=1e-15)
        profile = 3 + np.sin(2 * np.pi * expected_z / 8e-3) * (np.cos(np.pi / 8) if z_node else 1.0)
        material = materials_[("eps_" if field[0] == "E" else "mu_") + field[1].lower() * 2]
        # Outer-wall replacement values are excluded; compare interior traces.
        interior = material[1:-1] if dimension == 2 else material[1:-1, 1:-1]
        np.testing.assert_allclose(interior, np.broadcast_to(profile, interior.shape))

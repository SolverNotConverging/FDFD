"""Periodic eigenvalues satisfy Maxwell and preserve reciprocal mode pairs."""
import numpy as np
import pytest
from scipy import linalg

from fdfd_periodic_modes import Material, PeriodicModeSolver2D, materials
from fdfd_periodic_modes._eigensolve import generalized_eigs, eigenpair_residuals
from fdfd_periodic_modes.solver_2d import _PeriodicModeSolver2D


@pytest.mark.parametrize('polarization', ['TE', 'TM'])
def test_scipy_and_refined_match_dense_qz_with_conductor(polarization):
    solver = _PeriodicModeSolver2D(polarization, 20e9, 10e-3, 8e-3, 8, 7, 4)
    solver.add_rectangle(10.2, 1., (0, 2), (0, 7))
    solver.add_pec((2, 3), (1, 3))
    tensors = solver._effective_materials_and_masks()
    matrices = (solver._build_tm_system(tensors[0], tensors[2], tensors[4], tensors[8])
                if polarization == 'TM' else
                solver._build_te_system(tensors[1], tensors[3], tensors[5], tensors[11]))
    free = solver._free_mask(tensors[6], tensors[7], tensors[9], tensors[10])
    A, B = [matrix[free, :][:, free] for matrix in matrices]
    exact = linalg.eigvals(A.toarray(), B.toarray())
    shift = .3j*solver.k0
    for method in ('eigs', 'refined'):
        solver.solve(guess=shift, method=method, tol=1e-9, ncv=40)
        assert np.max(solver.eigenpair_residuals) < 1e-7
        for value in solver.eigenvalues:
            assert np.min(np.abs(exact-value)) < 1e-6*solver.k0
        if polarization == 'TM':
            # Reciprocity survives a finite PEC patch even though the two
            # transverse field components have different constraint masks.
            for value in exact[np.isfinite(exact)]:
                assert np.min(np.abs(exact+value)) < 1e-7*max(1., abs(value))


def test_arbitrary_singular_mass_matrix_matches_dense_qz():
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(7)
    A = csr_matrix(rng.normal(size=(12, 12))+1j*rng.normal(size=(12, 12)))
    B = csr_matrix(np.diag(np.r_[np.arange(1., 12.), 0.]))
    B = B @ csr_matrix(rng.normal(size=(12, 12)))
    exact = linalg.eigvals(A.toarray(), B.toarray())
    values, vectors = generalized_eigs(A, B, k=3, sigma=.2j, tol=1e-10, ncv=12)
    for value in values:
        assert np.min(np.abs(exact-value)) < 1e-8
    assert np.max(eigenpair_residuals(A, B, values, vectors)) < 1e-8


def test_3d_constrained_pencil_matches_dense_qz():
    from fdfd_periodic_modes.solver_3d import _PeriodicModeSolver3D
    solver = _PeriodicModeSolver3D(3, 2, 3, 10e-3, 8e-3, 6e-3, 20e9, 2)
    solver.add_pec((1, 2), (0, 1), (0, 1))
    tensors, *masks = solver._effective_materials_and_masks()
    A, B = solver._build_eigen_matrices(tensors, masks[2], masks[5])
    free = solver._free_mask(masks[0], masks[1], masks[3], masks[4])
    A, B = A[free, :][:, free], B[free, :][:, free]
    exact = linalg.eigvals(A.toarray(), B.toarray())
    for method in ('eigs', 'refined'):
        solver.solve(sigma_guess=.3j*solver.k0, method=method, tol=1e-9, ncv=40)
        assert np.max(solver.eigenpair_residuals) < 1e-7
        for value in solver.eigenvalues:
            assert np.min(np.abs(exact-value)) < 1e-6*solver.k0


def test_leaky_wave_cell_matches_refined_fem_reference():
    solver = PeriodicModeSolver2D(frequency=20e9, x_range=10e-3, z_range=8e-3,
        polarization='TM', boundary=materials.PEC)
    solver.add_rectangle(x_range=(0., 1.27e-3), z_range=(0., 8e-3),
        material=Material(name='substrate', epsilon=10.2))
    solver.add_rectangle(x_range=(1.27e-3, 1.32e-3), z_range=(1e-3, 2e-3), material=materials.PEC)
    solver.add_pml(thickness=2.5e-3, direction='x+')
    solver.mesh(resolution=(200, 80))
    result = solver.solve(num_modes=4, neff_guess=0., eigensolver_tolerance=1e-9, ncv=36)
    assert np.max(result.solve_info['residuals']) < 1e-7
    # FEM P1, maximum edge 100 um, no adaptive refinements. This inexpensive
    # grid permits finite geometry error; the benchmark records convergence.
    expected = np.array([.0143483929443405+.0926042962539239j,
                         -.0143483929443405-.0926042962539239j,
                         .5791833804852682-.5261466284273147j,
                         -.5791833804852682+.5261466284273147j])
    for value in expected:
        assert np.min(np.abs(result.neff-value)) < .035
    for value in result.neff:
        assert np.min(np.abs(result.neff+value)) < 1e-8


def test_default_pec_supports_uniform_tm_tem_and_dimensionless_pml():
    solver = PeriodicModeSolver2D(frequency=20e9, x_range=10e-3, z_range=8e-3,
        polarization='TM', background_material=Material(name='fill', epsilon=2.25))
    solver.mesh(resolution=(12, 9))
    result = solver.solve(num_modes=1, neff_guess=1.48)
    np.testing.assert_allclose(result.neff, [1.5], atol=1e-9)
    for frequency in (10e9, 30e9):
        solver = PeriodicModeSolver2D(frequency=frequency, x_range=10e-3, z_range=8e-3)
        solver.add_pml(thickness=2.5e-3, direction='x+', sigma_max=5.)
        solver.mesh(resolution=(20, 8))
        np.testing.assert_allclose(solver._backend.cell_eps_r_yy[-1], 1.-5j*.9**3)

"""Opt-in real-device CPU/cuDSS agreement across all FDFD solver families."""
import os
import numpy as np
import pytest
from scipy.constants import c
from fdfd_common import Material

pytestmark = pytest.mark.skipif(os.environ.get("FDFD_TEST_CUDA") != "1", reason="Set FDFD_TEST_CUDA=1 on a cuDSS host")

@pytest.mark.parametrize("dim", [1, 2])
def test_waveguide_cuda(dim):
    from fdfd_waveguide_modes import ModeSolver1D, ModeSolver2D
    solver = ModeSolver1D(frequency=c, x_range=1.) if dim == 1 else ModeSolver2D(frequency=c, x_range=1., y_range=.5)
    solver.mesh(resolution=32 if dim == 1 else (12, 8))
    cpu = solver.solve(backend="cpu", num_modes=1, neff_guess=.85, eigensolver_tolerance=1e-10)
    gpu = solver.solve(backend="cuda", num_modes=1, neff_guess=.85, eigensolver_tolerance=1e-10)
    np.testing.assert_allclose(gpu.neff, cpu.neff, rtol=1e-7, atol=1e-8)
    assert gpu.solve_info["backend"] == "cuda"
    assert np.max(gpu.solve_info["residuals"]) < 1e-8

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("method", ["eigs", "refined"])
def test_periodic_cuda(dim, method):
    from fdfd_periodic_modes import PeriodicModeSolver2D, PeriodicModeSolver3D
    kwargs = dict(frequency=c, x_range=1., z_range=.25, background_material=Material(epsilon=2.25))
    solver = PeriodicModeSolver2D(**kwargs) if dim == 2 else PeriodicModeSolver3D(y_range=.5, **kwargs)
    solver.mesh(resolution=(6, 5) if dim == 2 else (4, 3, 5))
    opts = dict(num_modes=1, neff_guess=1.4, eigensolver=method, eigensolver_tolerance=1e-9, ncv=40)
    cpu = solver.solve(backend="cpu", **opts)
    gpu = solver.solve(backend="cuda", **opts)
    np.testing.assert_allclose(gpu.neff, cpu.neff, rtol=1e-6, atol=1e-7)
    assert gpu.solve_info["backend"] == "cuda"
    assert np.max(gpu.solve_info["residuals"]) < 1e-7


def test_band_structure_cuda():
    from fdfd_band_structure import BandStructureSolver2D
    solver = BandStructureSolver2D(x_range=1., y_range=1.)
    solver.add_circle(center=(.5, .5), radius=.2, material=Material(epsilon=4.))
    solver.mesh(resolution=(8, 8))
    path = solver.make_bloch_path(points=((0., 0.), (1., .2)), num_points=2)
    cpu = solver.solve(backend="cpu", beta_path=path, num_modes=2, eigensolver_tolerance=1e-9)
    gpu = solver.solve(backend="cuda", beta_path=path, num_modes=2, eigensolver_tolerance=1e-9)
    for pol in ("TE", "TM"):
        np.testing.assert_allclose(gpu.eigenvalues[pol], cpu.eigenvalues[pol], rtol=1e-6, atol=1e-7)

@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_scattering_cuda(polarization):
    from fdfd_scattering import ScatteringSolver2D
    solver = ScatteringSolver2D(frequency=c, x_range=(-1., 1.), y_range=(-1., 1.), polarization=polarization)
    solver.add_circle(center=(0., 0.), radius=.2, material=Material(epsilon=2.))
    solver.add_pml(thickness=.3)
    solver.mesh(resolution=(20, 20))
    solver.add_source(angle=0.)
    solver.set_source_region(inset=.4)
    cpu = solver.solve(backend="cpu")
    gpu = solver.solve(backend="cuda")
    for name in cpu.fields:
        np.testing.assert_allclose(gpu.fields[name], cpu.fields[name], rtol=1e-7, atol=1e-8)
    assert gpu.solve_info["backend"] == "cuda"
    assert gpu.solve_info["relative_residual"] < 1e-9
    factor = solver._backend._systems[(polarization, "cuda")][2]
    solver.solve(backend="cuda")
    assert solver._backend._systems[(polarization, "cuda")][2] is factor
    repeated = solver.solve(backend="cuda", reuse_factorization=False)
    for name in gpu.fields:
        np.testing.assert_allclose(repeated.fields[name], gpu.fields[name], rtol=1e-7, atol=1e-8)
    solver._backend._clear_systems()
    assert factor.factor is None



def test_cuda_singular_mass_pencil():
    from scipy import linalg
    from scipy.sparse import csr_matrix
    from fdfd_common.cuda import eigenpairs
    from fdfd_periodic_modes._eigensolve import eigenpair_residuals
    rng = np.random.default_rng(7)
    A = csr_matrix(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    B = csr_matrix(np.diag(np.r_[np.arange(1., 12.), 0.])) @ csr_matrix(rng.normal(size=(12, 12)))
    exact = linalg.eigvals(A.toarray(), B.toarray())
    values, vectors = eigenpairs(A, B, k=3, sigma=.2j, tol=1e-10, ncv=12)
    for value in values:
        assert np.min(abs(exact - value)) < 1e-8
    assert np.max(eigenpair_residuals(A, B, values, vectors)) < 1e-8


@pytest.mark.parametrize("factory_name", ["slab", "rectangle"])
def test_cuda_mode_tracking(factory_name):
    from tests.fdfd.mode_tracking.test_tracking import slab, rectangle, config, FC
    from fdfd_mode_tracking import track_modes, PortSpec
    factory = slab if factory_name == "slab" else rectangle
    kwargs = dict(port=PortSpec(boundary="enclosed"), config=config(), progress=False)
    cpu = track_modes(factory, [.8 * FC], backend="cpu", **kwargs)
    gpu = track_modes(factory, [.8 * FC], backend="cuda", **kwargs)
    np.testing.assert_allclose([mode.beta for mode in gpu.export()],
        [mode.beta for mode in cpu.export()], rtol=1e-6, atol=1e-7)
    for sample in gpu.samples:
        assert sample.candidates.result.solve_info["backend"] == "cuda"

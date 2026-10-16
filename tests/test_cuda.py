"""Opt-in CPU/GPU agreement for every solver (needs CuPy and cuDSS).

Run with ``FDFD_TEST_CUDA=1`` and, if cuDSS is not on the loader path,
``CUDSS_LIBRARY`` pointing to ``cudss64_0.dll`` / ``libcudss.so``.
"""

import os

import numpy as np
import pytest

import fdfd
from fdfd import Material, materials, shapes
from fdfd.core.constants import C0

pytestmark = pytest.mark.skipif(
    os.environ.get("FDFD_TEST_CUDA") != "1", reason="set FDFD_TEST_CUDA=1 on a cuDSS host"
)
DIELECTRIC = Material(name="dielectric", epsilon=2.25)


@pytest.mark.parametrize("method", ["arpack", "refined"])
@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("device", ["cuda", "gpu", "hybrid", "auto"])
def test_waveguide(dimension, method, device):
    if dimension == 1:
        solver = fdfd.ModeSolver1D(frequency=C0, x_range=(0.0, 1.0))
        solver.mesh(resolution=32)
    else:
        solver = fdfd.ModeSolver2D(frequency=C0, x_range=(0.0, 1.0), y_range=(0.0, 0.5))
        solver.mesh(resolution=(12, 8))
    options = dict(num_modes=1, guess=0.85, tol=1e-10, method=method)
    cpu = solver.solve(device="cpu", **options)
    gpu = solver.solve(device=device, **options)
    np.testing.assert_allclose(gpu.neff, cpu.neff, rtol=1e-7, atol=1e-8)
    assert gpu.info.device == ("cuda" if device in ("gpu", "cuda") else "hybrid") and gpu.info.converged


@pytest.mark.parametrize("method", ["arpack", "refined"])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("device", ["cuda", "gpu", "hybrid", "auto"])
def test_periodic(dimension, method, device):
    settings = dict(frequency=C0, x_range=(0.0, 1.0), z_range=(0.0, 0.25), background_material=DIELECTRIC)
    if dimension == 2:
        solver = fdfd.PeriodicModeSolver2D(**settings)
        solver.mesh(resolution=(6, 5))
        options = dict(polarization="TM")
    else:
        solver = fdfd.PeriodicModeSolver3D(y_range=(0.0, 0.5), **settings)
        solver.mesh(resolution=(4, 3, 5))
        options = {}
    options.update(num_modes=1, guess=1.4, method=method, tol=1e-9, ncv=40)
    cpu = solver.solve(device="cpu", **options)
    gpu = solver.solve(device=device, **options)
    np.testing.assert_allclose(gpu.neff, cpu.neff, rtol=1e-6, atol=1e-7)
    assert gpu.info.device == ("cuda" if device in ("gpu", "cuda") else "hybrid") and gpu.info.converged


@pytest.mark.parametrize("device", ["cuda", "gpu", "hybrid", "auto"])
def test_band_structure(device):
    solver = fdfd.BandStructureSolver2D(x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    solver.add_circle(center=(0.5, 0.5), radius=0.2, material=Material(name="rod", epsilon=4.0))
    solver.mesh(resolution=(8, 8))
    path = fdfd.bloch_path(((0.0, 0.0), (1.0, 0.2)), num_points=2)
    cpu = solver.solve(beta_path=path, num_modes=2, tol=1e-9, progress=False)
    gpu = solver.solve(beta_path=path, num_modes=2, tol=1e-9, device=device, progress=False)
    for pol in ("TE", "TM"):
        # atol covers the static Gamma mode, a numerical zero.
        np.testing.assert_allclose(gpu.frequencies[pol], cpu.frequencies[pol], rtol=1e-6, atol=1e3)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
@pytest.mark.parametrize("device", ["cuda", "gpu", "hybrid", "auto"])
def test_scattering_reuses_its_factorization(polarization, device):
    solver = fdfd.ScatteringSolver2D(frequency=C0, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
    solver.add_circle(center=(0.0, 0.0), radius=0.2, material=Material(name="cylinder", epsilon=2.0))
    solver.add_pml(thickness=0.3)
    solver.mesh(resolution=(20, 20))
    solver.set_source(angle=0.0, inset=0.4)
    cpu = solver.solve(polarization=polarization)
    gpu = solver.solve(polarization=polarization, device=device)
    for name in cpu.fields:
        np.testing.assert_allclose(gpu.fields[name], cpu.fields[name], rtol=1e-7, atol=1e-8)
    assert (
        gpu.info.device == ("cuda" if device in ("gpu", "cuda") else "hybrid")
        and gpu.info.residuals[0] < 1e-9
    )
    factor = solver._kernel._systems[(polarization, "cuda" if device in ("gpu", "cuda") else "hybrid")][3]
    solver.set_source(angle=30.0, inset=0.4)
    solver.solve(polarization=polarization, device=device)
    assert (
        solver._kernel._systems[(polarization, "cuda" if device in ("gpu", "cuda") else "hybrid")][3]
        is factor
    )
    solver._kernel.close()
    assert factor.factor is None


def test_singular_mass_pencil():
    from scipy import linalg
    from scipy.sparse import csr_matrix

    from fdfd.core import eigs

    rng = np.random.default_rng(7)
    A = csr_matrix(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    B = csr_matrix(np.diag(np.r_[np.arange(1.0, 12.0), 0.0])) @ csr_matrix(rng.normal(size=(12, 12)))
    exact = linalg.eigvals(A.toarray(), B.toarray())
    pairs = eigs.solve(A, B, k=3, sigma=0.2j, tol=1e-10, ncv=12, device="cuda")
    for value in pairs.values:
        assert np.min(abs(exact - value)) < 1e-8
    assert np.max(pairs.residuals) < 1e-8


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("device", ["cuda", "gpu", "hybrid", "auto"])
def test_mode_tracking(dimension, device):
    width, height = 22.86e-3, 10.16e-3
    fc = C0 / (2 * width)
    if dimension == 1:
        dx = width / 80
        tracker = fdfd.ModeTracker1D(
            frequencies=[0.8 * fc], x_range=(-dx, width + dx), port=fdfd.PortSpec(boundary="enclosed")
        )
        tracker.add_layer(x_range=(-dx, 0.0), material=materials.PEC, name="left")
        tracker.add_layer(x_range=(width, width + dx), material=materials.PEC, name="right")
        tracker.mesh(resolution=82, subpixels=1)
    else:
        dx, dy = width / 32, height / 16
        bounds = ((-dx, width + dx), (-dy, height + dy))
        tracker = fdfd.ModeTracker2D(
            frequencies=[0.8 * fc],
            x_range=bounds[0],
            y_range=bounds[1],
            port=fdfd.PortSpec(boundary="enclosed"),
        )
        tracker.add_geometry(
            shape=shapes.Difference(
                shape=shapes.Rectangle(bounds=bounds),
                tool=shapes.Rectangle(bounds=((0.0, width), (0.0, height))),
            ),
            material=materials.PEC,
        )
        tracker.mesh(resolution=(34, 18), subpixels=1)
    config = fdfd.TrackingConfig(verification_beta_tolerance=0.006, verification_overlap=0.995)
    cpu = tracker.solve(num_modes=2, config=config, progress=False)
    gpu = tracker.solve(num_modes=2, config=config, device=device, progress=False)
    np.testing.assert_allclose(
        [s.candidates.result.neff for s in gpu.samples],
        [s.candidates.result.neff for s in cpu.samples],
        rtol=1e-6,
        atol=1e-7,
    )
    assert gpu.info.device == ("cuda" if device in ("gpu", "cuda") else "hybrid")
    assert all(
        s.candidates.result.info.device == ("cuda" if device in ("gpu", "cuda") else "hybrid")
        for s in gpu.samples
    )


def test_auto_device_selects_cuda_when_available():
    from fdfd.core import cuda

    assert cuda.available()
    solver = fdfd.ModeSolver1D(frequency=C0, x_range=(0.0, 1.0))
    solver.mesh(resolution=32)
    assert solver.solve(num_modes=1, guess=0.85, device="auto").info.device == "hybrid"

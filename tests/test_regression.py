"""Numerical regression against results recorded with fdfd 1.1.2.

``data/regression_reference.json`` was produced by the 1.x API before the 2.0
restructuring. Each case here rebuilds the same physical problem with the 2.0
API and must reproduce it. Intentional physics changes in 2.0 (the PML
stretch definition and default subpixel counts) are avoided by fixing those
settings explicitly, so any difference is a regression.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import fdfd
from fdfd import Material, materials, shapes

REFERENCE = json.loads((Path(__file__).parent / "data" / "regression_reference.json").read_text())
RTOL = 1e-7  # 2.0 uses CODATA epsilon_0/mu_0 throughout (1.x mixed them).


def complex_array(pairs):
    return np.array([complex(re, im) for re, im in np.asarray(pairs).reshape(-1, 2)])


def canonical(values):
    """Sort a spectrum, folding +/-gamma Bloch pairs onto one representative."""
    values = np.asarray(values, dtype=complex)
    flip = (values.real < -1e-12) | ((abs(values.real) <= 1e-12) & (values.imag > 0))
    values = np.where(flip, -values, values)
    return values[np.lexsort((np.round(values.imag, 6), np.round(values.real, 6)))]


def assert_neff(actual, expected, *, rtol=RTOL, atol=1e-9):
    np.testing.assert_allclose(np.asarray(actual), complex_array(expected), rtol=rtol, atol=atol)


def test_waveguide_1d_grounded_slab():
    s = fdfd.ModeSolver1D(frequency=30e9, x_range=(0.0, 10e-3))
    s.add_layer(x_range=(3e-3, 4.27e-3), material=Material(name="slab", epsilon=10.2))
    s.add_layer(x_range=(2.9e-3, 3e-3), material=materials.PEC)
    s.mesh(resolution=500, subpixels=8)
    result = s.solve(num_modes=4)
    reference = REFERENCE["wg1d_slab"]
    assert_neff(result.neff, reference["neff"])
    assert list(result.metadata["polarizations"]) == reference["pol"]


@pytest.mark.parametrize("polarization", ["TE", "TM", "both"])
def test_waveguide_1d_anisotropic_pmc(polarization):
    s = fdfd.ModeSolver1D(frequency=100e9, x_range=(0.0, 8e-3))
    s.add_layer(x_range=(3e-3, 4.5e-3), material=Material(name="fill", epsilon=(4.0, 5.0, 6.0)))
    s.add_layer(x_range=(2.9e-3, 3e-3), material=materials.PMC)
    s.add_layer(x_range=(4.5e-3, 4.6e-3), material=materials.PMC)
    s.mesh(resolution=400, subpixels=8)
    assert_neff(
        s.solve(num_modes=3, polarization=polarization).neff, REFERENCE["wg1d_aniso_pmc"][polarization]
    )


def test_waveguide_1d_copper_walls():
    s = fdfd.ModeSolver1D(frequency=20e9, x_range=(0.0, 6e-3))
    s.add_layer(x_range=(0.0, 0.5e-3), material=materials.copper)
    s.add_layer(x_range=(0.5e-3, 2.0e-3), material=Material(name="lossy", epsilon=3.0 - 0.01j))
    s.add_layer(x_range=(5.5e-3, 6e-3), material=materials.copper)
    s.mesh(resolution=240, subpixels=8)
    result = s.solve(num_modes=4)
    assert_neff(result.neff, REFERENCE["wg1d_copper"]["neff"], rtol=1e-6)
    assert list(result.metadata["polarizations"]) == REFERENCE["wg1d_copper"]["pol"]


def test_waveguide_2d_copper_rectangular_guide():
    s = fdfd.ModeSolver2D(frequency=100e9, x_range=(0.0, 12e-3), y_range=(0.0, 10e-3))
    wall = shapes.Difference(
        shape=shapes.Rectangle(bounds=((1.9e-3, 10.1e-3), (1.9e-3, 8.1e-3))),
        tool=shapes.Rectangle(bounds=((2e-3, 10e-3), (2e-3, 8e-3))),
    )
    s.add_geometry(shape=wall, material=materials.copper)
    s.mesh(resolution=(60, 50), subpixels=8)
    assert_neff(s.solve(num_modes=4).neff, REFERENCE["wg2d_rect_copper"]["neff"], rtol=1e-6)


def test_waveguide_2d_lossy_microstrip():
    s = fdfd.ModeSolver2D(frequency=50e9, x_range=(0.0, 12e-3), y_range=(0.0, 10e-3))
    s.add_rectangle(
        x_range=(2e-3, 10e-3), y_range=(4e-3, 5e-3), material=Material(name="sub", epsilon=4.0 - 1j)
    )
    s.add_rectangle(x_range=(5e-3, 7e-3), y_range=(5e-3, 5.1e-3), material=materials.copper, name="strip")
    s.add_rectangle(
        x_range=(500e-6, 11.5e-3), y_range=(3.9e-3, 4e-3), material=materials.copper, name="ground"
    )
    s.mesh(resolution=(120, 100), subpixels=8)
    assert_neff(s.solve(num_modes=2).neff, REFERENCE["wg2d_microstrip"]["neff"], rtol=1e-6)


def test_waveguide_2d_circular_fibre():
    s = fdfd.ModeSolver2D(frequency=100e9, x_range=(0.0, 10e-3), y_range=(0.0, 10e-3))
    s.add_circle(center=(5e-3, 5e-3), radius=3e-3, material=Material(name="core", epsilon=6.0))
    s.mesh(resolution=(40, 40), subpixels=8)
    result = s.solve(num_modes=4)
    assert_neff(result.neff, REFERENCE["wg2d_circle"]["neff"])
    assert result.info.converged


def test_waveguide_2d_pml_does_not_move_guided_modes():
    # 2.0 replaced the S/m, whole-cell PML with a dimensionless continuous
    # stretch; these modes are guided, so only tiny changes are allowed.
    s = fdfd.ModeSolver2D(frequency=50e9, x_range=(0.0, 24e-3), y_range=(0.0, 16e-3))
    s.add_rectangle(
        x_range=(0.0, 24e-3), y_range=(6e-3, 8e-3), material=Material(name="slab", epsilon=(3, 4, 5))
    )
    s.add_rectangle(
        x_range=(10e-3, 14e-3), y_range=(8e-3, 10e-3), material=Material(name="ridge", epsilon=6.0)
    )
    s.add_pml(thickness=3e-3, sides="x", strength=0.36)
    s.mesh(resolution=(60, 40), subpixels=8)
    assert_neff(s.solve(num_modes=4).neff, REFERENCE["wg2d_ridge_pml"]["neff"], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
@pytest.mark.parametrize("method", ["arpack", "refined"])
def test_periodic_2d_leaky_cell(polarization, method):
    s = fdfd.PeriodicModeSolver2D(
        frequency=20e9, x_range=(0.0, 10e-3), z_range=(0.0, 8e-3), boundary=materials.PEC
    )
    s.add_rectangle(x_range=(0.0, 1.27e-3), z_range=(0.0, 8e-3), material=Material(name="sub", epsilon=10.2))
    s.add_rectangle(x_range=(1.27e-3, 1.37e-3), z_range=(1e-3, 2e-3), material=materials.PEC)
    s.add_pml(thickness=2.5e-3, sides="x+")
    s.mesh(resolution=(100, 80), subpixels=8)
    result = s.solve(num_modes=4, polarization=polarization, guess=0.0, tol=1e-9, ncv=36, method=method)
    old = "eigs" if method == "arpack" else "refined"
    expected = complex_array(REFERENCE["per2d"][f"{polarization}_{old}"])
    np.testing.assert_allclose(canonical(result.neff), canonical(expected), rtol=RTOL, atol=1e-9)
    assert set(result.metadata["polarizations"]) == {polarization}


def test_periodic_2d_pmc_boundary():
    s = fdfd.PeriodicModeSolver2D(
        frequency=20e9, x_range=(0.0, 10e-3), z_range=(0.0, 8e-3), boundary=materials.PMC
    )
    s.add_rectangle(x_range=(0.0, 2e-3), z_range=(0.0, 4e-3), material=Material(name="d", epsilon=6.0))
    s.mesh(resolution=(50, 40), subpixels=8)
    result = s.solve(num_modes=3, polarization="TE", guess=1.5)
    np.testing.assert_allclose(
        canonical(result.neff),
        canonical(complex_array(REFERENCE["per2d"]["pmc_TE_default"])),
        rtol=RTOL,
        atol=1e-9,
    )


@pytest.mark.parametrize("method", ["arpack", "refined"])
def test_periodic_3d_image_guide(method):
    s = fdfd.PeriodicModeSolver3D(
        frequency=30e9, x_range=(0.0, 12e-3), y_range=(0.0, 8e-3), z_range=(0.0, 6e-3)
    )
    s.add_box(
        x_range=(4e-3, 8e-3),
        y_range=(1e-3, 4e-3),
        z_range=(0.0, 6e-3),
        material=Material(name="g", epsilon=4),
    )
    s.add_box(x_range=(0.0, 12e-3), y_range=(0.0, 1e-3), z_range=(0.0, 6e-3), material=materials.PEC)
    s.add_box(x_range=(4e-3, 8e-3), y_range=(4e-3, 5e-3), z_range=(2e-3, 4e-3), material=materials.PEC)
    s.add_pml(thickness=1.5e-3, sides="x+", strength=1.0)
    s.mesh(resolution=(16, 12, 8), subpixels=8)
    result = s.solve(num_modes=3, guess=0.0, method=method)
    expected = complex_array(REFERENCE["per3d"]["eigs" if method == "arpack" else "refined"])
    np.testing.assert_allclose(canonical(result.neff), canonical(expected), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("cell", [(10e-3, 10e-3), (10e-3, 7e-3)])
def test_band_structure(cell):
    a, b = cell
    s = fdfd.BandStructureSolver2D(x_range=(-a / 2, a / 2), y_range=(-b / 2, b / 2))
    s.add_circle(center=(0.0, 0.0), radius=0.2 * min(a, b), material=Material(name="rod", epsilon=8.9))
    s.mesh(resolution=(16, 16), subpixels=1)
    path = fdfd.bloch_path(((0.0, 0.0), (np.pi / a, 0.0), (np.pi / a, np.pi / b), (0.0, 0.0)), num_points=10)
    reference = REFERENCE["bands"][f"{a}_{b}"]
    np.testing.assert_allclose(path, reference["path"], rtol=1e-14)
    result = s.solve(beta_path=path, num_modes=4, progress=False)
    gamma = [0, path.shape[1] - 1]
    interior = slice(1, -1)
    for polarization in ("TE", "TM"):
        actual = result.frequencies[polarization]
        expected = complex_array(reference[polarization]).reshape(actual.shape)
        np.testing.assert_allclose(actual[:, interior], expected[:, interior], rtol=RTOL, atol=1.0)
        # 1.x shifted onto the static Gamma mode (sigma = 0), which cost those
        # bands up to ~1e-4 in accuracy; 2.0 shifts below the spectrum. The
        # static mode is a numerical zero: sqrt(1e-14 * lambda) is ~1 kHz.
        np.testing.assert_allclose(actual[:, gamma].real, expected[:, gamma].real, rtol=2e-4, atol=1e4)
    assert np.max(result.info.residuals[:, 1:]) < 1e-10


SCATTERING_CASES = [
    ("TE_cyl_plane_wave", "TE", Material(name="cyl", epsilon=4.0), "plane_wave"),
    ("TM_PEC_plane_wave", "TM", materials.PEC, "plane_wave"),
    ("TE_cyl_point", "TE", Material(name="cyl", epsilon=4.0), "point"),
    ("TM_cyl_plane_wave", "TM", Material(name="cyl", epsilon=2.0 - 0.1j), "plane_wave"),
]


@pytest.mark.parametrize(
    ("key", "polarization", "material", "kind"), SCATTERING_CASES, ids=[c[0] for c in SCATTERING_CASES]
)
def test_scattering(key, polarization, material, kind):
    s = fdfd.ScatteringSolver2D(frequency=3e9, x_range=(-150e-3, 150e-3), y_range=(-150e-3, 150e-3))
    s.add_circle(center=(0.0, 0.0), radius=25e-3, material=material)
    if kind == "point":
        s.set_source(kind="point", location=(-60e-3, 10e-3), amplitude=2.0, inset=75e-3)
    else:
        s.set_source(kind="plane_wave", angle=30.0, amplitude=1.0, inset=75e-3)
    s.mesh(resolution=(60, 60), subpixels=1)
    result = s.solve(polarization=polarization)
    for name, expected in REFERENCE["scattering"][key].items():
        values = result.fields[name]
        np.testing.assert_allclose(np.sum(abs(values) ** 2), expected["sum_abs2"], rtol=1e-9)
        np.testing.assert_allclose(
            values[17, 23, 0], complex_array(expected["probe"])[0], rtol=1e-9, atol=1e-14
        )


def _compare_sweep(sweep, reference, *, rtol=1e-7):
    np.testing.assert_allclose([s.frequency for s in sweep.samples], reference["frequencies"], rtol=1e-12)
    for sample, neff, tracks in zip(sweep.samples, reference["neff"], reference["tracks"], strict=True):
        assert_neff(sample.candidates.result.neff, neff, rtol=rtol, atol=1e-8)
        assert [int(i) for i in sample.candidate_indices] == tracks


def test_tracking_1d_anisotropic_crossing():
    width, cells = 22.86e-3, 60
    dx = width / cells
    fc = fdfd.core.constants.C0 / (2 * width)
    q = (4.0 - 2.25) / (1.0 - 2.25 / 9.0)
    frequencies = fc / np.sqrt(q) * np.array([0.9, 0.99, 1.01, 1.1])
    t = fdfd.ModeTracker1D(
        frequencies=frequencies,
        x_range=(-dx, width + dx),
        background_material=Material(name="aniso", epsilon=(2.25, 4.0, 9.0)),
        port=fdfd.PortSpec(boundary="enclosed"),
    )
    t.add_layer(x_range=(-dx, 0.0), material=materials.PEC)
    t.add_layer(x_range=(width, width + dx), material=materials.PEC)
    t.mesh(resolution=cells + 2, subpixels=1)
    sweep = t.solve(
        num_modes=4,
        tol=1e-10,
        progress=False,
        config=fdfd.TrackingConfig(
            max_depth=3, overlap_min=0.9, verification_overlap=0.997, verification_beta_tolerance=0.003
        ),
    )
    # 1.x used 8.854187817e-12 F/m for the analytical crossing; frequencies
    # therefore differ by ~1e-9 relative from the recorded ones.
    reference = dict(REFERENCE["track1d"])
    reference["frequencies"] = [s.frequency for s in sweep.samples]
    _compare_sweep(sweep, reference, rtol=1e-6)


def test_tracking_2d_rectangular_guide():
    a, b = 22.86e-3, 10.16e-3
    t = fdfd.ModeTracker2D(
        frequencies=np.array([8e9, 10e9, 12e9]),
        x_range=(0.0, a),
        y_range=(0.0, b),
        port=fdfd.PortSpec(boundary="enclosed"),
    )
    frame = shapes.Difference(
        shape=shapes.Rectangle(bounds=((0.0, a), (0.0, b))),
        tool=shapes.Rectangle(bounds=((a / 24, a * 23 / 24), (b / 12, b * 11 / 12))),
    )
    t.add_geometry(shape=frame, material=materials.PEC)
    t.mesh(resolution=(24, 12), subpixels=1)
    sweep = t.solve(num_modes=3, tol=1e-10, progress=False, config=fdfd.TrackingConfig(max_depth=2))
    _compare_sweep(sweep, REFERENCE["track2d"])

"""Scattering: TF/SF reconstruction, staggered fields, analytic cylinders, phasor convention."""

import numpy as np
import pytest
from scipy.special import h2vp, hankel2, jv, jvp

import fdfd
from fdfd import Material, materials
from fdfd.core.constants import ETA_0
from fdfd.scattering._kernel import PRIMARY, ScatteringKernel


def flat(values):
    return np.asarray(values).ravel(order="F")


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_transverse_reconstruction_removes_the_incident_interface_jump(polarization):
    kernel = ScatteringKernel(3e9, 120e-3, 100e-3, 12, 10)
    source = kernel.incident(polarization, kind="plane_wave", angle=30.0, location=None, amplitude=1.0)
    masks = kernel.total_field_masks(2 * kernel.dx)
    inside = 1.0 - masks[PRIMARY[polarization]].diagonal()
    fields = kernel._fields(polarization, inside * source, source, masks)
    dex, dey, dhx, dhy = kernel._derivatives(polarization)
    if polarization == "TE":
        expected = {"Hx": 1j * (dey @ source) / ETA_0, "Hy": -1j * (dex @ source) / ETA_0}
    else:
        expected = {"Ex": -1j * ETA_0 * (dhy @ source), "Ey": 1j * ETA_0 * (dhx @ source)}
    for name, values in expected.items():
        component_inside = 1 - masks[name].diagonal()
        np.testing.assert_allclose(flat(fields[name]), component_inside * values, atol=1e-12)


@pytest.mark.parametrize(
    ("polarization", "locations"),
    [
        ("TE", {"Ez": ("node", "node"), "Hx": ("node", "cell"), "Hy": ("cell", "node")}),
        ("TM", {"Hz": ("cell", "cell"), "Ex": ("cell", "node"), "Ey": ("node", "cell")}),
    ],
)
def test_three_staggered_field_components_round_trip(polarization, locations, tmp_path):
    solver = fdfd.ScatteringSolver2D(frequency=3e9, x_range=(-60e-3, 60e-3), y_range=(-50e-3, 50e-3))
    solver.mesh(resolution=(12, 10))
    solver.set_source(angle=30.0, inset=20e-3)
    result = solver.solve(polarization=polarization)
    assert set(result.fields) == set(locations) and result.polarization == polarization
    restored = fdfd.load(result.save(tmp_path / "scattering.h5"))
    for name, values in result.fields.items():
        assert values.shape == (
            *(n + (loc == "node") for n, loc in zip((12, 10), locations[name], strict=True)),
            1,
        )
        assert np.isfinite(values).all() and np.linalg.norm(values) > 0
        for coordinates, bounds, n, location in zip(
            result.field_coordinates[name], result.grid.bounds, (12, 10), locations[name], strict=True
        ):
            expected = (
                np.linspace(*bounds, n + 1)
                if location == "node"
                else bounds[0] + (np.arange(n) + 0.5) * (bounds[1] - bounds[0]) / n
            )
            np.testing.assert_allclose(coordinates, expected, atol=1e-15)
        np.testing.assert_array_equal(restored.fields[name], values)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
@pytest.mark.parametrize("scatterer", ["dielectric", "PEC"])
def test_cylinder_converges_to_the_cylindrical_wave_solution(polarization, scatterer):
    errors = []
    for cells in (120, 240):
        solver = fdfd.ScatteringSolver2D(frequency=3e9, x_range=(-150e-3, 150e-3), y_range=(-150e-3, 150e-3))
        solver.add_circle(
            center=(0.0, 0.0),
            radius=25e-3,
            material=materials.PEC if scatterer == "PEC" else Material(name="cylinder", epsilon=4.0),
        )
        solver.add_pml(thickness=50e-3)
        solver.mesh(resolution=(cells, cells))
        solver.set_source(inset=75e-3)
        result = solver.solve(polarization=polarization)
        component = "Ez" if polarization == "TE" else "Hz"
        xx, yy = np.meshgrid(*result.field_coordinates[component], indexing="ij")
        radius, theta = np.hypot(xx, yy), np.arctan2(yy, xx)
        selected = (radius > 35e-3) & (radius < 65e-3)
        r, angle = radius[selected], theta[selected]
        k, a = solver._kernel.k0, 25e-3
        # Ez continues with (1/mu) dEz/dn, Hz with (1/epsilon) dHz/dn.
        q = 2.0 if polarization == "TE" else 0.5
        exact = np.exp(-1j * k * r * np.cos(angle))
        for n in range(-12, 13):
            if scatterer == "PEC":
                coefficient = (
                    -jv(n, k * a) / hankel2(n, k * a)
                    if polarization == "TE"
                    else -jvp(n, k * a) / h2vp(n, k * a)
                )
            else:
                coefficient = (q * jv(n, k * a) * jvp(n, 2 * k * a) - jvp(n, k * a) * jv(n, 2 * k * a)) / (
                    h2vp(n, k * a) * jv(n, 2 * k * a) - q * hankel2(n, k * a) * jvp(n, 2 * k * a)
                )
            exact = exact + (-1j) ** n * coefficient * hankel2(n, k * r) * np.exp(1j * n * angle)
        calculated = result.fields[component][..., 0][selected]
        errors.append(np.linalg.norm(calculated - exact) / np.linalg.norm(exact))
        if scatterer == "PEC":
            assert np.any(result.metadata["material_background"]["conductor"])
            for name, field in result.fields.items():
                x, y = np.meshgrid(*result.field_coordinates[name], indexing="ij")
                np.testing.assert_array_equal(field[..., 0][x * x + y * y < (20e-3) ** 2], 0.0)
    assert errors[1] < 0.65 * errors[0]
    assert errors[1] < (0.03 if scatterer == "PEC" else 0.025)


def test_pec_object_must_lie_inside_the_total_field_region():
    solver = fdfd.ScatteringSolver2D(frequency=3e9, x_range=(-150e-3, 150e-3), y_range=(-150e-3, 150e-3))
    solver.add_circle(center=(0.0, 0.0), radius=25e-3, material=materials.PEC)
    solver.mesh(resolution=(60, 60))
    solver.set_source(inset=130e-3)
    with pytest.raises(fdfd.ConfigurationError, match="enclose every PEC"):
        solver.solve()


def test_solve_requires_a_source_and_validates_it():
    solver = fdfd.ScatteringSolver2D(frequency=3e9, x_range=(-0.1, 0.1), y_range=(-0.1, 0.1))
    with pytest.raises(fdfd.ConfigurationError, match="set_source"):
        solver.solve()
    with pytest.raises(fdfd.ConfigurationError, match="location"):
        solver.set_source(kind="point", inset=0.02)
    with pytest.raises(fdfd.ConfigurationError, match="point sources"):
        solver.set_source(location=(0.0, 0.0), inset=0.02)


def test_plane_wave_along_x_has_negative_spatial_phase():
    kernel = ScatteringKernel(2e9, 0.12, 0.08, 6, 4)
    field = kernel.incident("TE", kind="plane_wave", angle=0.0, location=None, amplitude=1.0)
    field = field.reshape(kernel.shape("Ez"), order="F")
    np.testing.assert_allclose(field[1:, :] / field[:-1, :], np.exp(-1j * kernel.k0 * kernel.dx))


def test_point_sources_use_outgoing_hankel_functions_of_the_second_kind():
    kernel = ScatteringKernel(2e9, 0.12, 0.08, 6, 4)
    location = (0.013, -0.007)
    for polarization, scale in (("TE", 1.0), ("TM", -1j / 4)):
        x, y = np.meshgrid(*kernel.coordinates[PRIMARY[polarization]], indexing="ij")
        radius = np.hypot(x - location[0], y - location[1])
        field = kernel.incident(polarization, kind="point", angle=0.0, location=location, amplitude=1.0)
        np.testing.assert_allclose(field, flat(scale * hankel2(0, kernel.k0 * radius)))


def test_pml_is_a_negative_imaginary_stretch():
    solver = fdfd.ScatteringSolver2D(frequency=2e9, x_range=(0.0, 0.12), y_range=(0.0, 0.08))
    solver.add_pml(thickness=0.02, sides="x", strength=3.0, order=2)
    solver.mesh(resolution=(6, 4))
    stretch = 1 - 3j * 0.5**2
    for values in (solver._kernel.cell_eps_r_zz, solver._kernel.cell_mu_r_zz):
        np.testing.assert_allclose(values[(0, -1), :], stretch)
        np.testing.assert_allclose(values[1:-1, :], 1.0)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_discrete_vacuum_wave_has_no_scattered_field(polarization):
    kernel = ScatteringKernel(3e9, 200e-3, 150e-3, 40, 30)
    # Use the exact Yee dispersion relation to isolate the TF/SF interface.
    kx = 2 * np.arcsin(kernel.k0 * kernel.dx / 2) / kernel.dx
    x, _ = np.meshgrid(*kernel.coordinates[PRIMARY[polarization]], indexing="ij")
    source = flat(np.exp(-1j * kx * x))
    masks = kernel.total_field_masks(30e-3)
    fields, residual = kernel.solve(polarization, source, masks)
    assert residual < 1e-10
    factors = (
        {"Ez": 1.0, "Hx": 0.0, "Hy": -1 / ETA_0}
        if polarization == "TE"
        else {"Hz": 1.0, "Ex": 0.0, "Ey": ETA_0}
    )
    for name, field in fields.items():
        x, y = kernel.coordinates[name]
        wave = np.exp(-1j * kx * x)[:, None] * np.ones((1, len(y)))
        inside = (1 - masks[name].diagonal()).reshape(field.shape, order="F")
        np.testing.assert_allclose(field, factors[name] * wave * inside, atol=1e-9, rtol=1e-10)

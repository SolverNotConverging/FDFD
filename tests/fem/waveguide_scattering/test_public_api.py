from cem_common import Material, SurfaceImpedance, materials, shapes
from fem_waveguide_scattering.materials import Material as _internal_Material
from fem_waveguide_scattering.modes import CrossSection as _internal_CrossSection
from fem_waveguide_scattering.modes import ModeSolver as _internal_ModeSolver
import numpy as np
import pytest

import fem_waveguide_scattering as wf


def test_documented_top_level_api_is_importable() -> None:
    expected = {
        "FrequencySweepResult",
        "IncidentMode",
        "Mode",
        "ModeSet",
        "WaveguideScatteringSolver2D",
        "ScatteringResult",
        "load_result",
    }
    assert expected <= set(wf.__all__)
    assert all(hasattr(wf, name) for name in expected)


def test_scattering_angle_sets_ky_and_is_preserved_across_frequency() -> None:
    common = {
        "x_range": (0.0, 1.0),
        "z_range": (-2.0, 2.0),
        "boundary": materials.PEC,
    }
    normal = wf.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.0, angle=0.0, **common)
    oblique = wf.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.0, angle=30.0, **common)
    catalog = oblique.solve_modes(
        max_refinements=0,
        num_modes=1,
        neff_guess=np.sqrt(0.75),
        num_elements=32,
    )
    seed_total_neff = catalog[0].neff.real
    incident = oblique.set_incident_mode(catalog[0])
    eta = oblique.ky / oblique.k0
    resolved_total_neff = np.hypot(incident.mode.neff.real, eta)
    doubled = oblique._clone_at_frequency(2.0 * oblique.frequency)

    assert normal.angle == pytest.approx(0.0)
    assert normal.ky == pytest.approx(0.0)
    assert oblique.angle == pytest.approx(30.0)
    assert oblique.ky == pytest.approx(oblique.k0 * seed_total_neff * np.sin(np.deg2rad(oblique.angle)))
    assert resolved_total_neff == pytest.approx(seed_total_neff, rel=5e-4)
    assert np.degrees(np.arctan2(eta, incident.mode.neff.real)) == pytest.approx(
        oblique.angle, abs=5e-4
    )
    assert doubled.angle == pytest.approx(oblique.angle)
    assert doubled.ky == pytest.approx(0.0)


@pytest.mark.parametrize("angle", [np.nan, np.inf, 1j, 90.0, -90.0, True])
def test_scattering_rejects_invalid_propagation_angle(angle: object) -> None:
    with pytest.raises(wf.ConfigurationError, match="angle"):
        wf.WaveguideScatteringSolver2D(frequency=1000000000.0, angle=angle, x_range=(0.0, 1.0), z_range=(-1.0, 1.0))


def test_scattering_rejects_angle_and_ky_together() -> None:
    with pytest.raises(wf.ConfigurationError, match="angle or ky"):
        wf.WaveguideScatteringSolver2D(frequency=1000000000.0, angle=0.0, ky=0.0, x_range=(0.0, 1.0), z_range=(-1.0, 1.0))


def test_scattering_rejects_unimplemented_pmc_instead_of_imposing_pec() -> None:
    with pytest.raises(wf.BackendCapabilityError, match="PMC"):
        wf.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.55e-06, ky=0.0, x_range=(0.0, 1e-06), z_range=(-1e-06, 1e-06), boundary=materials.PMC)


def test_solver_options_validate_direct_solver_controls() -> None:
    solver = wf.WaveguideScatteringSolver2D(frequency=1e9, x_range=1., z_range=2.)
    with pytest.raises(wf.ConfigurationError, match="tolerance"):
        solver.solve(linear_solver_tolerance=0.0)
    with pytest.raises(wf.ConfigurationError, match="quadrature_order"):
        solver.mesh(quadrature_order=2.5)


def test_obsolete_run_workflow_and_implicit_saving_are_unavailable() -> None:
    simulation = wf.WaveguideScatteringSolver2D(frequency=193400000000000.0, ky=0.0, x_range=(0.0, 1e-06), z_range=(-1e-06, 1e-06), boundary=materials.PEC)

    assert not hasattr(simulation, 'run')
    with pytest.raises(TypeError, match='h5_path'):
        simulation.solve(h5_path=None)


def test_integrated_solver_rejects_lossy_uniform_leads() -> None:
    simulation = wf.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.55e-06, ky=0.0, x_range=(0.0, 1e-06), z_range=(-2e-06, 2e-06), background_material=materials.Material(epsilon=1.0 - 0.01j, mu=1.0), boundary=materials.PEC)
    with pytest.raises(wf.ConfigurationError, match="lossless uniform lead"):
        simulation.solve_modes(max_refinements=0, num_modes=1, neff_guess=1.0)


def test_integrated_solver_rejects_evanescent_incident_mode() -> None:
    simulation = wf.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.55e-06, ky=0.0, x_range=(0.0, 2e-07), z_range=(-2e-06, 2e-06), background_material=materials.Material(epsilon=1.0, mu=1.0), boundary=materials.PEC)
    mode = simulation.solve_modes(
        max_refinements=0,
        num_modes=1,
        neff_guess=-3.7j,
        num_elements=32,
    )[0]
    assert mode.classification == "evanescent"
    with pytest.raises(wf.ConfigurationError, match="propagating, unit-power"):
        simulation.set_incident_mode(mode)


def test_open_integrated_modes_reject_unbound_pml_box_modes() -> None:
    simulation = wf.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.0, ky=0.0, x_range=(-0.5, 0.5), z_range=(-1.0, 1.0), background_material=materials.Material(epsilon=2.25, mu=1.0))
    simulation.add_pml(target_reflection=0.1, thickness=0.25, direction='all')
    with pytest.raises(wf.ModeSolverError, match="No bound guided mode"):
        simulation.solve_modes(max_refinements=0, num_modes=1, neff_guess=1.44, num_elements=40)


def test_high_level_solver_rejects_active_material_but_material_object_allows_it() -> None:
    active = _internal_Material(eps_r=1.0 + 0.01j)
    assert not active.is_passive
    with pytest.raises(wf.ConfigurationError, match="passive materials only"):
        wf.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.55e-06, ky=0.0, x_range=(0.0, 1e-06), z_range=(-1e-06, 1e-06), background_material=materials.Material(epsilon=active.eps_r, mu=1.0), boundary=materials.PEC)


def test_separate_pml_calls_preserve_the_other_axis() -> None:
    simulation = wf.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.55e-06, ky=0.0, x_range=(-2e-06, 2e-06), z_range=(-3e-06, 3e-06))
    simulation.add_pml(thickness=5e-07, direction='x')
    simulation.add_pml(thickness=7e-07, direction='z')
    assert simulation.pml.x is not None
    assert simulation.pml.z is not None


def test_resolving_modes_clears_incident_and_rejects_stale_mode_object() -> None:
    simulation = wf.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.0, ky=0.0, x_range=(0.0, 1.0), z_range=(-2.0, 2.0), boundary=materials.PEC)
    first = simulation.solve_modes(
        max_refinements=0,
        num_modes=1,
        neff_guess=1.0,
        num_elements=24,
    )[0]
    simulation.set_incident_mode(first)
    assert simulation.incident is not None

    simulation.solve_modes(max_refinements=0, num_modes=1, neff_guess=1.0, num_elements=24)

    assert simulation.incident is None
    with pytest.raises(wf.ConfigurationError, match="external or stale"):
        simulation.set_incident_mode(first)


def test_callback_device_accepts_explicit_compatible_mode_set() -> None:
    simulation = wf.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.0, ky=0.0, x_range=((0.0, 1.0), (-2.0, 2.0))[0], z_range=((0.0, 1.0), (-2.0, 2.0))[1], boundary=materials.PEC)
    simulation.set_material_field(material=materials.SpatialMaterial(name="actual", epsilon=lambda x, z: 1.0 + 0.02 * (np.abs(z) < 0.2)), background_material=materials.SpatialMaterial(name="background", epsilon=lambda x: np.ones_like(x)))
    cross_section = _internal_CrossSection(
        (0.0, 1.0),
        background=_internal_Material(),
        boundary="pec",
    )
    modes = _internal_ModeSolver(
        cross_section,
        wavelength=1.0,
        ky=0.0,
        num_elements=24,
    ).solve(max_refinements=0, num_modes=1, neff_guess=np.sqrt(0.75))

    bound = simulation.set_modes(modes)
    incident = simulation.set_incident_mode(bound[0])

    assert simulation.modes is bound
    assert incident.mode is bound[0]

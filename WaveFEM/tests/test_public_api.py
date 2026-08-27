import numpy as np
import pytest

import wavefem as wf


def test_documented_top_level_api_is_importable() -> None:
    expected = {
        "CrossSection",
        "FrequencySweepResult",
        "H5FileData",
        "H5ModeData",
        "H5ResultData",
        "IncidentMode",
        "Material",
        "Mode",
        "ModeSet",
        "ModeSolver",
        "PML",
        "Scattering2D",
        "ScatteringResult",
        "SolverOptions",
        "load_h5",
        "save_result_h5",
        "save_sweep_h5",
    }
    assert expected <= set(wf.__all__)
    assert all(hasattr(wf, name) for name in expected)


def test_scattering_rejects_unimplemented_pmc_instead_of_imposing_pec() -> None:
    with pytest.raises(NotImplementedError, match="PMC"):
        wf.Scattering2D(
            wavelength=1.55e-6,
            ky=0.0,
            x_span=(0.0, 1.0e-6),
            z_span=(-1.0e-6, 1.0e-6),
            transverse_boundary="pmc",
        )


def test_solver_options_validate_direct_solver_controls() -> None:
    options = wf.SolverOptions(tolerance=1e-9, quadrature_order=5)
    assert options.linear_solver == "direct"
    assert options.tolerance == pytest.approx(1e-9)
    with pytest.raises(wf.ConfigurationError, match="tolerance"):
        wf.SolverOptions(tolerance=0.0)
    with pytest.raises(wf.ConfigurationError, match="quadrature_order"):
        wf.SolverOptions(quadrature_order=2.5)


def test_run_rejects_disabling_its_required_hdf5_output() -> None:
    simulation = wf.Scattering2D(
        frequency=193.4e12,
        ky=0.0,
        x_span=(0.0, 1.0e-6),
        z_span=(-1.0e-6, 1.0e-6),
        transverse_boundary="pec",
    )

    with pytest.raises(wf.ConfigurationError, match="requires an HDF5 destination"):
        simulation.run(h5_path=None)  # type: ignore[arg-type]


def test_integrated_solver_rejects_lossy_uniform_leads() -> None:
    simulation = wf.Scattering2D(
        wavelength=1.55e-6,
        ky=0.0,
        x_span=(0.0, 1.0e-6),
        z_span=(-2.0e-6, 2.0e-6),
        background_eps=1.0 + 0.01j,
        transverse_boundary="pec",
    )
    with pytest.raises(wf.ConfigurationError, match="lossless uniform lead"):
        simulation.solve_modes(num_modes=1, neff_guess=1.0)


def test_integrated_solver_rejects_evanescent_incident_mode() -> None:
    simulation = wf.Scattering2D(
        wavelength=1.55e-6,
        ky=0.0,
        x_span=(0.0, 0.20e-6),
        z_span=(-2.0e-6, 2.0e-6),
        background_eps=1.0,
        transverse_boundary="pec",
    )
    mode = simulation.solve_modes(
        num_modes=1,
        neff_guess=3.7j,
        num_elements=32,
    )[0]
    assert mode.classification == "evanescent"
    with pytest.raises(wf.ConfigurationError, match="propagating, unit-power"):
        simulation.set_incident_mode(mode)


def test_open_integrated_modes_reject_unbound_pml_box_modes() -> None:
    simulation = wf.Scattering2D(
        wavelength=1.0,
        ky=0.0,
        x_span=(-0.5, 0.5),
        z_span=(-1.0, 1.0),
        background_eps=2.25,
    )
    simulation.add_pml(x=0.25, z=0.25, target_reflection=0.1)
    with pytest.raises(wf.ModeSolverError, match="No bound guided mode"):
        simulation.solve_modes(num_modes=1, neff_guess=1.44, num_elements=40)


def test_high_level_solver_rejects_active_material_but_material_object_allows_it() -> None:
    active = wf.Material(eps_r=1.0 - 0.01j)
    assert not active.is_passive
    with pytest.raises(wf.ConfigurationError, match="passive materials only"):
        wf.Scattering2D(
            wavelength=1.55e-6,
            ky=0.0,
            x_span=(0.0, 1.0e-6),
            z_span=(-1.0e-6, 1.0e-6),
            background_eps=active.eps_r,
            transverse_boundary="pec",
        )


def test_separate_pml_calls_preserve_the_other_axis() -> None:
    simulation = wf.Scattering2D(
        wavelength=1.55e-6,
        ky=0.0,
        x_span=(-2.0e-6, 2.0e-6),
        z_span=(-3.0e-6, 3.0e-6),
    )
    simulation.add_pml(x=0.5e-6)
    simulation.add_pml(z=0.7e-6)
    assert simulation.pml.x is not None
    assert simulation.pml.z is not None


def test_resolving_modes_clears_incident_and_rejects_stale_mode_object() -> None:
    simulation = wf.Scattering2D(
        wavelength=1.0,
        ky=0.0,
        x_span=(0.0, 1.0),
        z_span=(-2.0, 2.0),
        transverse_boundary="pec",
    )
    first = simulation.solve_modes(
        num_modes=1,
        neff_guess=1.0,
        num_elements=24,
    )[0]
    simulation.set_incident_mode(first)
    assert simulation.incident is not None

    simulation.solve_modes(num_modes=1, neff_guess=1.0, num_elements=24)

    assert simulation.incident is None
    with pytest.raises(wf.ConfigurationError, match="external or stale"):
        simulation.set_incident_mode(first)


def test_callback_device_accepts_explicit_compatible_mode_set() -> None:
    simulation = wf.Scattering2D.from_material_function(
        wavelength=1.0,
        ky=0.0,
        domain=((0.0, 1.0), (-2.0, 2.0)),
        eps_r=lambda x, z: 1.0 + 0.02 * (np.abs(z) < 0.2),
        eps_background=lambda x: np.ones_like(x),
        transverse_boundary="pec",
    )
    cross_section = wf.CrossSection(
        (0.0, 1.0),
        background=wf.Material(),
        boundary="pec",
    )
    modes = wf.ModeSolver(
        cross_section,
        wavelength=1.0,
        ky=0.0,
        num_elements=24,
    ).solve(num_modes=1, neff_guess=np.sqrt(0.75))

    bound = simulation.set_modes(modes)
    incident = simulation.set_incident_mode(bound[0])

    assert simulation.modes is bound
    assert incident.mode is bound[0]

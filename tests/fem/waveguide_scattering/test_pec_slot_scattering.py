from __future__ import annotations
from cem_common import Material, SurfaceImpedance, materials, shapes
from fem_waveguide_scattering.hdf5 import load_h5 as _internal_load_h5

import numpy as np
import pytest

import fem_waveguide_scattering as wf
from examples.fem.waveguide_scattering.grounded_slab_slot_2d import MM, build_simulation


def test_grounded_pec_slot_is_preserved_in_modes_and_frequency_clones() -> None:
    simulation = build_simulation()
    cross_section = simulation._cross_section()
    clone = simulation._clone_at_frequency(9.75e9)

    assert [(item.x, item.name) for item in cross_section.pec_boundaries] == [
        (0.0, "ground_plane")
    ]
    assert clone.frequency == pytest.approx(9.75e9)
    assert clone.geometry.pec_sheets == simulation.geometry.pec_sheets
    assert clone.geometry.pec_slots == simulation.geometry.pec_slots
    assert clone.geometry.pec_sheets is not simulation.geometry.pec_sheets
    assert clone.geometry.pec_slots is not simulation.geometry.pec_slots


def test_high_level_pec_api_accepts_only_compact_actual_only_sheet() -> None:
    simulation = wf.WaveguideScatteringSolver2D(frequency=10000000000.0, x_range=(-10.0 * MM, 10.0 * MM), z_range=(-20.0 * MM, 20.0 * MM), boundary=materials.PEC)
    plate = simulation.add_geometry(background=False, name='finite_plate', shape=shapes.Segment(start=(1.0 * MM, (-2.0 * MM, 3.0 * MM)[0]), end=(1.0 * MM, (-2.0 * MM, 3.0 * MM)[1])), material=materials.PEC)
    assert plate.background is False
    assert simulation._perturbation_z_bounds() == (
        plate.shape.start[1],
        plate.shape.end[1],
    )

    with pytest.raises(wf.ConfigurationError, match="must be compact"):
        simulation.add_geometry(background=False, shape=shapes.Segment(start=(2.0 * MM, simulation.z_range[0]), end=(2.0 * MM, simulation.z_range[1])), material=materials.PEC)


@pytest.mark.gmsh
@pytest.mark.slow
def test_grounded_slab_slot_has_boundary_only_scattering_and_h5_scene(
    tmp_path,
) -> None:
    simulation = build_simulation()
    mesh = simulation.mesh(max_element_size=1.0 * MM, wavelength_elements=10)

    assert mesh.released_pec_facets.size > 0
    assert np.intersect1d(mesh.actual_pec_facets, mesh.released_pec_facets).size == 0
    modes = simulation.solve_modes(
        max_refinements=0,
        num_modes=1,
        neff_guess=1.8,
        num_elements=96,
    )
    assert modes[0].neff.real == pytest.approx(1.798, abs=0.01)
    simulation.set_incident_mode(modes[0])
    destination = tmp_path / "grounded_slab_slot.h5"
    result = simulation.solve(max_refinements=0)
    result.save(destination)
    stored = _internal_load_h5(destination).results[0]

    assert result.solve_info["source_active_fraction"] == 0.0
    assert result.solve_info["released_pec_facet_count"] == mesh.released_pec_facets.size
    assert np.linalg.norm(result.E_scattered) > 0.0
    assert result.reflection > 1e-3
    assert result.radiated_power > 0.0
    assert result.absorbed_power == 0.0
    # The current released-screen source is a first-order boundary model; keep
    # a regression ceiling while its independent flux closure is improved.
    assert result.power_balance_error < 1.2e-1
    assert np.isfinite((result.S11.real, result.S11.imag, result.S21.real, result.S21.imag)).all()

    assert stored.scene is not None
    ground_lines = [
        line for line in stored.scene.lines
        if line.kind == "pec" and line.label == "ground_plane"
    ]
    assert len(ground_lines) == 2
    np.testing.assert_allclose(ground_lines[0].endpoints[:, 0], 0.0)
    np.testing.assert_allclose(ground_lines[1].endpoints[:, 0], 0.0)
    assert ground_lines[0].endpoints[1, 1] == pytest.approx(-1.0 * MM)
    assert ground_lines[1].endpoints[0, 1] == pytest.approx(1.0 * MM)

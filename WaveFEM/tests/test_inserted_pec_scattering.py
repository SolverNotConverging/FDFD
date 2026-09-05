from __future__ import annotations

import numpy as np
import pytest

import wavefem as wf


MM = 1.0e-3


@pytest.mark.gmsh
@pytest.mark.slow
def test_oblique_ground_slot_with_finite_top_plates_uses_both_pec_perturbations() -> None:
    frequency_hz = 20.0e9
    simulation = wf.Scattering2D(
        frequency=frequency_hz,
        angle=np.degrees(np.arcsin(0.10)),
        x_span=(-8.0 * MM, 10.0 * MM),
        z_span=(-15.0 * MM, 15.0 * MM),
    )
    simulation.add_rectangle(
        x=(0.0, 1.27 * MM),
        z="all",
        eps=10.2,
        background=True,
        name="dielectric_slab",
    )
    ground = simulation.add_pec(
        x=0.0,
        background=True,
        name="ground",
    )
    simulation.add_slot(
        pec=ground,
        z=(-0.5 * MM, 0.5 * MM),
        name="ground_slot",
    )
    simulation.add_pec(
        x=1.27 * MM,
        z=(-2.0 * MM, -0.8 * MM),
        background=False,
        name="left_top_plate",
    )
    simulation.add_pec(
        x=1.27 * MM,
        z=(0.8 * MM, 2.0 * MM),
        background=False,
        name="right_top_plate",
    )
    simulation.add_pml(x=2.0 * MM, z=3.0 * MM)
    simulation.set_monitors(left=-6.0 * MM, right=6.0 * MM)

    mesh = simulation.mesh(
        max_element_size=0.5 * MM,
        wavelength_elements=7,
    )
    modes = simulation.solve_modes(
        max_refinements=0,
        num_modes=1,
        neff_guess=1.8,
        num_elements=192,
    )
    simulation.set_incident_mode(modes[0])
    result = simulation.solve(max_refinements=0)

    assert mesh.released_pec_facets.size > 0
    assert mesh.inserted_pec_facets.size > 0
    assert result.solve_info["source_active_fraction"] == 0.0
    assert result.solve_info["released_pec_facet_count"] == mesh.released_pec_facets.size
    assert result.solve_info["inserted_pec_facet_count"] == mesh.inserted_pec_facets.size
    assert result.solve_info["prescribed_pec_dof_count"] > 0
    assert np.linalg.norm(result.E_scattered) > 0.0
    assert np.isfinite(
        (
            result.S11.real,
            result.S11.imag,
            result.S21.real,
            result.S21.imag,
        )
    ).all()
    assert result.power_balance_error < 2.0e-2
    plate_labels = {
        line.label
        for line in result.scene.lines
        if line.kind == "pec" and "top_plate" in line.label
    }
    assert plate_labels == {"left_top_plate", "right_top_plate"}


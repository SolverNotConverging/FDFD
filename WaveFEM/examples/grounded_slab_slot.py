"""Leaky-wave antenna formed by a finite slot in a grounded dielectric slab.

The background lead contains a zero-thickness PEC ground plane invariant in
z.  The actual device releases a finite part of that plane, so the aperture is
a boundary perturbation rather than a permittivity perturbation.  The default
run saves a complete HDF5 result, opens it in ``wavefem-viewer``, and displays
the electric-field magnitude with Matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import wavefem as wf

MM = 1.0e-3
DESIGN_FREQUENCY_HZ = 20.0e9
CORE_EPS_R = 10.2


def build_simulation(frequency_hz: float = DESIGN_FREQUENCY_HZ) -> wf.Scattering2D:
    """Return the grounded-slab slot configuration at one frequency."""

    simulation = wf.Scattering2D(
        frequency=frequency_hz,
        angle=0.0,
        x_span=(-20.0 * MM, 20.0 * MM),
        z_span=(-30.0 * MM, 30.0 * MM),
        background_eps=1.0,
    )
    simulation.add_rectangle(
        x=(0.0, 1.27 * MM),
        z="all",
        eps=CORE_EPS_R,
        background=True,
        name="dielectric_slab",
    )
    ground = simulation.add_pec(
        x=0.0,
        z="all",
        background=True,
        name="ground_plane",
    )
    simulation.add_slot(
        pec=ground,
        z=(-1.0 * MM, 1.0 * MM),
        name="ground_slot",
    )
    simulation.add_pml(x=4.0 * MM, z=6.0 * MM, target_reflection=1e-8)
    simulation.set_monitors(left=-20.0 * MM, right=20.0 * MM)
    return simulation


def solve_single(output: Path) -> wf.ScatteringResult:
    """Run the design-frequency solve and save its full HDF5 record."""

    simulation = build_simulation()
    mesh = simulation.mesh(max_element_size=2.0 * MM, wavelength_elements=10)
    modes = simulation.solve_modes(
        num_modes=1,
        neff_guess=1.8,
        num_elements=96,
    )
    simulation.set_incident_mode(modes[0])
    result = simulation.run(h5_path=output)

    print("selected maximum edge (mm) =", mesh.info.requested_maximum_edge / MM)
    print("surface-mode effective index =", modes[0].neff)
    print("S11 =", result.S11)
    print("S21 =", result.S21)
    print("R, T =", result.reflection, result.transmission)
    print("radiated, absorbed power (W/m) =", result.radiated_power, result.absorbed_power)
    print("power-balance error =", result.power_balance_error)
    print("released PEC facets =", result.solve_info["released_pec_facet_count"])
    print("HDF5 result =", result.h5_path)
    return result


def main() -> None:
    result = solve_single(Path("grounded_slab_slot.h5"))
    result.visualize("E", quantity="abs")
    result.visualize_with_gui()


if __name__ == "__main__":
    main()

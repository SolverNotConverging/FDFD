"""Grounded-slot radiation using matched ports and a closed NF2FF rectangle.

Writes an HDF5 result, numerical pattern, and PNG without opening a viewer.
Use build_simulation(matched_ports=False) for a longitudinal-PML comparison.
"""
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

try:
    from .grounded_slab_slot_2d import build_simulation
except ImportError:  # direct script execution
    from grounded_slab_slot_2d import build_simulation


OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs/examples/fem/waveguide_scattering/closed_contour_farfield_2d"


def solve_example():
    solver = build_simulation(matched_ports=True)
    solver.set_nf2ff_contour(x_range=(-.012, .012), z_range=(-.015, .015))
    solver.mesh(max_element_size=.001, element_order=2, wavelength_elements=10)
    # Resolve the thin substrate in the separate one-dimensional lead mesh.
    solver.solve_modes(num_modes=1, neff_guess=1.8, num_elements=384, max_refinements=0)
    solver.set_incident_mode(0)
    return solver.solve(max_refinements=0)


def save_pattern(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Midpoint angles avoid exact grazing, and omit a repeated 2*pi endpoint.
    theta = (np.arange(360) + .5) * 2*np.pi/360
    far = result.far_field(theta)
    result.save(directory / "results.h5")
    np.savez(directory / "far_field.npz", theta=theta, amplitude=far.amplitude,
             s_amplitude=far.s_amplitude, p_amplitude=far.p_amplitude,
             power_density=far.power_density, directivity=far.directivity,
             gain=far.gain, realized_gain=far.realized_gain,
             transverse_wavenumber=far.transverse_wavenumber)
    figure = Figure(figsize=(10, 9), layout="constrained")
    for index,(title,values) in enumerate((
        ("Power density (W/m/rad)",far.power_density),
        ("Directivity (2D)",far.directivity),
        ("Gain (2D)",far.gain),
        ("Realized gain (2D)",far.realized_gain),
    ),start=1):
        ax = figure.add_subplot(2,2,index,projection="polar")
        ax.plot(np.r_[theta,theta[0]+2*np.pi], np.r_[values,values[0]])
        ax.set_theta_zero_location("E")
        ax.set_title(title)
    figure.savefig(directory / "far_field.png", dpi=160)
    print("Closed-contour radiation (W/m):", far.integrated_power())
    print("Independent flux/modal estimate (W/m):", result.radiated_power)
    print("Energy balance relative error:", result.power_balance_error)
    print("Pattern and reloadable contour:", directory)
    return far


if __name__ == "__main__":
    save_pattern(solve_example(), OUTPUT_DIR)

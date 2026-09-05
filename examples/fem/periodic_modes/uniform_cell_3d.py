"""Uniform 3D periodic cell compared with the analytic TE10 effective index."""

import numpy as np
from fem_periodic_modes import PeriodicModeSolver3D
from cem_common import Material


def main():
    dielectric = Material(name="uniform dielectric", epsilon=2.25)
    common = dict(frequency=10e9, x_range=.02, z_range=.005,
                  background_material=dielectric)
    vector = PeriodicModeSolver3D(**common, y_range=0.01)
    # A fixed 3D mesh must resolve the Gauss filter without adaptive retries.
    vector.mesh(max_element_size=0.006, wavelength_elements=8)
    te10 = vector.solve(num_modes=1, max_refinements=0, neff_guess=1.3)
    expected = np.sqrt(2.25 - (np.pi / (.02 * vector.k0))**2)
    print("3D TE10 effective index:", te10[0].neff)
    print("Analytic TE10 effective index:", expected)
    vector.show()
    return te10


if __name__ == "__main__":
    main()

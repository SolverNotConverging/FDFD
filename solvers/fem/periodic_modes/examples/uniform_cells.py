"""Fixed-mesh 2D TEM and 3D TE10 periodic-cell benchmarks."""

import numpy as np
from fem_periodic_modes import PeriodicModeSolver2D, PeriodicModeSolver3D


def main():
    common = dict(frequency=10e9, x_range=.02, z_range=.005,
                  background_epsilon=2.25)
    scalar = PeriodicModeSolver2D(**common, polarization='TM')
    scalar.mesh(max_element_size=0.003, wavelength_elements=8)
    tem = scalar.solve(num_modes=1, max_refinements=0, neff_guess=1.5)
    print("2D TEM effective index (exact 1.5):", tem[0].neff)

    vector = PeriodicModeSolver3D(**common, y_range=0.01)
    # A fixed 3D mesh must resolve the Gauss filter without adaptive retries.
    vector.mesh(max_element_size=0.006, wavelength_elements=8)
    te10 = vector.solve(num_modes=1, max_refinements=0, neff_guess=1.3)
    expected = np.sqrt(2.25 - (np.pi / (.02 * vector.k0))**2)
    print("3D TE10 effective index:", te10[0].neff)
    print("Analytic TE10 effective index:", expected)
    scalar.show()
    vector.show()
    return tem, te10


if __name__ == "__main__":
    main()

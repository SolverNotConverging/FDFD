"""Fixed-mesh 2D TEM and 3D TE10 periodic-cell benchmarks."""

import numpy as np
from FEM_Periodic_Solver import PeriodicModeSolver2D, PeriodicModeSolver3D


def main():
    common = dict(frequency=10e9, x_range=.02, z_range=.005,
                  background_epsilon=2.25, num_modes=1)
    scalar = PeriodicModeSolver2D(**common, polarization="TM", neff_guess=1.5)
    scalar.discretize(max_element_size=.003, wavelength_elements=8)
    tem = scalar.solve(max_refinements=0)
    print("2D TEM effective index (exact 1.5):", tem[0].neff)

    vector = PeriodicModeSolver3D(**common, y_range=.01, neff_guess=1.3)
    # A fixed 3D mesh must resolve the Gauss filter without adaptive retries.
    vector.discretize(max_element_size=.006, wavelength_elements=8)
    te10 = vector.solve(max_refinements=0)
    expected = np.sqrt(2.25 - (np.pi / (.02 * vector.k0))**2)
    print("3D TE10 effective index:", te10[0].neff)
    print("Analytic TE10 effective index:", expected)
    scalar.visualize_with_gui()
    vector.visualize_with_gui()
    return tem, te10


if __name__ == "__main__":
    main()

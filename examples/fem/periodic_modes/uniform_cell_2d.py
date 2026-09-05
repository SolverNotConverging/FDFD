"""Uniform 2D periodic cell with the analytic TEM effective index."""

from fem_periodic_modes import PeriodicModeSolver2D
from cem_common import Material


def main():
    dielectric = Material(name="uniform dielectric", epsilon=2.25)
    common = dict(frequency=10e9, x_range=.02, z_range=.005,
                  background_material=dielectric)
    scalar = PeriodicModeSolver2D(**common, polarization='TM')
    scalar.mesh(max_element_size=0.003, wavelength_elements=8)
    tem = scalar.solve(num_modes=1, max_refinements=0, neff_guess=1.5)
    print("2D TEM effective index (exact 1.5):", tem[0].neff)

    scalar.show()
    return tem


if __name__ == "__main__":
    main()

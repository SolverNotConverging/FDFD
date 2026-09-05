"""Parallel-plate 1D modes compared with analytic cutoffs."""

import numpy as np
from fem_waveguide_modes import ModeSolver1D


def main():
    frequency = 299_792_458.0  # vacuum wavelength = 1 metre
    expected = np.sqrt(.75)
    line = ModeSolver1D(frequency=frequency, x_range=1.0)
    line.mesh(resolution=48)
    line_modes = line.solve(max_refinements=0, num_modes=3, neff_guess=expected)
    print("1D effective indices (TE/TM cutoff pair and TEM):", line_modes.neff)

    line.show()
    return line_modes


if __name__ == "__main__":
    main()

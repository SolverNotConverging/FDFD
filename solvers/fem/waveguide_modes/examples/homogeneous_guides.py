"""Compare fixed-mesh 1D and second-order 2D modes with analytic cutoffs."""

import numpy as np
from matplotlib import pyplot as plt
from fem_waveguide_modes import ModeSolver1D, ModeSolver2D


def main():
    frequency = 299_792_458.0  # vacuum wavelength = 1 metre
    expected = np.sqrt(.75)
    line = ModeSolver1D(frequency=frequency, x_range=1.0)
    line.mesh(resolution=48)
    line_modes = line.solve(max_refinements=0, num_modes=3, neff_guess=expected)
    print("1D effective indices (TE/TM cutoff pair and TEM):", line_modes.neff)

    guide = ModeSolver2D(frequency=frequency, x_range=1.0, y_range=0.5)
    guide.mesh(resolution=(5, 3), element_order=2)
    vector_modes = guide.solve(max_refinements=0, num_modes=1, neff_guess=expected)
    print("2D TE10 effective index:", vector_modes[0].neff)
    print("Analytic TE10 effective index:", expected)
    # Keep both interactive controllers alive until their windows close.
    viewers = [line.show(block=False), guide.show(block=False)]
    plt.show()
    del viewers
    return line_modes, vector_modes


if __name__ == "__main__":
    main()

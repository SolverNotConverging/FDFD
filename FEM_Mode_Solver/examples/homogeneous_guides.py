"""Compare fixed-mesh 1D and second-order 2D modes with analytic cutoffs."""

import numpy as np
from matplotlib import pyplot as plt
from FEM_Mode_Solver import ModeSolver1D, ModeSolver2D, visualize_with_gui


def main():
    frequency = 299_792_458.0  # vacuum wavelength = 1 metre
    expected = np.sqrt(.75)
    line = ModeSolver1D(frequency, 1., 3, neff_guess=expected)
    line.discretize(resolution=48)
    line_modes = line.solve(max_refinements=0)
    print("1D effective indices (TE/TM cutoff pair and TEM):", line_modes.neff)

    guide = ModeSolver2D(frequency, 1., .5, 1, neff_guess=expected)
    guide.discretize(resolution=(5, 3), element_order=2)
    vector_modes = guide.solve(max_refinements=0)
    print("2D TE10 effective index:", vector_modes[0].neff)
    print("Analytic TE10 effective index:", expected)
    # Keep both interactive controllers alive until their windows close.
    viewers = [visualize_with_gui(line, component="Ey", show=False),
               visualize_with_gui(guide, component="Ey", show=False)]
    plt.show()
    del viewers
    return line_modes, vector_modes


if __name__ == "__main__":
    main()

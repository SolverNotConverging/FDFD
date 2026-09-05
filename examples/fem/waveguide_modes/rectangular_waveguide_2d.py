"""Second-order rectangular waveguide modes compared with the analytic TE10 cutoff."""

import numpy as np
from fem_waveguide_modes import ModeSolver2D


def main():
    frequency = 299_792_458.0
    expected = np.sqrt(.75)
    guide = ModeSolver2D(frequency=frequency, x_range=1.0, y_range=0.5)
    guide.mesh(resolution=(5, 3), element_order=2)
    vector_modes = guide.solve(max_refinements=0, num_modes=1, neff_guess=expected)
    print("2D TE10 effective index:", vector_modes[0].neff)
    print("Analytic TE10 effective index:", expected)
    guide.show()
    return vector_modes


if __name__ == "__main__":
    main()

"""Solve a periodic iris-loaded WR-90 rectangular-waveguide filter cell."""

from __future__ import annotations

from fem_periodic_modes import Box, PeriodicModeSolver3D


MM = 1.0e-3


def main() -> None:
    solver = PeriodicModeSolver3D(frequency=12000000000.0, x_range=(0.0, 22.86 * MM), y_range=(0.0, 10.16 * MM), z_range=(0.0, 8.0 * MM), boundary='pec')
    solver.add_pec(shape=Box((0.0, 4.0 * MM), (0.0, 10.16 * MM), (3.6 * MM, 4.4 * MM)), name='left_iris')
    solver.add_pec(shape=Box((18.86 * MM, 22.86 * MM), (0.0, 10.16 * MM), (3.6 * MM, 4.4 * MM)), name='right_iris')
    solver.mesh(max_element_size=4.0 * MM)

    # The fixed-mesh Arnoldi solve can plateau around 2e-10; use a practical
    # eigensolver tolerance while retaining the independent QEP/Gauss filters.
    modes = solver.solve(direction='all', eigensolver='auto', max_refinements=0, eigensolver_tolerance=1e-08, num_modes=2, neff_guess=0.7)
    print("neff:", modes.neff)
    solver.show()


if __name__ == "__main__":
    main()

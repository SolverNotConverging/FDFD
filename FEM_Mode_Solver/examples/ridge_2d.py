"""Solve a small full-vector rectangular dielectric-waveguide example."""

from __future__ import annotations

from FEM_Mode_Solver import ModeSolver2D


def main() -> None:
    frequency = 10e14
    solver = ModeSolver2D(
        frequency=frequency,
        x_range=(-2.0e-6, 2.0e-6),
        y_range=(-1.5e-6, 1.5e-6),
        num_modes=4,
        background_epsilon=1,
        boundary="pec",
    )
    solver.add_rectangle(
        epsilon=3.45**2,
        mu=1.0,
        x_range=(-0.1e-6, 0.1e-6),
        y_range=(-0.11e-6, 0.11e-6),
        name="silicon_core",
    )
    solver.add_rectangle(
        epsilon=1.44**2,
        mu=1.0,
        x_range=(-1.0e-6, 1.0e-6),
        y_range=(-0.2e-6, -0.11e-6),
        name="slab",
    )

    solver.discretize(max_element_size=100e-9, quadrature_order=4)
    modes = solver.solve(max_refinements=0)

    for number, mode in enumerate(modes, start=1):
        print(
            f"mode {number}: neff={mode.neff:.9g}, "
            f"residual={mode.residual:.3e}, "
            f"divergence={mode.divergence_residual:.3e}"
        )

    solver.visualize_with_gui()


if __name__ == "__main__":
    main()

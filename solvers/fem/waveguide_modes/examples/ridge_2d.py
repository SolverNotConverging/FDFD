"""Solve a small full-vector rectangular dielectric-waveguide example."""

from __future__ import annotations

from fem_waveguide_modes import ModeSolver2D


def main() -> None:
    frequency = 10e14
    solver = ModeSolver2D(frequency=frequency, x_range=(-2e-06, 2e-06), y_range=(-1.5e-06, 1.5e-06), background_epsilon=1, boundary='pec')
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

    solver.mesh(max_element_size=1e-07, quadrature_order=4)
    modes = solver.solve(max_refinements=0, num_modes=4)

    for number, mode in enumerate(modes, start=1):
        print(
            f"mode {number}: neff={mode.neff:.9g}, "
            f"residual={mode.residual:.3e}, "
            f"divergence={mode.divergence_residual:.3e}"
        )

    solver.show()


if __name__ == "__main__":
    main()

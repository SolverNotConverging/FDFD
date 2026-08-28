"""Solve a small full-vector rectangular dielectric-waveguide example."""

from __future__ import annotations

from FEM_Mode_Solver import ModeSolver2D


def main() -> None:
    wavelength = 1.55e-6
    frequency = 299_792_458.0 / wavelength
    solver = ModeSolver2D(
        frequency=frequency,
        x_range=(-2.0e-6, 2.0e-6),
        y_range=(-1.5e-6, 1.5e-6),
        num_modes=2,
        background_epsilon=1.44**2,
        boundary="pec",
    )
    solver.add_rectangle(
        epsilon=3.45**2,
        mu=1.0,
        x_range=(-0.1e-6, 0.1e-6),
        y_range=(-0.11e-6, 0.11e-6),
        name="silicon_core",
    )

    solver.discretize(max_element_size=100e-9, quadrature_order=4)
    modes = solver.solve(neff_guess=3.2, divergence_tolerance=1e-6)

    for number, mode in enumerate(modes, start=1):
        print(
            f"mode {number}: neff={mode.neff:.9g}, "
            f"residual={mode.residual:.3e}, "
            f"divergence={mode.divergence_residual:.3e}"
        )

    solver.visualize(
        mode=1,
        component="E",
        quantity="magnitude",
        material=True,
        mesh=True,
        show=True,
    )


if __name__ == "__main__":
    main()

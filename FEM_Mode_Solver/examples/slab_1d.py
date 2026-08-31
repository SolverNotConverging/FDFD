"""Solve and visualize the leading modes of a dielectric slab."""

from __future__ import annotations

from FEM_Mode_Solver import ModeSolver1D


def main() -> None:
    wavelength = 1.55e-6
    frequency = 299_792_458.0 / wavelength
    solver = ModeSolver1D(
        frequency,
        x_range=(-3.0e-6, 3.0e-6),
        num_modes=4,
        background_epsilon=1.44**2,
    )
    solver.add_layer(
        epsilon=3.45**2,
        mu=1.0,
        x_range=(-0.25e-6, 0.25e-6),
        name="silicon_core",
    )

    solver.discretize(max_element_size=60e-9)
    modes = solver.solve(neff_guess=3.2)

    for number, mode in enumerate(modes, start=1):
        print(
            f"mode {number}: polarization={mode.polarization}, "
            f"neff={mode.neff:.9g}, alpha={mode.alpha:.4g} 1/m, "
            f"residual={mode.residual:.3e}"
        )

    solver.visualize_with_gui()


if __name__ == "__main__":
    main()

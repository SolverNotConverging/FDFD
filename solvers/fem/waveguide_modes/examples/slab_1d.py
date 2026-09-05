"""Solve and visualize the leading modes of a dielectric slab."""

from __future__ import annotations

from fem_waveguide_modes import ModeSolver1D


def main() -> None:
    wavelength = 1.55e-6
    frequency = 299_792_458.0 / wavelength
    solver = ModeSolver1D(frequency=frequency, x_range=(-3e-06, 3e-06), background_epsilon=1.44 ** 2)
    solver.add_layer(
        epsilon=3.45**2,
        mu=1.0,
        x_range=(-0.25e-6, 0.25e-6),
        name="silicon_core",
    )

    solver.mesh(max_element_size=6e-08)
    modes = solver.solve(neff_guess=3.2, max_refinements=0, num_modes=4)

    for number, mode in enumerate(modes, start=1):
        print(
            f"mode {number}: polarization={mode.polarization}, "
            f"neff={mode.neff:.9g}, alpha={mode.alpha:.4g} 1/m, "
            f"residual={mode.residual:.3e}"
        )

    solver.show()


if __name__ == "__main__":
    main()

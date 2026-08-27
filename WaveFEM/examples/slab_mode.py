"""Solve full-vector modes of a symmetric dielectric slab.

The outer PEC walls are explicit.  Increase ``padding`` and ``num_elements``
together to verify that a bound mode is insensitive to this truncation before
using it as an incident field.
"""

from __future__ import annotations

from wavefem.constants import C0
from wavefem.materials import Material
from wavefem.modes import CrossSection, ModeSolver


def main() -> None:
    frequency_hz = 193.414489e12
    k0 = 2.0 * 3.141592653589793 * frequency_hz / C0
    padding = 3.0e-6
    half_core = 0.25e-6

    cross_section = CrossSection(
        x_span=(-padding, padding),
        background=Material(eps_r=1.44**2),
        boundary="pec",
    )
    cross_section.add_layer(
        x=(-half_core, half_core),
        material=Material(eps_r=3.45**2),
        name="core",
    )

    solver = ModeSolver(
        cross_section,
        frequency=frequency_hz,
        ky=0.15 * k0,
        num_elements=60,
    )
    modes = solver.solve(num_modes=4, neff_guess=3.2)
    for index, mode in enumerate(modes):
        print(
            f"mode {index}: neff={mode.neff:.9g}, "
            f"power={mode.power:.6g} W/m, direction={mode.direction}, "
            f"residual={mode.residual:.2e}, div={mode.divergence_residual:.2e}"
        )


if __name__ == "__main__":
    main()

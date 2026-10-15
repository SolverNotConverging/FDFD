"""Modes of a copper rectangular waveguide (surface-impedance walls)."""

from pathlib import Path

import fdfd
from fdfd import materials, shapes

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fdfd.ModeSolver2D(frequency=100e9, x_range=(0.0, 12e-3), y_range=(0.0, 10e-3))
# One Boolean frame avoids overlapping conductor assignments at the corners.
wall = shapes.Difference(
    shape=shapes.Rectangle(bounds=((1.9e-3, 10.1e-3), (1.9e-3, 8.1e-3))),
    tool=shapes.Rectangle(bounds=((2e-3, 10e-3), (2e-3, 8e-3))),
)
solver.add_geometry(shape=wall, material=materials.copper, name="copper wall")
solver.mesh(resolution=(120, 100))
result = solver.solve(device="auto", num_modes=4)

print("Effective indices:", result.neff)
print("Attenuation (dB/m):", 20 / 2.302585 * result.attenuation_constant)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

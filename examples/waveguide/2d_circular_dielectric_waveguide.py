"""Hybrid modes of a circular dielectric rod; the HE11 pair is degenerate."""

from pathlib import Path

import fdfd

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fdfd.ModeSolver2D(frequency=100e9, x_range=(0.0, 10e-3), y_range=(0.0, 10e-3))
solver.add_circle(
    center=(5e-3, 5e-3), radius=3e-3, material=fdfd.Material(name="dielectric core", epsilon=6.0), name="core"
)
solver.mesh(resolution=(50, 50))
result = solver.solve(device="auto", num_modes=4)

print("Effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

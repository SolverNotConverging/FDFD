"""Leaky Bloch modes of a periodically loaded dielectric image guide (3D cell).

This takes a few minutes; reduce the resolution for a quick look.
"""

from pathlib import Path

import fdfd
from fdfd import materials

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fdfd.PeriodicModeSolver3D(
    frequency=30e9, x_range=(0.0, 12e-3), y_range=(0.0, 8e-3), z_range=(0.0, 6e-3)
)
solver.add_box(
    x_range=(4e-3, 8e-3),
    y_range=(1e-3, 4e-3),
    z_range=(0.0, 6e-3),
    material=fdfd.Material(name="guide dielectric", epsilon=4.0),
    name="image guide",
)
solver.add_box(
    x_range=(0.0, 12e-3), y_range=(0.0, 1e-3), z_range=(0.0, 6e-3), material=materials.PEC, name="ground"
)
solver.add_box(
    x_range=(4e-3, 8e-3),
    y_range=(4e-3, 5e-3),
    z_range=(2e-3, 4e-3),
    material=materials.PEC,
    name="loading tooth",
)
solver.add_pml(thickness=1.5e-3, sides="x+", strength=1.0)
solver.mesh(resolution=(48, 32, 32))
result = solver.solve(device="auto", num_modes=4, guess=0.0, method="refined")

print("Bloch effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

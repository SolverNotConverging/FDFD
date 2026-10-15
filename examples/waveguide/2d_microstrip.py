"""Quasi-TEM modes of a copper microstrip on a lossy substrate."""

from pathlib import Path

import fdfd
from fdfd import materials

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fdfd.ModeSolver2D(frequency=50e9, x_range=(0.0, 12e-3), y_range=(0.0, 10e-3))
solver.add_rectangle(
    x_range=(2e-3, 10e-3),
    y_range=(4e-3, 5e-3),
    material=fdfd.Material(name="lossy substrate", epsilon=4.0 - 1j),
)
solver.add_rectangle(x_range=(5e-3, 7e-3), y_range=(5e-3, 5.1e-3), material=materials.copper, name="strip")
solver.add_rectangle(
    x_range=(0.5e-3, 11.5e-3), y_range=(3.9e-3, 4e-3), material=materials.copper, name="ground"
)
solver.mesh(resolution=(120, 100))
result = solver.solve(device="auto", num_modes=4)

print("Effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

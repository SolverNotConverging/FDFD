"""A ridge on an anisotropic slab, with PML on the open x ends."""

from pathlib import Path

import fdfd

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fdfd.ModeSolver2D(frequency=50e9, x_range=(0.0, 24e-3), y_range=(0.0, 16e-3))
solver.add_rectangle(
    x_range=(0.0, 24e-3),
    y_range=(6e-3, 8e-3),
    material=fdfd.Material(name="anisotropic slab", epsilon=(3.0, 4.0, 5.0)),
)
solver.add_rectangle(
    x_range=(10e-3, 14e-3), y_range=(8e-3, 10e-3), material=fdfd.Material(name="ridge", epsilon=6.0)
)
solver.add_pml(thickness=3e-3, sides="x", strength=1.0)
solver.mesh(resolution=(80, 56))
result = solver.solve(device="auto", num_modes=4)

print("Effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

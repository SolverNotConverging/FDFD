"""Leaky Bloch modes of a periodically loaded grounded slab (TM, PEC ground).

The geometry matches the FEM reference in benchmarks/periodic; the grid
resolves the 100 um PEC patch and the slab interfaces.
"""

from pathlib import Path

import fdfd
from fdfd import materials

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fdfd.PeriodicModeSolver2D(
    frequency=20e9, x_range=(0.0, 10e-3), z_range=(0.0, 8e-3), boundary=materials.PEC
)
solver.add_rectangle(
    x_range=(0.0, 1.27e-3),
    z_range=(0.0, 8e-3),
    material=fdfd.Material(name="antenna substrate", epsilon=10.2),
    name="grounded slab",
)
solver.add_rectangle(
    x_range=(1.27e-3, 1.37e-3), z_range=(1e-3, 2e-3), material=materials.PEC, name="PEC patch"
)
solver.add_pml(thickness=2.5e-3, sides="x+")
solver.mesh(resolution=(100, 80))
# Leaky (fast-wave) modes have Re(neff) < 1, so search around neff = 0.
result = solver.solve(device="auto", num_modes=4, polarization="TM", guess=0.0, tol=1e-9, ncv=36)

print("Bloch effective indices:", result.neff)
print("Backward errors:", result.info.residuals)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

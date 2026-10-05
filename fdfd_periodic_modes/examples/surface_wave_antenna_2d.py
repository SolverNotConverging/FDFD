"""surface wave antenna 2d with a material-first periodic unit cell.

The compact grid is a workflow demonstration; refine before interpreting leakage.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fdfd_periodic_modes import Material, materials, PeriodicModeSolver2D

OUTPUT = _ROOT / "fdfd_periodic_modes/outputs/examples/surface_wave_antenna_2d"

frequency = 25000000000.0
dielectric = Material(name="guide dielectric", epsilon=4.)
solver = PeriodicModeSolver2D(frequency=frequency, x_range=.01, z_range=.006, polarization="TM")
solver.add_rectangle(x_range=(0., .002), z_range=(0., .006), material=dielectric, name="slab")
solver.add_rectangle(x_range=(0., .0005), z_range=(0., .006), material=materials.PEC, name="ground")
solver.add_rectangle(x_range=(.002, .0025), z_range=(.0015, .003), material=materials.PEC, name="loading tooth")
solver.add_pml(thickness=.0015, direction="x", sigma_max=1.)
solver.mesh(resolution=(240, 160))
result = solver.solve(num_modes=2, neff_guess=1.5, eigensolver="eigs")
print("Bloch effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

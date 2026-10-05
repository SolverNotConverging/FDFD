"""image guide leaky wave antenna 3d with a material-first periodic unit cell.

The compact grid is a workflow demonstration; refine before interpreting leakage.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fdfd_periodic_modes import Material, materials, PeriodicModeSolver3D

OUTPUT = _ROOT / "fdfd_periodic_modes/outputs/examples/image_guide_leaky_wave_antenna_3d"

frequency = 30e9
dielectric = Material(name="guide dielectric", epsilon=4.)
solver = PeriodicModeSolver3D(frequency=frequency, x_range=12e-3, y_range=8e-3, z_range=6e-3)
solver.add_box(x_range=(4e-3, 8e-3), y_range=(1e-3, 4e-3), z_range=(0., 6e-3), material=dielectric, name="image guide")
solver.add_box(x_range=(0., 12e-3), y_range=(0., 1e-3), z_range=(0., 6e-3), material=materials.PEC, name="ground")
solver.add_box(x_range=(4e-3, 8e-3), y_range=(4e-3, 5e-3), z_range=(2e-3, 4e-3), material=materials.PEC, name="loading tooth")
solver.add_pml(thickness=1.5e-3, direction="x", sigma_max=1.)
solver.mesh(resolution=(12, 8, 8))
result = solver.solve(num_modes=2, neff_guess=1.5, eigensolver="eigs")
print("Bloch effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

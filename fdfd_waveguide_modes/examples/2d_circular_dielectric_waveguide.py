"""circular dielectric waveguide 2d: define materials, assign shapes, mesh, and solve."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fdfd_waveguide_modes import Material, materials, ModeSolver2D

OUTPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/2d_circular_dielectric_waveguide"

solver = ModeSolver2D(frequency=100e9, x_range=10e-3, y_range=10e-3, background_material=materials.vacuum)
core = Material(name="dielectric core", epsilon=6.)
solver.add_circle(center=(5e-3, 5e-3), radius=3e-3, material=core, name="core")
solver.mesh(resolution=(50, 50))
result = solver.solve(num_modes=4)
print("Effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

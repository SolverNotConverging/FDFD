"""microstrip 2d: define materials, assign shapes, mesh, and solve."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fdfd_waveguide_modes import Material, materials, ModeSolver2D

OUTPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/2d_microstrip"

solver = ModeSolver2D(frequency=50e9, x_range=12e-3, y_range=10e-3, background_material=materials.vacuum)
substrate = Material(name="lossy substrate", epsilon=4.-1j)
copper = materials.copper
solver.add_rectangle(x_range=(2e-3, 10e-3), y_range=(4e-3, 5e-3), material=substrate)
solver.add_rectangle(x_range=(5e-3, 7e-3), y_range=(5e-3, 5.1e-3), material=copper, name="strip")
solver.add_rectangle(x_range=(500e-6, 11.5e-3), y_range=(3.9e-3, 4e-3), material=copper, name="ground")
solver.mesh(resolution=(120, 100))
result = solver.solve(num_modes=4)
print("Effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

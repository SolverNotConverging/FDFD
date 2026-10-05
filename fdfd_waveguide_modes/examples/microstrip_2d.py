"""microstrip 2d: define materials, assign shapes, mesh, and solve."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fdfd_waveguide_modes import Material, materials, ModeSolver2D

OUTPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/microstrip_2d"

solver = ModeSolver2D(frequency=50e9, x_range=.012, y_range=.010, background_material=materials.vacuum)
substrate = Material(name="lossy substrate", epsilon=4.-1j)
copper = materials.copper
solver.add_rectangle(x_range=(.002, .010), y_range=(.004, .005), material=substrate)
solver.add_rectangle(x_range=(.005, .007), y_range=(.005, .0051), material=copper, name="strip")
solver.add_rectangle(x_range=(.0005, .0115), y_range=(.0039, .004), material=copper, name="ground")
solver.mesh(resolution=(120, 100))
result = solver.solve(num_modes=4, neff_guess=1.7)
print("Effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

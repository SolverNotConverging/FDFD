"""rectangular waveguide 2d: define materials, assign shapes, mesh, and solve."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fdfd_waveguide_modes import materials, shapes, ModeSolver2D

OUTPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/rectangular_waveguide_2d"

solver = ModeSolver2D(frequency=100e9, x_range=.012, y_range=.010, background_material=materials.vacuum)
copper = materials.copper
# One Boolean frame avoids overlapping conductor assignments at the corners.
wall = shapes.Difference(
    shape=shapes.Rectangle(bounds=((.0019, .0101), (.0019, .0081))),
    tool=shapes.Rectangle(bounds=((.002, .010), (.002, .008))),
)
solver.add_geometry(shape=wall, material=copper, name="copper wall")
solver.mesh(resolution=(120, 100))
result = solver.solve(num_modes=4, neff_guess=.99)
print("Effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

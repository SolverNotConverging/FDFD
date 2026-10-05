"""microstrip 2d: define materials, assign shapes, mesh, and solve."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "cem_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path
from cem_common import Material, materials, shapes
from fdfd_waveguide_modes import ModeSolver2D

OUTPUT = _ROOT / "outputs/fdfd_waveguide_modes/examples/microstrip_2d"


def build_solver():
    solver = ModeSolver2D(frequency=50e9, x_range=.012, y_range=.010, background_material=materials.vacuum)
    substrate = Material(name="lossy substrate", epsilon=4.-1j)
    copper = materials.copper
    solver.add_rectangle(x_range=(.002, .010), y_range=(.004, .005), material=substrate)
    solver.add_rectangle(x_range=(.005, .007), y_range=(.005, .0051), material=copper, name="strip")
    solver.add_rectangle(x_range=(.0005, .0115), y_range=(.0039, .004), material=copper, name="ground")
    return solver


def main():
    solver = build_solver()
    solver.mesh(resolution=(120, 100))
    result = solver.solve(num_modes=4, neff_guess=1.7)
    print("Effective indices:", result.neff)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT / "modes.h5")
    result.show()
    return result


if __name__ == "__main__":
    main()

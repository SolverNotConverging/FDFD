"""ridge dielectric waveguide 2d: define materials, assign shapes, mesh, and solve."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "fdfd_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path
from fdfd_common import Material, materials, shapes
from fdfd_waveguide_modes import ModeSolver2D

OUTPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/ridge_dielectric_waveguide_2d"


def build_solver():
    solver = ModeSolver2D(frequency=50e9, x_range=.024, y_range=.016, background_material=materials.vacuum)
    slab = Material(name="anisotropic slab", epsilon=(3., 4., 5.))
    ridge = Material(name="ridge", epsilon=6.)
    solver.add_rectangle(x_range=(0., .024), y_range=(.006, .008), material=slab)
    solver.add_rectangle(x_range=(.010, .014), y_range=(.008, .010), material=ridge)
    solver.add_pml(thickness=.003, direction="x", sigma_max=1.)
    return solver


def main():
    solver = build_solver()
    solver.mesh(resolution=(80, 56))
    result = solver.solve(num_modes=4, neff_guess=2.)
    print("Effective indices:", result.neff)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT / "modes.h5")
    result.show()
    return result


if __name__ == "__main__":
    main()

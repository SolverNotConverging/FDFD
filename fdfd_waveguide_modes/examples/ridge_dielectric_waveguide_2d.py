"""ridge dielectric waveguide 2d: define materials, assign shapes, mesh, and solve."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fdfd_waveguide_modes import Material, materials, ModeSolver2D

OUTPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/ridge_dielectric_waveguide_2d"

solver = ModeSolver2D(frequency=50e9, x_range=24e-3, y_range=16e-3, background_material=materials.vacuum)
slab = Material(name="anisotropic slab", epsilon=(3., 4., 5.))
ridge = Material(name="ridge", epsilon=6.)
solver.add_rectangle(x_range=(0., 24e-3), y_range=(6e-3, 8e-3), material=slab)
solver.add_rectangle(x_range=(10e-3, 14e-3), y_range=(8e-3, 10e-3), material=ridge)
solver.add_pml(thickness=3e-3, direction="x", sigma_max=1.)
solver.mesh(resolution=(80, 56))
result = solver.solve(num_modes=4, neff_guess=2.)
print("Effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

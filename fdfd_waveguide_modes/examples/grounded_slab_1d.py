"""grounded slab 1d: define materials, assign shapes, mesh, and solve."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fdfd_waveguide_modes import Material, materials, ModeSolver1D

OUTPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/grounded_slab_1d"

solver = ModeSolver1D(frequency=30e9, x_range=10e-3, background_material=materials.vacuum)
slab = Material(name="slab", epsilon=10.2)
ground = materials.PEC
solver.add_layer(x_range=(3e-3, 4.27e-3), material=slab)
solver.add_layer(x_range=(2.9e-3, 3e-3), material=ground)
solver.add_pml(thickness=800e-6, sigma_max=10.)
solver.mesh(resolution=1000)
result = solver.solve(num_modes=4)
print("Effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

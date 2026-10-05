"""parallel plate waveguide 1d: define materials, assign shapes, mesh, and solve."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fdfd_waveguide_modes import Material, materials, ModeSolver1D

OUTPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/1d_parallel_plate_waveguide"

solver = ModeSolver1D(frequency=100e9, x_range=8e-3, background_material=materials.vacuum)
dielectric = Material(name="anisotropic fill", epsilon=(4., 5., 6.))
wall = materials.PMC
solver.add_layer(x_range=(3e-3, 4.5e-3), material=dielectric)
solver.add_layer(x_range=(2.9e-3, 3e-3), material=wall)
solver.add_layer(x_range=(4.5e-3, 4.6e-3), material=wall)
solver.add_pml(thickness=800e-6, sigma_max=10.)
solver.mesh(resolution=800)
result = solver.solve(num_modes=4)
print("Effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

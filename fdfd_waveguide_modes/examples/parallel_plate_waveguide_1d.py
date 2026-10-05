"""parallel plate waveguide 1d: define materials, assign shapes, mesh, and solve."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fdfd_waveguide_modes import Material, materials, ModeSolver1D

OUTPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/parallel_plate_waveguide_1d"

solver = ModeSolver1D(frequency=100e9, x_range=.008, background_material=materials.vacuum)
dielectric = Material(name="anisotropic fill", epsilon=(4., 5., 6.))
wall = materials.PMC
solver.add_layer(x_range=(.003, .0045), material=dielectric)
solver.add_layer(x_range=(.0029, .003), material=wall)
solver.add_layer(x_range=(.0045, .0046), material=wall)
solver.add_pml(thickness=.0008, sigma_max=10.)
solver.mesh(resolution=800)
result = solver.solve(num_modes=4, neff_guess=2.)
print("Effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

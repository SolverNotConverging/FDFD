"""Surface-wave modes of a grounded dielectric slab, open on top through a PML."""

from pathlib import Path

import fdfd
from fdfd import materials

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fdfd.ModeSolver1D(frequency=30e9, x_range=(0.0, 10e-3))
solver.add_layer(x_range=(3e-3, 4.27e-3), material=fdfd.Material(name="slab", epsilon=10.2))
solver.add_layer(x_range=(2.9e-3, 3e-3), material=materials.PEC, name="ground")
solver.add_pml(thickness=800e-6)
solver.mesh(resolution=1000)
result = solver.solve(device="auto", num_modes=4)

print("Effective indices:", result.neff)
print("Polarizations:", result.metadata["polarizations"])
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

"""An anisotropic layer between two PMC walls."""

from pathlib import Path

import fdfd
from fdfd import materials

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fdfd.ModeSolver1D(frequency=100e9, x_range=(0.0, 8e-3))
solver.add_layer(
    x_range=(3e-3, 4.5e-3), material=fdfd.Material(name="anisotropic fill", epsilon=(4.0, 5.0, 6.0))
)
solver.add_layer(x_range=(2.9e-3, 3e-3), material=materials.PMC, name="lower wall")
solver.add_layer(x_range=(4.5e-3, 4.6e-3), material=materials.PMC, name="upper wall")
solver.add_pml(thickness=800e-6)
solver.mesh(resolution=800)
result = solver.solve(device="auto", num_modes=4)

print("Effective indices:", result.neff)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()

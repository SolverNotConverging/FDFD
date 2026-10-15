"""TM plane-wave scattering by a perfectly conducting cylinder."""

from pathlib import Path

import fdfd
from fdfd import materials

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fdfd.ScatteringSolver2D(frequency=3e9, x_range=(-150e-3, 150e-3), y_range=(-150e-3, 150e-3))
solver.add_circle(center=(0.0, 0.0), radius=25e-3, material=materials.PEC)
solver.add_pml(thickness=50e-3)
solver.set_source(kind="plane_wave", angle=0.0, inset=75e-3)
solver.mesh(resolution=(240, 240))
result = solver.solve(device="auto", polarization="TM")

OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "scattering.h5")
result.show()

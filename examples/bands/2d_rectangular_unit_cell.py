"""Photonic bands of a rectangular (a x b) lattice of dielectric rods."""

from pathlib import Path

import numpy as np

import fdfd

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

a, b = 10e-3, 7e-3
solver = fdfd.BandStructureSolver2D(x_range=(-a / 2, a / 2), y_range=(-b / 2, b / 2))
solver.add_circle(
    center=(0.0, 0.0), radius=0.2 * min(a, b), material=fdfd.Material(name="dielectric rod", epsilon=8.9)
)
solver.mesh(resolution=(20, 20))
path = fdfd.bloch_path(((0.0, 0.0), (np.pi / a, 0.0), (np.pi / a, np.pi / b), (0.0, 0.0)), num_points=100)
result = solver.solve(device="auto", beta_path=path, num_modes=4)

print("First TE band (GHz):", np.round(result.frequencies["TE"][0].real / 1e9, 3))
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "bands.h5")
result.plot().savefig(OUTPUT / "bands.png", dpi=160)
result.show()

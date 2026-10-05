"""Sweep a dielectric guide, reusing the same material at each frequency."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import csv
import numpy as np

from fdfd_waveguide_modes import Material, ModeSolver2D

OUTPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/dielectric_waveguide_2d_dispersion"

core = Material(name="dielectric core", epsilon=4.)
rows = []
for frequency in np.linspace(20e9, 60e9, 5):
    solver = ModeSolver2D(frequency=frequency, x_range=10e-3, y_range=10e-3)
    solver.add_circle(center=(5e-3, 5e-3), radius=2e-3, material=core)
    solver.mesh(resolution=(40, 40))
    result = solver.solve(num_modes=3, neff_guess=1.8)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT / f"modes_{frequency/1e9:.0f}GHz.h5")
    for mode, neff in enumerate(result.neff, start=1):
        rows.append(dict(frequency_hz=frequency, mode=mode, neff_real=neff.real, neff_imag=neff.imag))
    print(f"{frequency/1e9:.0f} GHz: {result.neff}")
with (OUTPUT / "dispersion.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

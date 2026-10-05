"""Sweep a dielectric guide, reusing the same material at each frequency."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import csv
import numpy as np

from fdfd_waveguide_modes import plot_dispersion, Material, ModeSolver1D

OUTPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/layered_waveguide_1d_dispersion"

core = Material(name="dielectric core", epsilon=4.)
frequencies = np.linspace(20e9, 60e9, 5)
neff_sweep = []
rows = []
for frequency in frequencies:
    solver = ModeSolver1D(frequency=frequency, x_range=10e-3)
    solver.add_layer(x_range=(3e-3, 7e-3), material=core)
    solver.mesh(resolution=200)
    result = solver.solve(num_modes=3, neff_guess=1.8)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT / f"modes_{frequency/1e9:.0f}GHz.h5")
    neff_sweep.append(result.neff)
    for mode, neff in enumerate(result.neff, start=1):
        rows.append(dict(frequency_hz=frequency, mode=mode, neff_real=neff.real, neff_imag=neff.imag))
    print(f"{frequency/1e9:.0f} GHz: {result.neff}")
with (OUTPUT / "dispersion.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

plot_dispersion(frequencies, neff_sweep)

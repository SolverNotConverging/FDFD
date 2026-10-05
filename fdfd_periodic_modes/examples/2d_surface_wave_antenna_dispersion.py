"""Explicit frequency sweep of the 2D periodic guide."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import csv
import numpy as np
from tqdm.auto import tqdm
from fdfd_periodic_modes import plot_dispersion, Material, materials, PeriodicModeSolver2D

OUTPUT = _ROOT / "fdfd_periodic_modes/outputs/examples/2d_surface_wave_antenna_dispersion"

frequencies = np.linspace(25e9, 35e9, 30)
neff_sweep = []
rows = []
for case, frequency in enumerate(tqdm(frequencies, desc="Frequency sweep", unit="frequency"), start=1):
    dielectric = Material(name="guide dielectric", epsilon=4.)
    solver = PeriodicModeSolver2D(frequency=frequency, x_range=10e-3, z_range=6e-3, polarization="TM")
    solver.add_rectangle(x_range=(0., 2e-3), z_range=(0., 6e-3), material=dielectric, name="slab")
    solver.add_rectangle(x_range=(0., 500e-6), z_range=(0., 6e-3), material=materials.PEC, name="ground")
    solver.add_rectangle(x_range=(2e-3, 2.5e-3), z_range=(1.5e-3, 3e-3), material=materials.PEC, name="loading tooth")
    solver.add_pml(thickness=1.5e-3, direction="x", sigma_max=1.)
    solver.mesh(resolution=(24, 16))
    result = solver.solve(num_modes=4, neff_guess=0, eigensolver="eigs", ncv=40)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT / f"case_{case:03d}_{frequency / 1e9:.6g}GHz.h5")
    neff_sweep.append(result.neff)
    for mode, neff in enumerate(result.neff, start=1):
        rows.append(dict(frequency_hz=frequency, mode=mode, neff_real=neff.real, neff_imag=neff.imag))
with (OUTPUT / "dispersion.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

plot_dispersion(frequencies, neff_sweep)

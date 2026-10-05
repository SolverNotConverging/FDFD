"""Explicit frequency sweep of the 2D periodic guide."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import csv
import numpy as np
from fdfd_periodic_modes import Material, materials, PeriodicModeSolver2D

OUTPUT = _ROOT / "fdfd_periodic_modes/outputs/examples/surface_wave_antenna_2d_dispersion"

rows=[]
for frequency in np.linspace(25e9,35e9,3):
    dielectric = Material(name="guide dielectric", epsilon=4.)
    solver = PeriodicModeSolver2D(frequency=frequency, x_range=.01, z_range=.006, polarization="TM")
    solver.add_rectangle(x_range=(0., .002), z_range=(0., .006), material=dielectric, name="slab")
    solver.add_rectangle(x_range=(0., .0005), z_range=(0., .006), material=materials.PEC, name="ground")
    solver.add_rectangle(x_range=(.002, .0025), z_range=(.0015, .003), material=materials.PEC, name="loading tooth")
    solver.add_pml(thickness=.0015, direction="x", sigma_max=1.)
    solver.mesh(resolution=(24, 16))
    result=solver.solve(num_modes=2,neff_guess=1.5,eigensolver="eigs")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    result.save(OUTPUT / f"modes_{frequency/1e9:.0f}GHz.h5")
    for mode,neff in enumerate(result.neff):
        rows.append(dict(frequency_hz=frequency,mode=mode,neff_real=neff.real,neff_imag=neff.imag))
with (OUTPUT / "dispersion.csv").open("w",newline="") as stream:
    writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)

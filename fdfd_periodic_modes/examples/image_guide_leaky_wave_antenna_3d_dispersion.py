"""Explicit frequency sweep of the 3D periodic guide."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import csv
import numpy as np
from fdfd_periodic_modes import Material, materials, PeriodicModeSolver3D

OUTPUT = _ROOT / "fdfd_periodic_modes/outputs/examples/image_guide_leaky_wave_antenna_3d_dispersion"

rows=[]
for frequency in np.linspace(25e9,35e9,3):
    dielectric = Material(name="guide dielectric", epsilon=4.)
    solver = PeriodicModeSolver3D(frequency=frequency, x_range=12e-3, y_range=8e-3, z_range=6e-3)
    solver.add_box(x_range=(4e-3, 8e-3), y_range=(1e-3, 4e-3), z_range=(0., 6e-3), material=dielectric, name="image guide")
    solver.add_box(x_range=(0., 12e-3), y_range=(0., 1e-3), z_range=(0., 6e-3), material=materials.PEC, name="ground")
    solver.add_box(x_range=(4e-3, 8e-3), y_range=(4e-3, 5e-3), z_range=(2e-3, 4e-3), material=materials.PEC, name="loading tooth")
    solver.add_pml(thickness=1.5e-3, direction="x", sigma_max=1.)
    solver.mesh(resolution=(12, 8, 8))
    result=solver.solve(num_modes=2,neff_guess=1.5,eigensolver="eigs")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    result.save(OUTPUT / f"modes_{frequency/1e9:.0f}GHz.h5")
    for mode,neff in enumerate(result.neff):
        rows.append(dict(frequency_hz=frequency,mode=mode,neff_real=neff.real,neff_imag=neff.imag))
with (OUTPUT / "dispersion.csv").open("w",newline="") as stream:
    writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)

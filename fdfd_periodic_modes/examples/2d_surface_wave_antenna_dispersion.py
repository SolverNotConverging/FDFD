"""Grounded-slab leaky-wave dispersion, matching the validated FEM geometry."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import csv
import numpy as np
from tqdm.auto import tqdm
from fdfd_periodic_modes import plot_dispersion, Material, materials, PeriodicModeSolver2D

OUTPUT = _ROOT / "fdfd_periodic_modes/outputs/examples/2d_surface_wave_antenna_dispersion"

frequencies = np.linspace(18e9, 22e9, 50)
neff_sweep = []
rows = []
for case, frequency in enumerate(tqdm(frequencies, desc="Frequency sweep", unit="frequency"), start=1):
    substrate = Material(name="antenna substrate", epsilon=10.2)
    solver = PeriodicModeSolver2D(frequency=frequency, x_range=(0., 10e-3),
                                  z_range=(0., 8e-3), polarization="TM", boundary=materials.PEC)
    solver.add_rectangle(x_range=(0., 1.27e-3), z_range=(0., 8e-3),
                         material=substrate, name="grounded_dielectric_slab")
    solver.add_rectangle(x_range=(1.27e-3, 1.37e-3), z_range=(1e-3, 2e-3),
                         material=materials.PEC, name="top_pec_perturbation")
    solver.add_pml(thickness=2.5e-3, direction="x+")
    solver.mesh(resolution=(100, 80))
    result = solver.solve(num_modes=4, neff_guess=0., eigensolver_tolerance=1e-9, ncv=36)
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

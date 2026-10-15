"""Frequency sweep of the 3D image-guide cell (several minutes per frequency)."""

from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import fdfd
from fdfd import materials

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

dielectric = fdfd.Material(name="guide dielectric", epsilon=4.0)
frequencies = np.linspace(25e9, 35e9, 3)
neff = []
OUTPUT.mkdir(parents=True, exist_ok=True)
for frequency in tqdm(frequencies, desc="Frequency sweep", unit="frequency"):
    solver = fdfd.PeriodicModeSolver3D(
        frequency=frequency, x_range=(0.0, 12e-3), y_range=(0.0, 8e-3), z_range=(0.0, 6e-3)
    )
    solver.add_box(
        x_range=(4e-3, 8e-3),
        y_range=(1e-3, 4e-3),
        z_range=(0.0, 6e-3),
        material=dielectric,
        name="image guide",
    )
    solver.add_box(
        x_range=(0.0, 12e-3), y_range=(0.0, 1e-3), z_range=(0.0, 6e-3), material=materials.PEC, name="ground"
    )
    solver.add_box(
        x_range=(4e-3, 8e-3),
        y_range=(4e-3, 5e-3),
        z_range=(2e-3, 4e-3),
        material=materials.PEC,
        name="loading tooth",
    )
    solver.add_pml(thickness=1.5e-3, sides="x+", strength=1.0)
    solver.mesh(resolution=(48, 32, 32))
    result = solver.solve(device="auto", num_modes=4, guess=0.0)
    result.save(OUTPUT / f"{frequency / 1e9:08.3f}GHz.h5")
    neff.append(result.neff)

fdfd.plot_dispersion(frequencies, neff, connect=False)

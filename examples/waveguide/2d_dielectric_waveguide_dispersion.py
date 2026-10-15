"""Frequency sweep of a circular dielectric rod; one saved result per frequency."""

from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import fdfd

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

core = fdfd.Material(name="dielectric core", epsilon=4.0)
frequencies = np.linspace(20e9, 60e9, 5)
neff = []
OUTPUT.mkdir(parents=True, exist_ok=True)
for frequency in tqdm(frequencies, desc="Frequency sweep", unit="frequency"):
    solver = fdfd.ModeSolver2D(frequency=frequency, x_range=(0.0, 10e-3), y_range=(0.0, 10e-3))
    solver.add_circle(center=(5e-3, 5e-3), radius=2e-3, material=core)
    solver.mesh(resolution=(40, 40))
    result = solver.solve(device="auto", num_modes=3)
    result.save(OUTPUT / f"{frequency / 1e9:08.3f}GHz.h5")
    neff.append(result.neff)

fdfd.plot_dispersion(frequencies, neff)

"""Frequency sweep of a dielectric slab guide; one saved result per frequency.

Plot the saved sweep later with ``examples/postprocessing/plot_saved_dispersion.py``.
"""

from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import fdfd

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

core = fdfd.Material(name="dielectric core", epsilon=4.0)
frequencies = np.linspace(20e9, 60e9, 50)
neff = []
OUTPUT.mkdir(parents=True, exist_ok=True)
for frequency in tqdm(frequencies, desc="Frequency sweep", unit="frequency"):
    solver = fdfd.ModeSolver1D(frequency=frequency, x_range=(0.0, 10e-3))
    solver.add_layer(x_range=(3e-3, 7e-3), material=core)
    solver.mesh(resolution=200)
    result = solver.solve(device="auto", num_modes=3)
    result.save(OUTPUT / f"{frequency / 1e9:08.3f}GHz.h5")
    neff.append(result.neff)

# Each column is the n-th mode by descending Re(neff); for branches that keep
# their identity through crossings, use fdfd.ModeTracker1D instead.
fdfd.plot_dispersion(frequencies, neff)

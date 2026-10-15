"""Frequency sweep of the leaky-wave cell of 2d_surface_wave_antenna.py.

Plot the saved sweep later with examples/postprocessing/plot_saved_dispersion.py.
"""

from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import fdfd
from fdfd import materials

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

substrate = fdfd.Material(name="antenna substrate", epsilon=10.2)
frequencies = np.linspace(18e9, 22e9, 50)
neff = []
OUTPUT.mkdir(parents=True, exist_ok=True)
for frequency in tqdm(frequencies, desc="Frequency sweep", unit="frequency"):
    solver = fdfd.PeriodicModeSolver2D(frequency=frequency, x_range=(0.0, 10e-3), z_range=(0.0, 8e-3))
    solver.add_rectangle(
        x_range=(0.0, 1.27e-3), z_range=(0.0, 8e-3), material=substrate, name="grounded slab"
    )
    solver.add_rectangle(
        x_range=(1.27e-3, 1.37e-3), z_range=(1e-3, 2e-3), material=materials.PEC, name="PEC patch"
    )
    solver.add_pml(thickness=2.5e-3, sides="x+")
    solver.mesh(resolution=(100, 80))
    result = solver.solve(device="auto", num_modes=4, polarization="TM", guess=0.0, tol=1e-9, ncv=36)
    result.save(OUTPUT / f"{frequency / 1e9:08.3f}GHz.h5")
    neff.append(result.neff)

# Bloch modes are returned nearest to the guess, so their order can change
# between frequencies: plot unconnected samples.
fdfd.plot_dispersion(frequencies, neff, connect=False)

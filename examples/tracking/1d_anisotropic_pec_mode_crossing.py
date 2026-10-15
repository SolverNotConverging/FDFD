"""Track a true TE1/TM1 crossing between PEC plates filled with an anisotropic dielectric.

For this uniform guide

    neff_TE**2 = eps_y - q,    neff_TM**2 = eps_x - (eps_x / eps_z) * q,

with q = (c / (2 w f))**2. The different slopes give a real crossing, which
the tracker follows through without swapping branch identities.
"""

from pathlib import Path

import numpy as np

import fdfd
from fdfd import materials
from fdfd.core.constants import C0

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

width, cells = 22.86e-3, 120
dx = width / cells
epsilon = (2.25, 4.0, 9.0)
q_crossing = (epsilon[1] - epsilon[0]) / (1.0 - epsilon[0] / epsilon[2])
crossing_frequency = C0 / (2 * width) / np.sqrt(q_crossing)
frequencies = crossing_frequency * np.array([0.82, 0.90, 0.96, 0.99, 1.01, 1.04, 1.10, 1.18])

tracker = fdfd.ModeTracker1D(
    frequencies=frequencies,
    x_range=(-dx, width + dx),
    background_material=fdfd.Material(name="diagonal anisotropic dielectric", epsilon=epsilon),
    port=fdfd.PortSpec(boundary="enclosed", name="anisotropic crossing port"),
)
tracker.add_layer(x_range=(-dx, 0.0), material=materials.PEC, name="left plate")
tracker.add_layer(x_range=(width, width + dx), material=materials.PEC, name="right plate")
tracker.mesh(resolution=cells + 2, subpixels=1)
sweep = tracker.solve(
    device="auto",
    num_modes=5,
    config=fdfd.TrackingConfig(
        max_depth=6, overlap_min=0.9, verification_overlap=0.997, verification_beta_tolerance=0.003
    ),
)

print(f"Analytical crossing: {crossing_frequency / 1e9:.4f} GHz")
OUTPUT.mkdir(parents=True, exist_ok=True)
sweep.save(OUTPUT / "tracked_modes.h5")
sweep.show()

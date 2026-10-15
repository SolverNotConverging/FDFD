"""TE modes of a PEC parallel-plate guide tracked through cutoff, with the interactive viewer."""

from pathlib import Path

import numpy as np

import fdfd
from fdfd import materials
from fdfd.core.constants import C0

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

width, cells = 22.86e-3, 96
dx = width / cells
cutoff = C0 / (2 * width)
tracker = fdfd.ModeTracker1D(
    frequencies=cutoff * np.array([0.72, 0.82, 0.92, 1.08, 1.18, 1.28]),
    x_range=(-dx, width + dx),
    port=fdfd.PortSpec(boundary="enclosed", name="parallel-plate port"),
)
tracker.add_layer(x_range=(-dx, 0.0), material=materials.PEC, name="left plate")
tracker.add_layer(x_range=(width, width + dx), material=materials.PEC, name="right plate")
tracker.mesh(resolution=cells + 2, subpixels=1)
sweep = tracker.solve(
    device="auto",
    num_modes=4,
    polarization="TE",
    config=fdfd.TrackingConfig(max_depth=6, verification_overlap=0.995, verification_beta_tolerance=0.004),
)
OUTPUT.mkdir(parents=True, exist_ok=True)
sweep.save(OUTPUT / "tracked_modes.h5")
sweep.show()

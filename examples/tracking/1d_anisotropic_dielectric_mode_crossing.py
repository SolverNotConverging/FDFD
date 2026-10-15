"""Track TE/TM branches of an open anisotropic slab across a wide band."""

from pathlib import Path

import numpy as np

import fdfd

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

width, cells = 30e-3, 120
dx = width / cells
tracker = fdfd.ModeTracker1D(
    frequencies=np.arange(10e9, 50e9, 2e9),
    x_range=(-dx, width + dx),
    port=fdfd.PortSpec(boundary="open", name="anisotropic slab port"),
)
tracker.add_layer(
    x_range=(12e-3, 18e-3),
    material=fdfd.Material(name="diagonal anisotropic dielectric", epsilon=(2.25, 4.0, 9.0)),
)
tracker.mesh(resolution=cells + 2, subpixels=10)
sweep = tracker.solve(
    device="auto",
    num_modes=5,
    config=fdfd.TrackingConfig(
        max_depth=6, overlap_min=0.9, verification_overlap=0.997, verification_beta_tolerance=0.003
    ),
)

OUTPUT.mkdir(parents=True, exist_ok=True)
sweep.save(OUTPUT / "tracked_modes.h5")
sweep.show()

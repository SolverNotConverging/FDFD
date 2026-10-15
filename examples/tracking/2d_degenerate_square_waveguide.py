"""Track the exactly degenerate TE10/TE01 subspace of a square PEC guide.

There is no PML; the two lowest vector modes span the TE10/TE01 eigenspace,
which the tracker carries as a subspace rather than as two arbitrary vectors.
"""

from pathlib import Path

import numpy as np

import fdfd
from fdfd import materials, shapes

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

width, cells = 20e-3, 28
dx = width / cells
tracker = fdfd.ModeTracker2D(
    frequencies=np.linspace(9e9, 20e9, 11),
    x_range=(-dx, width + dx),
    y_range=(-dx, width + dx),
    port=fdfd.PortSpec(boundary="enclosed", name="square PEC port"),
)
wall = shapes.Difference(
    shape=shapes.Rectangle(bounds=((-dx, width + dx), (-dx, width + dx))),
    tool=shapes.Rectangle(bounds=((0.0, width), (0.0, width))),
)
tracker.add_geometry(shape=wall, material=materials.PEC, name="PEC enclosure")
tracker.mesh(resolution=(cells + 2, cells + 2), subpixels=1)
sweep = tracker.solve(
    device="auto",
    num_modes=12,
    config=fdfd.TrackingConfig(
        cluster_gap=2e-4, max_depth=4, verification_overlap=0.995, verification_beta_tolerance=0.01
    ),
)

OUTPUT.mkdir(parents=True, exist_ok=True)
sweep.save(OUTPUT / "tracked_modes.h5")
sweep.show()

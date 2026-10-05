"""Track a true TE/TM crossing in a diagonal-anisotropic parallel-plate guide."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np

from fdfd_mode_tracking import Material, materials, ModeTracker1D, PortSpec, TrackingConfig

OUTPUT = (_ROOT
          / 'fdfd_mode_tracking/outputs/examples/anisotropic_mode_crossing_1d')
C0 = 1 / np.sqrt(8.854187817e-12 * 4e-7 * np.pi)

width, cells = 30e-3, 120
dx = width / cells
epsilon = (2.25, 4.0, 9.0)
medium = Material(name='diagonal anisotropic dielectric', epsilon=epsilon)
frequencies = np.arange(10e9, 50e9, 2e9)

tracker = ModeTracker1D(
    frequencies=frequencies,
    x_range=(-dx, width + dx),
    background_material=materials.vacuum,
    port=PortSpec(boundary='open', name='anisotropic crossing port'),
)

tracker.add_layer(x_range=(12e-3, 18e-3), material=medium)
tracker.mesh(resolution=cells + 2, subpixels=10)
sweep = tracker.solve(
    num_modes=5,
    polarization='both',
    tracking_config=TrackingConfig(
        max_depth=6,
        overlap_min=.9,
        verification_overlap=.997,
        verification_beta_tolerance=.003,
    ),
)

sweep.show(component='E', quantity='magnitude')

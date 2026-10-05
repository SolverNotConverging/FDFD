"""Material-first 2D dielectric-guide sweep and all-candidate mode viewer."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np

from fdfd_mode_tracking import Material, ModeTracker2D, PortSpec, TrackingConfig

OUTPUT = _ROOT / 'fdfd_mode_tracking/outputs/examples/tracked_dielectric_waveguide_2d'

core = Material(name='dielectric core', epsilon=4.)
tracker = ModeTracker2D(
    frequencies=np.linspace(20e9, 50e9, 3),
    x_range=(-5e-3, 5e-3), y_range=(-5e-3, 5e-3),
    port=PortSpec(boundary='open', name='dielectric-guide port'),
)
tracker.add_circle(center=(0., 0.), radius=1.5e-3, material=core, name='core')
tracker.mesh(resolution=(20, 20), subpixels=3)
sweep = tracker.solve(
    num_modes=3,
    tracking_config=TrackingConfig(max_depth=3, max_solves=80,
                                   verification_overlap=.995,
                                   verification_beta_tolerance=.006,
                                   edge_fraction_max=.01),
)
OUTPUT.mkdir(parents=True, exist_ok=True)
sweep.save(OUTPUT/'tracked_modes.h5')
sweep.show(component='E', quantity='magnitude')

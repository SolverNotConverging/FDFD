"""Track a true TE/TM crossing in a diagonal-anisotropic parallel-plate guide."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np

from fdfd_mode_tracking import Material, materials, ModeTracker1D, PortSpec, TrackingConfig

OUTPUT = (_ROOT
          / 'fdfd_mode_tracking/outputs/examples/1d_anisotropic_pec_mode_crossing')
C0 = 1 / np.sqrt(8.854187817e-12 * 4e-7 * np.pi)

width, cells = 22.86e-3, 120
dx = width / cells
epsilon = (2.25, 4.0, 9.0)
medium = Material(name='diagonal anisotropic dielectric', epsilon=epsilon)

# For this uniform guide, TE1 and TM1 obey
#   neff_TE^2 = eps_y - q,
#   neff_TM^2 = eps_x - (eps_x/eps_z) q,
# where q=(c/(2*w*f))^2. Their distinct slopes produce a real crossing.
vacuum_cutoff = C0 / (2 * width)
q_crossing = ((epsilon[1] - epsilon[0])
              / (1.0 - epsilon[0] / epsilon[2]))
crossing_frequency = vacuum_cutoff / np.sqrt(q_crossing)
frequencies = crossing_frequency * np.array([.82, .90, .96, .99, 1.01, 1.04, 1.10, 1.18])

tracker = ModeTracker1D(
    frequencies=frequencies,
    x_range=(-dx, width + dx),
    background_material=medium,
    port=PortSpec(boundary='enclosed', name='anisotropic crossing port'),
)
tracker.add_layer(x_range=(-dx, 0.), material=materials.PEC, name='left plate')
tracker.add_layer(x_range=(width, width + dx), material=materials.PEC, name='right plate')
tracker.mesh(resolution=cells + 2, subpixels=1)
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

sweep.metadata['analytical_crossing_frequency_hz'] = float(crossing_frequency)
OUTPUT.mkdir(parents=True, exist_ok=True)
sweep.save(OUTPUT / 'tracked_modes.h5')
sweep.show(component='E', quantity='magnitude')

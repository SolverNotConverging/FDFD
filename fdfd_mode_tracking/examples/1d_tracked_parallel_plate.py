"""Material-first 1D frequency sweep with the interactive tracked-mode viewer."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np

from fdfd_mode_tracking import materials, ModeTracker1D, PortSpec, TrackingConfig

OUTPUT = _ROOT / 'fdfd_mode_tracking/outputs/examples/1d_tracked_parallel_plate'
C = 1/np.sqrt(8.854187817e-12*4e-7*np.pi)

width, cells = 22.86e-3, 96
dx = width/cells
cutoff = C/(2*width)
tracker = ModeTracker1D(
    frequencies=cutoff*np.array([.72, .82, .92, 1.08, 1.18, 1.28]),
    x_range=(-dx, width+dx),
    port=PortSpec(boundary='enclosed', name='parallel-plate port'),
)
tracker.add_layer(x_range=(-dx, 0.), material=materials.PEC, name='left plate')
tracker.add_layer(x_range=(width, width+dx), material=materials.PEC, name='right plate')
tracker.mesh(resolution=cells+2, subpixels=1)
sweep = tracker.solve(
    num_modes=4, polarization='TE',
    tracking_config=TrackingConfig(max_depth=6, verification_overlap=.995,
                                   verification_beta_tolerance=.004),
)
OUTPUT.mkdir(parents=True, exist_ok=True)
sweep.save(OUTPUT/'tracked_modes.h5')
sweep.show(component='Ey')

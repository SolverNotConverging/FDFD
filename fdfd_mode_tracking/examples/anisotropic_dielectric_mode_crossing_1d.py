"""Track a true TE/TM crossing in a diagonal-anisotropic parallel-plate guide."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "fdfd_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path

import numpy as np

from fdfd_common import Material, materials
from fdfd_mode_tracking import ModeTracker1D, PortSpec, TrackingConfig

OUTPUT = (_ROOT
          / 'outputs/fdfd_mode_tracking/examples/anisotropic_mode_crossing_1d')
C0 = 1 / np.sqrt(8.854187817e-12 * 4e-7 * np.pi)


def main(*, show=True):
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

    if show:
        sweep.show(component='E', quantity='magnitude')
    return sweep


if __name__ == '__main__':
    main()

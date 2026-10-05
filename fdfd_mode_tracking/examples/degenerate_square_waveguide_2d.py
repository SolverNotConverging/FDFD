"""Track the exactly degenerate TE10/TE01 subspace of a square PEC guide."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "fdfd_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path

import numpy as np

from fdfd_common import materials
from fdfd_mode_tracking import ModeTracker2D, PortSpec, TrackingConfig


OUTPUT = (_ROOT
          / 'fdfd_mode_tracking/outputs/examples/degenerate_square_waveguide_2d')


def main(*, show=True):
    width, cells = 20e-3, 28
    dx = width / cells
    tracker = ModeTracker2D(
        frequencies=np.linspace(9e9, 20e9, 11),
        x_range=(-dx, width + dx),
        y_range=(-dx, width + dx),
        port=PortSpec(boundary='enclosed', name='square PEC port'),
    )

    # Four material regions make the physical PEC enclosure explicit. There is
    # no PML: the two lowest vector modes span the TE10/TE01 eigenspace.
    tracker.add_rectangle(x_range=(-dx, 0.), y_range=(-dx, width + dx),
                          material=materials.PEC, name='left wall')
    tracker.add_rectangle(x_range=(width, width + dx), y_range=(-dx, width + dx),
                          material=materials.PEC, name='right wall')
    tracker.add_rectangle(x_range=(0., width), y_range=(-dx, 0.),
                          material=materials.PEC, name='bottom wall')
    tracker.add_rectangle(x_range=(0., width), y_range=(width, width + dx),
                          material=materials.PEC, name='top wall')
    tracker.mesh(resolution=(cells + 2, cells + 2), subpixels=1)
    sweep = tracker.solve(
        num_modes=12,
        tracking_config=TrackingConfig(
            cluster_gap=2e-4,
            max_depth=4,
            verification_overlap=.995,
            verification_beta_tolerance=.01,
        ),
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    sweep.save(OUTPUT / 'tracked_modes.h5')
    if show:
        sweep.show(component='E', quantity='magnitude')
    return sweep


if __name__ == '__main__':
    main()

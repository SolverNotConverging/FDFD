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
          / 'fdfd_mode_tracking/outputs/examples/anisotropic_mode_crossing_1d')
C0 = 1 / np.sqrt(8.854187817e-12 * 4e-7 * np.pi)


def main(*, show=True):
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
    if show:
        sweep.show(component='E', quantity='magnitude')
    return sweep


if __name__ == '__main__':
    main()

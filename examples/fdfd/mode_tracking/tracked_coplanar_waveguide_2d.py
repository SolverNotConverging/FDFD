"""Open coplanar waveguide: frequency sweep and all-mode interactive viewer.

The finite substrate and coplanar grounds have vacuum on all four sides. There
is no backing ground or housing. Artificial outer boundaries are checked by
domain enlargement, not declared physical walls. This is a finite-width board,
not an infinite-substrate CPW. The tracker does not preselect a polarization.
All dimensions are in metres and propagation is along z. No PML is used.
"""
from pathlib import Path

import numpy as np

from cem_common import Material, materials
from fdfd_mode_tracking import ModeTracker2D, PortSpec, TrackingConfig


OUTPUT = (Path(__file__).resolve().parents[3]
          / 'outputs/examples/fdfd/mode_tracking/tracked_coplanar_waveguide_2d')


def build_tracker(*, frequencies=None, air_padding=8e-3, cell_size=.1e-3):
    """Leave air_padding on EVERY side of the board/metal bounding box.

    Defaults give a 28 by 17.4 mm domain and a 280 by 174 grid: six cells
    across each slot and two through the metal. Verification additionally
    doubles the mesh resolution and enlarges the exterior twice, preserving
    the physical board and conductor geometry. Larger domains/finer grids are
    expensive with the uniform-grid eigensolver; do not bypass verification.
    """
    if not np.isfinite(air_padding) or air_padding <= 0:
        raise ValueError('air_padding must be positive and finite.')
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError('cell_size must be positive and finite.')
    if frequencies is None:
        frequencies = np.linspace(6e9, 30e9, 7)
    half_width = 6e-3
    substrate_height = 1.2e-3
    strip_width, gap, metal_thickness = 1.2e-3, .6e-3, .2e-3
    x_range = (-half_width-air_padding, half_width+air_padding)
    y_range = (-air_padding, substrate_height+metal_thickness+air_padding)
    tracker = ModeTracker2D(
        frequencies=frequencies,
        x_range=x_range, y_range=y_range,
        port=PortSpec(boundary='open', name='open coplanar-waveguide port'),
    )
    tracker.add_rectangle(
        x_range=(-half_width, half_width), y_range=(0., substrate_height),
        material=Material(name='substrate', epsilon=4.), name='substrate',
    )
    metal_y = (substrate_height, substrate_height+metal_thickness)
    tracker.add_rectangle(x_range=(-strip_width/2, strip_width/2),
                          y_range=metal_y, material=materials.PEC, name='signal strip')
    tracker.add_rectangle(x_range=(-half_width, -strip_width/2-gap),
                          y_range=metal_y, material=materials.PEC, name='left ground')
    tracker.add_rectangle(x_range=(strip_width/2+gap, half_width),
                          y_range=metal_y, material=materials.PEC, name='right ground')

    # Roundoff guard avoids turning an exactly integral cell count into N+1.
    resolution = tuple(int(np.ceil((hi-lo)/cell_size - 1e-10))
                       for lo, hi in (x_range, y_range))
    tracker.mesh(resolution=resolution, subpixels=1)
    return tracker


def main(*, show=True, output=OUTPUT, frequencies=None,
         air_padding=4e-3, cell_size=.2e-3):
    tracker = build_tracker(frequencies=frequencies, air_padding=air_padding,
                            cell_size=cell_size)
    sweep = tracker.solve(
        num_modes=6,  # All returned candidates are tracked; index guess is automatic.
        tracking_config=TrackingConfig(
            max_depth=3, max_solves=160,
            verification_overlap=.9,
            verification_beta_tolerance=.01,
        ),
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sweep.save(output / 'tracked_modes.h5')
    if show:
        sweep.show(component='E', quantity='magnitude')
    return sweep


if __name__ == '__main__':
    main()

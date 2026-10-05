"""Open coplanar waveguide: frequency sweep and all-mode interactive viewer.

The finite substrate and coplanar grounds have vacuum on all four sides. There
is no backing ground or housing. Artificial outer boundaries are checked by
domain enlargement, not declared physical walls. This is a finite-width board,
not an infinite-substrate CPW. The tracker does not preselect a polarization.
All dimensions are in metres and propagation is along z. No PML is used.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np

from fdfd_mode_tracking import Material, materials, ModeTracker2D, PortSpec, TrackingConfig

OUTPUT = (_ROOT
          / 'fdfd_mode_tracking/outputs/examples/tracked_coplanar_waveguide_2d')

frequencies = np.linspace(6e9, 30e9, 7)
air_padding = 4e-3
cell_size = .2e-3

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

sweep = tracker.solve(
    num_modes=6,  # All returned candidates are tracked; index guess is automatic.
    tracking_config=TrackingConfig(
        max_depth=3, max_solves=160,
        verification_overlap=.9,
        verification_beta_tolerance=.01,
    ),
)
OUTPUT.mkdir(parents=True, exist_ok=True)
sweep.save(OUTPUT / 'tracked_modes.h5')
sweep.show(component='E', quantity='magnitude')

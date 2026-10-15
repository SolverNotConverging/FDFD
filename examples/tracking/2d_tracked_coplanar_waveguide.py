"""Open coplanar waveguide: frequency sweep with the all-candidate viewer.

The finite substrate and coplanar grounds have vacuum on all four sides; there
is no backing ground or housing. The artificial outer boundaries are checked
by domain enlargement rather than declared physical walls, and no PML is used.
All dimensions are in metres; propagation is along z.
"""

from pathlib import Path

import numpy as np

import fdfd
from fdfd import materials

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

frequencies = np.linspace(6e9, 30e9, 7)
air_padding = 4e-3
cell_size = 0.2e-3

half_width = 6e-3
substrate_height = 1.2e-3
strip_width, gap, metal_thickness = 1.2e-3, 0.6e-3, 0.2e-3
x_range = (-half_width - air_padding, half_width + air_padding)
y_range = (-air_padding, substrate_height + metal_thickness + air_padding)
tracker = fdfd.ModeTracker2D(
    frequencies=frequencies,
    x_range=x_range,
    y_range=y_range,
    port=fdfd.PortSpec(boundary="open", name="open coplanar-waveguide port"),
)
tracker.add_rectangle(
    x_range=(-half_width, half_width),
    y_range=(0.0, substrate_height),
    material=fdfd.Material(name="substrate", epsilon=4.0),
    name="substrate",
)
metal_y = (substrate_height, substrate_height + metal_thickness)
tracker.add_rectangle(
    x_range=(-strip_width / 2, strip_width / 2), y_range=metal_y, material=materials.PEC, name="signal strip"
)
tracker.add_rectangle(
    x_range=(-half_width, -strip_width / 2 - gap), y_range=metal_y, material=materials.PEC, name="left ground"
)
tracker.add_rectangle(
    x_range=(strip_width / 2 + gap, half_width), y_range=metal_y, material=materials.PEC, name="right ground"
)
tracker.mesh(max_element_size=cell_size, subpixels=1)

sweep = tracker.solve(
    device="auto",
    num_modes=6,
    config=fdfd.TrackingConfig(
        max_depth=3, max_solves=160, verification_overlap=0.9, verification_beta_tolerance=0.01
    ),
)
OUTPUT.mkdir(parents=True, exist_ok=True)
sweep.save(OUTPUT / "tracked_modes.h5")
sweep.show()

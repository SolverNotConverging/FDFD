"""Track TE1 through cutoff between physical PEC plates and export port profiles.

This uses the lower-level :func:`fdfd.tracking.track_modes`, which takes a
factory building the solver at each frequency and verification grid.
"""

import csv
from pathlib import Path

import numpy as np

import fdfd
from fdfd import materials
from fdfd.core.constants import C0
from fdfd.tracking import track_modes

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

width, cells = 22.86e-3, 128
dx = width / cells


def solver_for_frequency(frequency, verification):
    """Called for every frequency and every refined verification grid."""
    solver = fdfd.ModeSolver1D(frequency=frequency, x_range=(-dx, width + dx))
    solver.add_layer(x_range=(-dx, 0.0), material=materials.PEC, name="left plate")
    solver.add_layer(x_range=(width, width + dx), material=materials.PEC, name="right plate")
    solver.mesh(resolution=(cells + 2) * verification.mesh_factor, subpixels=1)
    return solver


cutoff = C0 / (2 * width)
sweep = track_modes(
    solver_for_frequency,
    cutoff * np.array([0.75, 0.85, 1.15, 1.25]),
    device="auto",
    port=fdfd.PortSpec(boundary="enclosed"),
    config=fdfd.TrackingConfig(
        num_candidates=2, max_candidates=4, polarization="TE", guess=-0.8j, max_depth=6
    ),
)
OUTPUT.mkdir(parents=True, exist_ok=True)
sweep.save(OUTPUT / "tracked_modes.h5")
with (OUTPUT / "port_modes.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(
        (
            "frequency_hz",
            "beta_real",
            "beta_imag",
            "real_power_w_per_m",
            "reactive_power_var_per_m",
            "propagation",
        )
    )
    for mode in sweep.export():
        writer.writerow(
            (
                mode.frequency,
                mode.beta.real,
                mode.beta.imag,
                mode.complex_power.real,
                mode.complex_power.imag,
                mode.metadata["propagation"],
            )
        )
print(f"Tracked {len(sweep.samples)} samples using {sweep.metadata['eigensolves']} solves.")
print(f"Cutoff/unresolved intervals (Hz): {sweep.unresolved_intervals}")
print(f"Port profiles: {OUTPUT}")

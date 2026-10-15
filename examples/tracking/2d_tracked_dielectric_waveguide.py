"""Modes of an open circular dielectric rod tracked across frequency."""

from pathlib import Path

import numpy as np

import fdfd

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem

tracker = fdfd.ModeTracker2D(
    frequencies=np.linspace(20e9, 50e9, 3),
    x_range=(-5e-3, 5e-3),
    y_range=(-5e-3, 5e-3),
    port=fdfd.PortSpec(boundary="open", name="dielectric-guide port"),
)
tracker.add_circle(
    center=(0.0, 0.0), radius=1.5e-3, material=fdfd.Material(name="dielectric core", epsilon=4.0), name="core"
)
tracker.mesh(resolution=(20, 20), subpixels=3)
sweep = tracker.solve(
    device="auto",
    num_modes=3,
    config=fdfd.TrackingConfig(
        max_depth=3,
        max_solves=80,
        verification_overlap=0.995,
        verification_beta_tolerance=0.006,
        edge_fraction_max=0.01,
    ),
)
OUTPUT.mkdir(parents=True, exist_ok=True)
sweep.save(OUTPUT / "tracked_modes.h5")
sweep.show()

"""Sweep a dielectric guide, reusing the same material at each frequency."""
import csv
from pathlib import Path
import numpy as np
from cem_common import Material
from fdfd_waveguide_modes import ModeSolver2D

OUTPUT = Path(__file__).resolve().parents[3] / "outputs/examples/fdfd/waveguide_modes/dielectric_waveguide_2d_dispersion"


def main():
    core = Material(name="dielectric core", epsilon=4.)
    rows = []
    for frequency in np.linspace(20e9, 60e9, 5):
        solver = ModeSolver2D(frequency=frequency, x_range=.01, y_range=.01)
        solver.add_circle(center=(.005, .005), radius=.002, material=core)
        solver.mesh(resolution=(40, 40))
        result = solver.solve(num_modes=3, neff_guess=1.8)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        result.save(OUTPUT / f"modes_{frequency/1e9:.0f}GHz.h5")
        for mode, neff in enumerate(result.neff):
            rows.append(dict(frequency_hz=frequency, mode=mode, neff_real=neff.real, neff_imag=neff.imag))
        print(f"{frequency/1e9:.0f} GHz: {result.neff}")
    with (OUTPUT / "dispersion.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    main()

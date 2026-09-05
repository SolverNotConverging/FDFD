"""Explicit frequency sweep of the 2D periodic guide."""
import csv
import importlib.util
from pathlib import Path
import numpy as np

OUTPUT = Path(__file__).resolve().parents[3] / "outputs/examples/fdfd/periodic_modes/surface_wave_antenna_2d_dispersion"


def main():
    spec = importlib.util.spec_from_file_location("guide_example", Path(__file__).with_name("surface_wave_antenna_2d.py"))
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    rows=[]
    for frequency in np.linspace(25e9,35e9,3):
        solver=example.build_solver(frequency=frequency)
        solver.mesh(resolution=(24, 16))
        result=solver.solve(num_modes=2,neff_guess=1.5,eigensolver="eigs")
        OUTPUT.mkdir(parents=True,exist_ok=True)
        result.save(OUTPUT / f"modes_{frequency/1e9:.0f}GHz.h5")
        for mode,neff in enumerate(result.neff):
            rows.append(dict(frequency_hz=frequency,mode=mode,neff_real=neff.real,neff_imag=neff.imag))
    with (OUTPUT / "dispersion.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return rows


if __name__ == "__main__":
    main()

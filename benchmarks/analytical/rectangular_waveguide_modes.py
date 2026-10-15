"""Grid convergence of the TE10 mode of a rectangular PEC waveguide.

The exact solution is beta = sqrt(k0**2 - (pi / width)**2). Frequency,
dimensions and mode are fixed while the grid is refined.

    python benchmarks/analytical/rectangular_waveguide_modes.py --check
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter

import fdfd
from fdfd import materials, shapes
from fdfd.core.constants import C0

DEFAULT_OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem


def compare(levels=(8, 12, 16)):
    width, height, frequency = 22.86e-3, 10.16e-3, 10e9
    k0 = 2 * np.pi * frequency / C0
    exact_beta = np.sqrt(k0**2 - (np.pi / width) ** 2)
    exact_neff = exact_beta / k0
    rows = []
    for cells_x in levels:
        cells_y = max(4, round(cells_x * height / width))
        start = perf_counter()
        # One solid cell on each side puts the PEC inner faces exactly at the
        # requested clear width and height for every resolution.
        dx, dy = width / cells_x, height / cells_y
        bounds = ((-dx, width + dx), (-dy, height + dy))
        solver = fdfd.ModeSolver2D(frequency=frequency, x_range=bounds[0], y_range=bounds[1])
        wall = shapes.Difference(
            shape=shapes.Rectangle(bounds=bounds), tool=shapes.Rectangle(bounds=((0.0, width), (0.0, height)))
        )
        solver.add_geometry(shape=wall, material=materials.PEC, name="wall")
        solver.mesh(resolution=(cells_x + 2, cells_y + 2), subpixels=1)
        result = solver.solve(num_modes=1, guess=exact_neff)
        neff = result.neff[0]
        row = dict(
            cells_x=cells_x,
            cells_y=cells_y,
            nominal_h_m=max(dx, dy),
            frequency_hz=frequency,
            exact_neff=exact_neff,
            neff_real=neff.real,
            neff_imag=neff.imag,
            relative_neff_error=abs(neff - exact_neff) / abs(exact_neff),
            backward_error=float(result.info.residuals[0]),
            elapsed_seconds=perf_counter() - start,
        )
        rows.append(row)
        print(f"nx={cells_x:3}: neff={neff:.8g}; relative error={row['relative_neff_error']:.3e}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="require a finite passive mode with <1%% error on the finest grid",
    )
    args = parser.parse_args()
    if min(args.levels) < 4:
        parser.error("Each level must have at least four cells across the width.")
    rows = compare(sorted(set(args.levels)))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    figure = Figure(figsize=(7, 4.5))
    ax = figure.subplots()
    ax.semilogy(
        [r["nominal_h_m"] * 1e3 for r in rows], [max(r["relative_neff_error"], 1e-16) for r in rows], "o-"
    )
    ax.set_xticks(sorted({r["nominal_h_m"] * 1e3 for r in rows}))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set(
        xlabel="Nominal mesh spacing (mm)",
        ylabel="Relative effective-index error",
        title="Rectangular PEC waveguide: TE10 at 10 GHz",
    )
    ax.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output / "convergence.png", dpi=160)
    if args.check:
        finest = rows[-1]
        if not (
            np.isfinite(finest["relative_neff_error"])
            and finest["relative_neff_error"] < 0.01
            and abs(finest["neff_imag"]) < 1e-8
        ):
            raise SystemExit(f"TE10 analytical check failed: {finest}")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()

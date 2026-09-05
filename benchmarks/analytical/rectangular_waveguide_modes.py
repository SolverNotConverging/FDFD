"""Compare FEM and FDFD TE10 modes against a rectangular PEC waveguide.

The exact vacuum solution is beta = sqrt(k0**2 - (pi / width)**2).
Refine spatial resolution while keeping frequency, dimensions, and the mode fixed.
"""
from cem_common import materials, shapes
import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from scipy.constants import c
from fdfd_waveguide_modes import ModeSolver2D as FDFDModeSolver2D
from fem_waveguide_modes import ModeSolver2D as FEMModeSolver2D

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'outputs/benchmarks/analytical/rectangular_waveguide_modes'


def compare(levels=(8, 12, 16)):
    width, height, frequency = 22.86e-3, 10.16e-3, 10e9
    k0 = 2 * np.pi * frequency / c
    exact_beta = np.sqrt(k0**2 - (np.pi / width)**2)
    exact_neff = exact_beta / k0
    rows = []
    for cells_x in levels:
        cells_y = max(4, round(cells_x * height / width))
        for method in ('FDFD', 'FEM'):
            start = perf_counter()
            if method == 'FDFD':
                # One solid cell on each side places the PEC inner faces at
                # exactly the requested clear width/height for every resolution.
                nx, ny = cells_x + 2, cells_y + 2
                dx, dy = width / cells_x, height / cells_y
                total_x, total_y = nx * dx, ny * dy
                solver = FDFDModeSolver2D(
                    frequency=frequency, x_range=total_x,
                    y_range=total_y,
                )
                wall = shapes.Difference(
                    shape=shapes.Rectangle(bounds=((0., total_x), (0., total_y))),
                    tool=shapes.Rectangle(bounds=((dx, (nx-1)*dx), (dy, (ny-1)*dy))),
                )
                solver.add_geometry(shape=wall, material=materials.PEC, name='wall')
                solver.mesh(resolution=(nx, ny))
                result = solver.solve(num_modes=1, neff_guess=exact_neff)
                neff = result.neff[0]
                elements = cells_x * cells_y
                algebraic_residual = ''  # Not exposed by this FDFD solve API.
            else:
                solver = FEMModeSolver2D(frequency=frequency, x_range=width, y_range=height, boundary=materials.PEC)
                mesh = solver.mesh(resolution=(cells_x + 1, cells_y + 1), element_order=1)
                mode = solver.solve(
                    num_modes=1, neff_guess=exact_neff, max_refinements=0,
                    dense_linearization_limit=4,
                )[0]
                neff, elements, algebraic_residual = mode.neff, len(mesh.elements), mode.residual
            row = dict(
                method=method, cells_x=cells_x, cells_y=cells_y, elements=elements,
                nominal_h_m=max(width / cells_x, height / cells_y), frequency_hz=frequency,
                exact_neff=exact_neff, neff_real=neff.real, neff_imag=neff.imag,
                exact_beta_rad_per_m=exact_beta, beta_real_rad_per_m=k0 * neff.real,
                relative_neff_error=abs(neff - exact_neff) / abs(exact_neff),
                algebraic_residual=algebraic_residual, elapsed_seconds=perf_counter() - start,
            )
            rows.append(row)
            print(f'{method:4} nx={cells_x:2}: neff={neff:.8g}; relative error={row["relative_neff_error"]:.3e}', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', type=int, nargs='+', default=[8, 12, 16])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Require finite passive modes and <1%% error on the finest grid.')
    args = parser.parse_args()
    if min(args.levels) < 4:
        parser.error('Each level must have at least four cells across the width.')
    rows = compare(sorted(set(args.levels)))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    figure = Figure(figsize=(7, 4.5))
    ax = figure.subplots()
    for method in ('FDFD', 'FEM'):
        selected = [row for row in rows if row['method'] == method]
        ax.semilogy([r['nominal_h_m'] * 1e3 for r in selected], [max(r['relative_neff_error'], 1e-16) for r in selected], 'o-', label=method)
    ax.set_xticks(sorted({r['nominal_h_m'] * 1e3 for r in rows}))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set(xlabel='Nominal mesh spacing (mm)', ylabel='Relative effective-index error', title='Rectangular PEC waveguide: TE10 at 10 GHz')
    ax.grid(True, which='both', alpha=.3)
    ax.legend()
    figure.tight_layout()
    figure.savefig(args.output / 'convergence.png', dpi=160)
    if args.check:
        for method in ('FDFD', 'FEM'):
            finest = [r for r in rows if r['method'] == method][-1]
            if not (np.isfinite(finest['relative_neff_error']) and finest['relative_neff_error'] < .01 and abs(finest['neff_imag']) < 1e-8):
                raise SystemExit(f'{method} failed the TE10 analytical check: {finest}')
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()

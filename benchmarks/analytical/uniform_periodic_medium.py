"""Compare lossless and passive lossy FEM periodic TEM modes with theory.

For a homogeneous medium, neff = sqrt(epsilon_r * mu_r). Under exp(+i omega t),
passive epsilon has negative imaginary part and alpha = -Im(k0 * neff).
"""
from cem_common import Material, materials
import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.constants import c
from fem_periodic_modes import PeriodicModeSolver2D

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'outputs/benchmarks/analytical/uniform_periodic_medium'


def compare():
    rows = []
    for name, epsilon in (('lossless', 2.25), ('lossy', 2.25 - .02j)):
        medium = Material(name=f"{name} dielectric", epsilon=epsilon)
        exact = np.sqrt(complex(epsilon))
        for frequency in (8e9, 10e9, 12e9):
            solver = PeriodicModeSolver2D(frequency=frequency, x_range=0.004, z_range=0.005, polarization='TM', boundary=materials.PEC, background_material=medium)
            solver.mesh(max_element_size=.001)
            mode = solver.solve(
                num_modes=1, neff_guess=exact, max_refinements=0,
                direction='forward', eigensolver='dense',
            )[0]
            k0 = 2 * np.pi * frequency / c
            row = dict(
                material=name, frequency_hz=frequency,
                exact_neff_real=exact.real, exact_neff_imag=exact.imag,
                neff_real=mode.neff.real, neff_imag=mode.neff.imag,
                exact_beta_rad_per_m=k0 * exact.real, beta_rad_per_m=k0 * mode.neff.real,
                exact_alpha_np_per_m=-k0 * exact.imag, alpha_np_per_m=-k0 * mode.neff.imag,
                relative_neff_error=abs(mode.neff - exact) / abs(exact),
                algebraic_residual=mode.residual,
            )
            rows.append(row)
            print(f'{name:8} {frequency / 1e9:g} GHz: neff={mode.neff:.9g}; error={row["relative_neff_error"]:.3e}', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Require <0.05%% index error, passive attenuation, and <1%% lossy-attenuation error.')
    args = parser.parse_args()
    rows = compare()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    figure = Figure(figsize=(10, 4))
    beta_ax, alpha_ax = figure.subplots(1, 2)
    for material in ('lossless', 'lossy'):
        subset = [r for r in rows if r['material'] == material]
        frequencies = [r['frequency_hz'] / 1e9 for r in subset]
        for ax, value, theory in ((beta_ax, 'beta_rad_per_m', 'exact_beta_rad_per_m'), (alpha_ax, 'alpha_np_per_m', 'exact_alpha_np_per_m')):
            ax.plot(frequencies, [r[theory] for r in subset], '-', label=f'{material}: analytical')
            ax.plot(frequencies, [r[value] for r in subset], 'o', label=f'{material}: FEM')
            ax.set_xlabel('Frequency (GHz)')
            ax.grid(True, alpha=.3)
            ax.legend(fontsize=8)
    beta_ax.set(ylabel='Phase constant (rad/m)', title='Uniform periodic TEM mode')
    alpha_ax.set(ylabel='Attenuation (Np/m)', title='Passive attenuation with positive-time phasors')
    figure.tight_layout()
    figure.savefig(args.output / 'comparison.png', dpi=160)
    if args.check:
        for row in rows:
            if not (np.isfinite(row['relative_neff_error']) and row['relative_neff_error'] < 5e-4 and row['alpha_np_per_m'] >= -1e-8):
                raise SystemExit(f'Periodic analytical check failed: {row}')
            if row['material'] == 'lossy' and abs(row['alpha_np_per_m'] / row['exact_alpha_np_per_m'] - 1) >= .01:
                raise SystemExit(f'Lossy attenuation differs from theory by at least 1%: {row}')
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()

"""Grid convergence against the validated FEM grounded-slab periodic cell.

Stored FEM references use first-order triangles: 50e-6 m maximum edge at
20e9 Hz for the antenna, and 100e-6 m for the other comparisons. All use
max_refinements=0, TM, PEC walls and x+ PML. No FEM installation is
required to run the FDFD comparison. See the linked family documentation.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import numpy as np
from matplotlib.figure import Figure
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm
from fdfd_periodic_modes import Material, PeriodicModeSolver2D, materials

FEM_REFERENCE = np.array([
    -.01454590826862274-.09323259689950363j,
    .01454590826862245+.09323259689950367j,
    .5792212652919826-.5252337359649764j,
    -.5792212652919808+.5252337359649766j,
])
FEM_REFERENCES = {
    'antenna': {
        18e9: np.array([-.41798552756884316-.02520160351518428j,
                       .4179855275688467+.025201603515184426j,
                       .9160543684780482-.07654551260303112j,
                       -.9160543684780493+.07654551260303151j]),
        20e9: FEM_REFERENCE,
        22e9: np.array([-.4316608264408845+.012947476542689419j,
                       .43166082644088466-.012947476542689105j,
                       -.740619221038214-.024576247156232158j,
                       .7406192210382201+.024576247156232633j]),
    },
    'slab': {
        20e9: np.array([-.06546748694914711-7.102375504829061e-6j,
                       .06546748694914724+7.1023755047032695e-6j,
                       -.5576661675321186+.5883053964858241j,
                       .5576661675321186-.5883053964858249j]),
    },
}
DEFAULT_OUTPUT = ROOT/'fdfd_periodic_modes/outputs/benchmarks/leaky_wave_fem_reference'


def compare(resolutions, frequencies=(20e9,), case="antenna"):
    rows = []
    jobs = [(frequency, nx, nz) for frequency in frequencies for nx, nz in resolutions]
    for frequency, nx, nz in tqdm(jobs, desc='FEM comparison', unit='solve'):
        reference = FEM_REFERENCES[case][frequency]
        solver = PeriodicModeSolver2D(frequency=frequency, x_range=(0., 10e-3),
            z_range=(0., 8e-3), polarization='TM', boundary=materials.PEC)
        solver.add_rectangle(x_range=(0., 1.27e-3), z_range=(0., 8e-3),
            material=Material(name='antenna substrate', epsilon=10.2))
        if case == 'antenna':
            solver.add_rectangle(x_range=(1.27e-3, 1.32e-3), z_range=(1e-3, 2e-3), material=materials.PEC)
        solver.add_pml(thickness=2.5e-3, direction='x+', order=3, sigma_max=5.)
        solver.mesh(resolution=(nx, nz))
        result = solver.solve(num_modes=4, neff_guess=0., eigensolver_tolerance=1e-9, ncv=36)
        reference_indices, calculated_indices = linear_sum_assignment(
            np.abs(reference[:, None]-result.neff[None, :]))
        for reference_index, calculated_index in zip(reference_indices, calculated_indices):
            expected, value = reference[reference_index], result.neff[calculated_index]
            rows.append(dict(case=case, frequency_hz=frequency, nx=nx, nz=nz, dx_m=10e-3/nx, dz_m=8e-3/nz,
                mode=reference_index+1, neff_real=value.real, neff_imag=value.imag,
                fem_neff_real=expected.real, fem_neff_imag=expected.imag,
                relative_complex_error=abs(value-expected)/abs(expected),
                maxwell_residual=result.solve_info['residuals'][calculated_index]))
        tqdm.write(f'{case} at {frequency/1e9:g} GHz, {nx} x {nz}: maximum complex-neff error {max(r["relative_complex_error"] for r in rows[-4:]):.2%}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grids', nargs='+', default=['1000x320'])
    parser.add_argument('--frequencies', nargs='+', type=float, default=[18e9, 20e9, 22e9])
    parser.add_argument('--case', choices=tuple(FEM_REFERENCES), default='antenna')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Require <3%% complex-neff error on the finest grid at every frequency.')
    args = parser.parse_args()
    resolutions = [tuple(map(int, grid.lower().split('x'))) for grid in args.grids]
    if any(len(pair) != 2 or min(pair) < 2 for pair in resolutions):
        parser.error('Use grid sizes such as 1000x320.')
    if any(f not in FEM_REFERENCES[args.case] for f in args.frequencies):
        parser.error('No stored FEM reference for this case/frequency; slab supports 20e9.')
    rows = compare(resolutions, args.frequencies, args.case)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    figure = Figure(figsize=(8, 5))
    axis = figure.subplots()
    for frequency in args.frequencies:
        for mode in range(1, 5):
            selected = [row for row in rows if row['mode'] == mode and row['frequency_hz'] == frequency]
            axis.loglog([r['dx_m']*1e6 for r in selected], [r['relative_complex_error'] for r in selected], 'o-', label=f'{frequency/1e9:g} GHz, mode {mode}')
    axis.set(xlabel='x cell width (um)', ylabel='Relative complex-neff error against FEM', title=f'Periodic {args.case}: FDFD vs FEM')
    axis.grid(True, which='both', alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output/'convergence.png', dpi=160)
    if args.check:
        finest = max(resolutions, key=lambda pair: pair[0]*pair[1])
        selected = [r for r in rows if (r['nx'], r['nz']) == finest]
        if any(r['relative_complex_error'] > .03 or r['maxwell_residual'] > 1e-6 for r in selected):
            raise SystemExit('FDFD did not meet the FEM comparison tolerance; refine the grid.')
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()

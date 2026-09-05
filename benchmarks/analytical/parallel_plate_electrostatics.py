"""Compare FEM capacitors and a Poisson problem against exact 1D solutions.

No fringing is present: the 2D capacitor's top/bottom boundaries are insulating.
The charged problem compares the interpolated P1 field, not only nodal values.
"""
from cem_common import Material, shapes
import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.constants import epsilon_0
from scipy.integrate import trapezoid
from fem_electrostatics import ElectrostaticSolver

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'outputs/benchmarks/analytical/parallel_plate_electrostatics'


def compare(levels=(8, 16, 32)):
    length, height, epsilon, voltage = .01, .005, 2., 10.
    dielectric = Material(name="uniform dielectric", epsilon=epsilon)
    density = 1e-6  # Free volume charge in C/m^3.
    rows, profiles = [], []
    for level in levels:
        for dimension in (1, 2):
            solver = ElectrostaticSolver(dim=dimension, x_range=length, y_range=height if dimension == 2 else None, outer_potential=None, background_material=dielectric)
            solver.set_potential(potential=0.0, name='ground', geometry='left')
            solver.set_potential(potential=voltage, name='drive', geometry='right')
            solver.mesh(max_element_size=length / level)
            result = solver.solve(max_refinements=0)
            exact_potential = voltage * result.coordinates[:, 0] / length
            # 1D quantities are per plate area; 2D quantities are per z length.
            transverse_measure = 1. if dimension == 1 else height
            exact_capacitance = epsilon_0 * epsilon * transverse_measure / length
            capacitance = result.conductor_charge('drive') / voltage
            exact_energy = .5 * exact_capacitance * voltage**2
            rows.append(dict(
                case=f'capacitor_{dimension}d', cells_across=level,
                nominal_h_m=length / level, nodes=len(result.coordinates),
                potential_relative_error=np.max(np.abs(result.potential - exact_potential)) / voltage,
                capacitance=capacitance, exact_capacitance=exact_capacitance,
                capacitance_units='F/m^2' if dimension == 1 else 'F/m',
                relative_capacitance_error=abs(capacitance / exact_capacitance - 1),
                relative_energy_error=abs(result.energy / exact_energy - 1),
            ))

        solver = ElectrostaticSolver(dim=1, x_range=length, outer_potential=0.0, background_material=dielectric)
        solver.add_charge_density(density=density, geometry=shapes.Interval(bounds=(0.0, length)))
        solver.mesh(max_element_size=length / level)
        result = solver.solve(max_refinements=0)
        x = np.linspace(0., length, 4001)
        exact = density * x * (length - x) / (2 * epsilon_0 * epsilon)
        order = np.argsort(result.coordinates[:, 0])
        numerical = np.interp(x, result.coordinates[order, 0], result.potential[order])
        relative_l2 = np.sqrt(trapezoid((numerical - exact)**2, x) / trapezoid(exact**2, x))
        rows.append(dict(
            case='space_charge_1d', cells_across=level, nominal_h_m=length / level,
            nodes=len(result.coordinates), potential_relative_error=relative_l2,
            capacitance='', exact_capacitance='', capacitance_units='',
            relative_capacitance_error='', relative_energy_error='',
        ))
        profiles = [dict(x_m=xi, exact_potential_v=vi, fem_potential_v=ui) for xi, vi, ui in zip(x, exact, numerical)]
        print(f'Target L/h={level:2}: charged-potential relative L2 error={relative_l2:.3e}', flush=True)
    return rows, profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Check capacitor identities and <1%% charged-potential error at the finest level.')
    args = parser.parse_args()
    if min(args.levels) < 2:
        parser.error('Use at least two cells across the plates.')
    rows, profiles = compare(sorted(set(args.levels)))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (('comparison.csv', rows), ('potential.csv', profiles)):
        with (args.output / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    figure = Figure(figsize=(10, 4))
    profile_ax, error_ax = figure.subplots(1, 2)
    profile_ax.plot([r['x_m'] for r in profiles], [r['exact_potential_v'] for r in profiles], label='Analytical')
    profile_ax.plot([r['x_m'] for r in profiles], [r['fem_potential_v'] for r in profiles], '--', label='FEM (finest mesh)')
    profile_ax.set(xlabel='x (m)', ylabel='Potential (V)', title='Grounded plates with uniform space charge')
    profile_ax.legend()
    charged = [r for r in rows if r['case'] == 'space_charge_1d']
    error_ax.loglog([r['nominal_h_m'] for r in charged], [r['potential_relative_error'] for r in charged], 'o-')
    error_ax.set(xlabel='Requested maximum element size (m)', ylabel='Relative L2 potential error', title='Piecewise-linear potential convergence')
    error_ax.grid(True, which='both', alpha=.3)
    figure.tight_layout()
    figure.savefig(args.output / 'comparison.png', dpi=160)
    if args.check:
        for row in rows:
            if row['case'].startswith('capacitor') and not (row['potential_relative_error'] < 1e-8 and row['relative_capacitance_error'] < 1e-7 and row['relative_energy_error'] < 1e-7):
                raise SystemExit(f'Capacitor analytical check failed: {row}')
        if not np.isfinite(charged[-1]['potential_relative_error']) or charged[-1]['potential_relative_error'] >= .01:
            raise SystemExit('Charged-potential error exceeds 1% at the finest level.')
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()

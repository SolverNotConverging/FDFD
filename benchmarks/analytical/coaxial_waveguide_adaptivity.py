"""Compare adaptive FEM coaxial TEM fields with the analytical 1/r solution.

The present backend responds to its Maxwell jump estimator by globally remeshing
at h/1.5. This measures that actual adaptive solve workflow, including its budget
and stopping diagnostics; it does not claim local element marking.
"""
from cem_common import materials, shapes
import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.figure import Figure
from scipy.constants import epsilon_0, mu_0
from skfem.quadrature import get_quadrature_tri

from fem_waveguide_modes import ModeSolver2D

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'outputs/benchmarks/analytical/coaxial_waveguide_adaptivity'
INNER_RADIUS = 1e-3
OUTER_RADIUS = 4e-3
FREQUENCY = 1e9
QUADRATURE_ORDER = 4


def make_solver():
    solver = ModeSolver2D(frequency=FREQUENCY, x_range=(-0.0045, 0.0045), y_range=(-0.0045, 0.0045), boundary=materials.PEC)
    solver.add_geometry(name='inner_conductor', shape=shapes.Circle(center=(0, 0), radius=INNER_RADIUS), material=materials.PEC)
    # The outer annulus covers all model corners. Only its intersection with
    # the rectangular bounds is used, leaving exactly the coaxial dielectric.
    solver.add_geometry(clip=True, name='outer_conductor', shape=shapes.Annulus(center=(0, 0), outer_radius=0.007, inner_radius=OUTER_RADIUS), material=materials.PEC)
    solver.mesh(
        max_element_size=1e-3, boundary_refinement=None,
        element_order=1, quadrature_order=QUADRATURE_ORDER,
    )
    return solver


def field_errors(mode):
    """Integrate full-vector errors on the numerical triangles, phase aligned.

    Both solutions carry 1 W of forward time-average power. Align only the
    arbitrary global phase, never fit the numerical amplitude to the theory.
    Straight triangle edges approximate the circular walls, so the errors
    include geometry approximation and are integrated on the numerical domain.
    """
    fields = mode.fields
    x, y = fields.coordinates
    points, cells = fields.mesh_points, fields.mesh_cells
    triangles = points[cells]
    u, v = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    jacobian = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    _, reference_weights = get_quadrature_tri(QUADRATURE_ORDER)
    weights = (jacobian[:, None] * reference_weights).ravel()
    if fields.metadata['element_quadrature_shape'] != (len(cells), len(reference_weights)):
        raise ValueError('Unexpected field sampling; quadrature weights would not match.')
    eta = np.sqrt(mu_0 / epsilon_0)
    impedance = eta * np.log(OUTER_RADIUS / INNER_RADIUS) / (2 * np.pi)
    voltage = np.sqrt(2 * impedance)  # Peak phasor, 1 W time-average power.
    factor = voltage / np.log(OUTER_RADIUS / INNER_RADIUS) / (x*x + y*y)
    exact_e = np.column_stack((factor*x, factor*y, np.zeros_like(x)))
    exact_h = np.column_stack((-factor*y, factor*x, np.zeros_like(x))) / eta
    numerical_e = np.column_stack([mode.component(c) for c in ('Ex', 'Ey', 'Ez')])
    numerical_h = np.column_stack([mode.component(c) for c in ('Hx', 'Hy', 'Hz')])
    overlap = np.sum(weights[:, None] * np.conj(exact_e) * numerical_e)
    if abs(overlap) == 0 or mode.normalization != 'unit-power':
        raise ValueError('The selected mode is not a forward, unit-power TEM mode.')
    phase = np.conj(overlap) / abs(overlap)

    def relative_error(numerical, exact):
        return float(np.sqrt(np.sum(weights[:, None] * abs(phase*numerical - exact)**2)
                             / np.sum(weights[:, None] * abs(exact)**2)))

    measured_power = .5 * np.sum(weights * np.real(
        numerical_e[:, 0] * np.conj(numerical_h[:, 1])
        - numerical_e[:, 1] * np.conj(numerical_h[:, 0])))
    return dict(
        electric_relative_l2_error=relative_error(numerical_e, exact_e),
        magnetic_relative_l2_error=relative_error(numerical_h, exact_h),
        integrated_power_w=float(measured_power),
        relative_area_error=abs(weights.sum() / (np.pi*(OUTER_RADIUS**2-INNER_RADIUS**2)) - 1),
        exact_impedance_ohm=impedance,
    )


def compare(max_refinements=4, adaptive_tolerance=.05):
    rows, histories, meshes = [], [], []
    # Repeat the same initial mesh with increasing solve budgets because the
    # public result contains final fields, not intermediate field snapshots.
    for budget in range(max_refinements + 1):
        start = perf_counter()
        solver = make_solver()
        result = solver.solve(
            num_modes=1, neff_guess=1.001, max_refinements=budget,
            adaptive_tolerance=adaptive_tolerance, dense_linearization_limit=4,
        )
        mode = result[0]
        history = result.solve_info['adaptive_history']
        row = dict(
            refinement_budget=budget, completed_refinements=history[-1]['refinement'],
            elements=len(mode.fields.mesh_cells), nodes=len(mode.fields.mesh_points),
            frequency_hz=FREQUENCY, inner_radius_m=INNER_RADIUS, outer_radius_m=OUTER_RADIUS,
            neff_real=mode.neff.real, neff_imag=mode.neff.imag,
            absolute_neff_error=abs(mode.neff - 1),
            **field_errors(mode), algebraic_residual=mode.residual,
            adaptive_residual=result.solve_info['adaptive_residual'],
            adaptive_tolerance=adaptive_tolerance,
            adaptive_converged=result.solve_info['adaptive_converged'],
            stopping_reason=history[-1]['status'], elapsed_seconds=perf_counter()-start,
        )
        rows.append(row)
        histories.extend(dict(refinement_budget=budget, **step) for step in history)
        meshes.append((mode.fields.mesh_points, mode.fields.mesh_cells))
        print(f'Budget {budget}: {row["elements"]} triangles; E error={row["electric_relative_l2_error"]:.4%}; '
              f'neff error={row["absolute_neff_error"]:.3e}; estimator={row["adaptive_residual"]:.3e}; '
              f'{row["stopping_reason"]}', flush=True)
        if row['adaptive_converged']:
            break
    return rows, histories, meshes


def save_report(output, rows, histories, meshes):
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in (('comparison.csv', rows), ('adaptive_history.csv', histories)):
        with (output / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    figure = Figure(figsize=(11, 4.5))
    axes = figure.subplots(1, 2)
    counts = [r['elements'] for r in rows]
    for key, label, style in (
        ('electric_relative_l2_error', 'Electric field, relative L2', 'o-'),
        ('magnetic_relative_l2_error', 'Magnetic field, relative L2', 'x--'),
        ('relative_area_error', 'Annulus area, relative error', 's:'),
    ):
        axes[0].loglog(counts, [max(r[key], 1e-16) for r in rows], style, label=label)
    for key, label, style in (
        ('adaptive_residual', 'Adaptive Maxwell jump estimator', 'o-'),
        ('absolute_neff_error', '|neff - 1|', 's--'),
        ('algebraic_residual', 'Eigenproblem residual', '^:'),
    ):
        axes[1].loglog(counts, [max(r[key], 1e-16) for r in rows], style, label=label)
    axes[1].axhline(rows[0]['adaptive_tolerance'], color='gray', ls='--', label='Adaptive tolerance')
    for ax in axes:
        ax.set(xlabel='Triangles in the dielectric annulus', ylabel='Dimensionless error / diagnostic')
        ax.grid(True, which='both', alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_title('Coaxial TEM: error against analytical fields')
    axes[1].set_title('Solver diagnostics (distinct from field error)')
    figure.tight_layout()
    figure.savefig(output / 'convergence.png', dpi=160)
    figure = Figure(figsize=(10, 5))
    for ax, index in zip(figure.subplots(1, 2), (0, len(meshes)-1)):
        points, cells = meshes[index]
        ax.triplot(points[:, 0]*1e3, points[:, 1]*1e3, cells, lw=.3, color='#34678a')
        ax.set_aspect('equal')
        ax.set(xlabel='x (mm)', ylabel='y (mm)', title=f'Budget {rows[index]["refinement_budget"]}: {len(cells)} triangles')
    figure.suptitle('Coaxial FEM mesh: estimator-controlled global remeshing')
    figure.tight_layout()
    figure.savefig(output / 'meshes.png', dpi=160)


def check(rows):
    for row in rows:
        diagnostics = [row[key] for key in (
            'electric_relative_l2_error', 'magnetic_relative_l2_error',
            'absolute_neff_error', 'algebraic_residual', 'adaptive_residual')]
        if not np.all(np.isfinite(diagnostics)):
            raise ValueError('Nonfinite coaxial diagnostics.')
        if row['absolute_neff_error'] > 1e-6 or row['algebraic_residual'] > 1e-8:
            raise ValueError('Coaxial TEM eigenproblem check failed.')
        if abs(row['integrated_power_w'] - 1) > 1e-8:
            raise ValueError('Coaxial mode does not carry 1 W forward power.')
        if row['adaptive_converged'] != (row['adaptive_residual'] <= row['adaptive_tolerance']):
            raise ValueError('Adaptive stopping diagnostics are inconsistent.')
    for key in ('electric_relative_l2_error', 'magnetic_relative_l2_error'):
        if rows[-1][key] >= .05:
            raise ValueError(f'Finest {key} must be below 5%; increase the refinement budget.')
        if len(rows) > 1 and rows[-1][key] >= rows[0][key]:
            raise ValueError(f'Refinement did not reduce {key}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-refinements', type=int, default=4)
    parser.add_argument('--adaptive-tolerance', type=float, default=.05)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Check TEM index, forward power, and <5%% final field errors.')
    args = parser.parse_args()
    if args.max_refinements < 0 or not np.isfinite(args.adaptive_tolerance) or args.adaptive_tolerance <= 0:
        parser.error('Use a nonnegative refinement budget and finite positive tolerance.')
    rows, histories, meshes = compare(args.max_refinements, args.adaptive_tolerance)
    save_report(args.output, rows, histories, meshes)
    if args.check:
        check(rows)
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()

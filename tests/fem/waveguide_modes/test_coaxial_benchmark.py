"""Physical and geometry checks for the public coaxial benchmark workflow."""
from cem_common import Material, SurfaceImpedance, materials, shapes
from pathlib import Path
import runpy

import numpy as np
import pytest

from fem_waveguide_modes import ModeSolver2D
from cem_common.shapes import Circle
from cem_common.errors import ConfigurationError, GeometryError

BENCHMARK = runpy.run_path(str(
    Path(__file__).resolve().parents[3] / 'benchmarks/analytical/coaxial_waveguide_adaptivity.py'
))


def test_clipped_coax_mesh_has_only_annulus_and_named_circular_walls():
    solver = BENCHMARK['make_solver']()
    mesh = solver.mesh_data
    radii = np.linalg.norm(mesh.nodes, axis=1)
    assert np.all(radii >= 1e-3 - 1e-12)
    assert np.all(radii <= 4e-3 + 1e-12)
    for name, radius in (('inner_conductor', 1e-3), ('outer_conductor', 4e-3)):
        facets = mesh.mesh.facets[:, mesh.boundary_facets[name]]
        assert facets.size
        np.testing.assert_allclose(np.linalg.norm(mesh.nodes[facets], axis=2), radius, atol=1e-12)


def test_pec_shape_requires_explicit_clipping_and_unambiguous_arguments():
    solver = ModeSolver2D(frequency=1e9, x_range=1., y_range=1.)
    shape = shapes.Annulus(center=(0.5, 0.5), outer_radius=1.0, inner_radius=0.4)
    with pytest.raises(GeometryError, match='outside'):
        solver.add_geometry(shape=shape, material=materials.PEC)
    with pytest.raises(TypeError, match='x_range'):
        solver.add_geometry(x_range=(0, 1), shape=shape, material=materials.PEC)
    with pytest.raises(TypeError, match='shape'):
        solver.add_geometry(material=materials.PEC, clip=True)
    with pytest.raises(GeometryError, match='does not intersect'):
        solver.add_geometry(clip=True, shape=shapes.Circle(center=(3, 3), radius=0.1), material=materials.PEC)


def test_coaxial_tem_field_error_improves_with_adaptive_budget():
    rows, histories, _ = BENCHMARK['compare'](max_refinements=2)
    assert [r['completed_refinements'] for r in rows] == [0, 1, 2]
    assert all(r['stopping_reason'] == 'refinement_limit' for r in rows)
    assert len(histories) == 6
    for key in ('electric_relative_l2_error', 'magnetic_relative_l2_error'):
        assert rows[-1][key] < .12
        assert rows[-1][key] < .6 * rows[0][key]
    assert all(r['absolute_neff_error'] < 1e-8 for r in rows)
    np.testing.assert_allclose([r['integrated_power_w'] for r in rows], 1., atol=1e-10)


def test_coaxial_estimator_can_stop_before_budget_and_shape_edits_invalidate():
    solver = BENCHMARK['make_solver']()
    result = solver.solve(num_modes=1, neff_guess=1.001, max_refinements=2,
                          adaptive_tolerance=.8, dense_linearization_limit=4)
    assert result.solve_info['adaptive_converged']
    assert len(result.solve_info['adaptive_history']) == 1
    assert result.solve_info['adaptive_history'][0]['status'] == 'tolerance'
    solver.add_geometry(name='insert', shape=shapes.Circle(center=(0.002, 0), radius=0.0001), material=materials.PEC)
    assert solver.mesh_data is None
    assert solver.result is None

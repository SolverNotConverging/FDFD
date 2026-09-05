"""Public lifecycle across the modal FEM implementations."""
from cem_common import Material, SurfaceImpedance, materials, shapes
import numpy as np
import pytest

from cem_common import NoResultError
from fem_waveguide_modes import ModeSolver1D, ModeSolver2D
from fem_periodic_modes import PeriodicModeSolver2D, PeriodicModeSolver3D


@pytest.mark.parametrize('solver_type,ranges,mesh_settings,solve_settings', [
    (ModeSolver1D, dict(x_range=.02), dict(resolution=24), {}),
    (ModeSolver2D, dict(x_range=.02, y_range=.01), dict(resolution=(4, 3)), {}),
    (PeriodicModeSolver2D, dict(x_range=.02, z_range=.005), dict(max_element_size=.004), {}),
    (PeriodicModeSolver3D, dict(x_range=.02, y_range=.01, z_range=.005), dict(max_element_size=.006), dict(eigensolver='dense')),
])
def test_modal_lifecycle(solver_type, ranges, mesh_settings, solve_settings, tmp_path):
    solver = solver_type(frequency=10e9, **ranges)
    assert solver.mesh_data is None and solver.result is None
    with pytest.raises(NoResultError, match='solve'):
        solver.show()
    mesh = solver.mesh(**mesh_settings)
    assert mesh is solver.mesh_data and solver.result is None
    result = solver.solve(num_modes=1, neff_guess=.66, max_refinements=0, **solve_settings)
    assert solver.result is result
    assert result.frequency == 10e9
    assert result.mesh_data is not None
    assert result.mode(0) is result[0]
    assert np.isfinite(result.neff).all()
    from importlib import import_module
    result.save(tmp_path / 'result.h5')
    loaded = import_module(solver_type.__module__.split('.')[0]).load_result(tmp_path / 'result.h5')
    np.testing.assert_allclose(loaded.neff, result.neff)
    assert loaded.mesh_data is not None
    assert loaded.solve_info['adaptive_residual'] == result.solve_info['adaptive_residual']
    import matplotlib.pyplot as plt
    figure = loaded.plot(component='Ey')
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    loaded.save(tmp_path / 'loaded.h5')
    saved_settings = dict(solver._mesh_settings)
    solver.set_boundary(material=materials.PMC)
    assert solver.mesh_data is None and solver.result is None
    solver.set_boundary(material=materials.PEC)
    solver.solve(num_modes=1, neff_guess=.66, max_refinements=0, **solve_settings)
    assert solver._mesh_settings == saved_settings


@pytest.mark.parametrize('dimension', [1, 2])
def test_electrostatic_lifecycle_and_roundtrip(dimension, tmp_path):
    from fem_electrostatics import ElectrostaticSolver, load_result
    solver = ElectrostaticSolver(dim=dimension, x_range=1., outer_potential=None)
    assert solver.result is None and solver.mesh_data is None
    solver.set_potential(potential=0.0, name='ground', geometry='left')
    solver.set_potential(potential=1.0, name='signal', geometry='right')
    solver.mesh(max_element_size=.2)
    result = solver.solve(max_refinements=0)
    np.testing.assert_allclose(result.potential, result.coordinates[:, 0], atol=1e-12)
    result.save(tmp_path / 'static.h5')
    loaded = load_result(tmp_path / 'static.h5')
    np.testing.assert_array_equal(loaded.potential, result.potential)
    assert loaded.solve_info == result.solve_info
    assert loaded.conductor_charge('signal') == result.conductor_charge('signal')
    loaded.save(tmp_path / 'static-again.h5')
    import matplotlib.pyplot as plt
    plt.close(loaded.plot(component='Ex'))

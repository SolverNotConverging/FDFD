"""Result envelopes and lazy sweep behavior across FEM families."""
import h5py
import numpy as np
import pytest

from fem_common import PersistenceError


@pytest.mark.parametrize('package', ['fem_waveguide_modes', 'fem_periodic_modes', 'fem_waveguide_scattering', 'fem_electrostatics'])
def test_old_archive_is_rejected(package, tmp_path):
    from importlib import import_module
    path = tmp_path/'old.h5'
    with h5py.File(path,'w') as archive:
        archive.attrs['format'] = 'old-result'
    with pytest.raises(PersistenceError, match='[Ii]ncompatible'):
        import_module(package).load_result(path)


def test_periodic_sweep_lazy_roundtrip_and_context(tmp_path, monkeypatch):
    from fem_periodic_modes import PeriodicModeSolver2D, PeriodicSweepResult, load_result
    cases=[]
    for frequency in (10e9, 11e9):
        solver=PeriodicModeSolver2D(frequency=frequency,x_range=.02,z_range=.005)
        solver.mesh(max_element_size=.004)
        cases.append(solver.solve(num_modes=1, neff_guess=.66, max_refinements=0))
    sweep=PeriodicSweepResult.from_results(cases)
    path=tmp_path/'sweep.h5'
    sweep.save(path)
    from fem_periodic_modes.persistence import PeriodicH5Archive
    original = PeriodicH5Archive.load_case
    reads=[]
    def read(self,index,*args,**kwargs):
        reads.append(index)
        return original(self,index,*args,**kwargs)
    monkeypatch.setattr(PeriodicH5Archive,'load_case',read)
    loaded=load_result(path)
    assert isinstance(loaded,PeriodicSweepResult) and len(loaded)==2
    assert not reads
    case=loaded[1]
    assert reads==[1]
    np.testing.assert_array_equal(case.neff,cases[1].neff)
    assert case.mesh_data.metadata['context']['outer_boundary']=='pec'
    loaded.save(path)  # Atomic replacement while reading the old lazy archive.
    assert len(load_result(path))==2
    figure=case.plot(component='Ey')
    assert figure.axes


def test_electrostatic_archive_preserves_tensor_and_boundaries(tmp_path):
    from fem_electrostatics import ElectrostaticSolver,load_result
    solver=ElectrostaticSolver(x_range=1.,background_epsilon=((2.,.25),(.25,1.)),outer_potential=None)
    solver.set_potential(region='left',potential=0.,name='ground')
    solver.set_potential(region='right',potential=1.,name='drive')
    solver.mesh(max_element_size=.2)
    result=solver.solve(max_refinements=0)
    path=tmp_path/'static.h5'
    result.save(path)
    loaded=load_result(path)
    context=loaded.mesh_data.metadata['context']
    assert len(context['potentials'])==2
    assert context['background']==result.mesh_data.metadata['context']['background']
    np.testing.assert_array_equal(loaded.element_displacement_field,result.element_displacement_field)
    with h5py.File(path,'r+') as archive:
        archive.attrs['time_convention']='exp(-i*omega*t)'
    with pytest.raises(PersistenceError,match='time_convention'):
        load_result(path)

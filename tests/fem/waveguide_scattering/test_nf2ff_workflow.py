import numpy as np
import pytest

from fem_waveguide_scattering import load_result, ConfigurationError
from fem_waveguide_scattering.nf2ff import capture_contour
from examples.fem.waveguide_scattering.grounded_slab_slot_2d import build_simulation


@pytest.mark.gmsh
@pytest.mark.slow
def test_grounded_slot_closed_contour_matching_persistence_and_contour_movement(tmp_path):
    s=build_simulation(matched_ports=True)
    s.set_nf2ff_contour(x_range=(-.012,.012),z_range=(-.015,.015))
    s.mesh(max_element_size=.001,element_order=2,wavelength_elements=10)
    s.solve_modes(num_modes=1,neff_guess=1.8,num_elements=384,max_refinements=0)
    s.set_incident_mode(0)
    r=s.solve(max_refinements=0)
    theta=(np.arange(120)+.5)*2*np.pi/120
    far=r.far_field(theta)
    assert far.integrated_power()>0.1
    np.testing.assert_allclose(far.directivity,2*np.pi*far.power_density/far.integrated_power())
    np.testing.assert_allclose(far.gain,2*np.pi*far.power_density/(r.incident_power-r.reflected_power))
    np.testing.assert_allclose(far.realized_gain,2*np.pi*far.power_density/r.incident_power)
    subset=r.far_field([0.0,0.25])
    assert subset.radiated_power==pytest.approx(far.integrated_power(),rel=.01)
    assert np.isfinite(subset.directivity).all()
    assert far.integrated_power()==pytest.approx(r.radiated_power,rel=.04)
    assert r.power_balance_error < .01
    # The mesh already includes the modal monitors and x/PML interfaces.
    # Capture a larger complete rectangle from the *same* solved coefficients.
    s._nf2ff_request=(None,None,None)
    larger=capture_contour(s,s._adaptive_system,s._adaptive_coefficients)
    far_large=larger.far_field(theta)
    relative=np.linalg.norm(far_large.amplitude-far.amplitude)/np.linalg.norm(far.amplitude)
    assert relative<.06
    assert larger.weights.sum()>r.nf2ff.weights.sum()
    path=tmp_path/'radiation.h5'
    r.save(path)
    import h5py
    with h5py.File(path) as archive:
        saved=archive['results']['000000']['radiation_pattern']
        assert saved.attrs['definition']=='2d-isotropic-2pi'
        assert saved['theta'].shape==(720,)
        np.testing.assert_allclose(saved['directivity'][:],
            2*np.pi*saved['power_density'][:]/saved.attrs['radiated_power'])
        np.testing.assert_allclose(saved['gain'][:],
            2*np.pi*saved['power_density'][:]/saved.attrs['accepted_power'])
        np.testing.assert_allclose(saved['realized_gain'][:],
            2*np.pi*saved['power_density'][:]/saved.attrs['incident_power'])
    restored=load_result(path)
    np.testing.assert_array_equal(restored.far_field(theta).amplitude,far.amplitude)
    assert restored.nf2ff.exterior.pec.sum()==1
    clone=s._clone_at_frequency(s.frequency*1.01)
    assert clone._matched_ports and clone._nf2ff_request==s._nf2ff_request


def test_contour_rejects_cut_slot_and_pml_overlap():
    s=build_simulation()
    s.set_nf2ff_contour(x_range=(-.01,.01),z_range=(-.0005,.0005))
    with pytest.raises(ConfigurationError,match="enclose every PEC slot"):
        s.mesh()
    s.set_nf2ff_contour(x_range=(-.019,.019),z_range=(-.015,.015))
    with pytest.raises(ConfigurationError,match="before the PML"):
        s.mesh()


@pytest.mark.gmsh
def test_uniform_open_guide_closed_contour_has_no_radiation():
    s=build_simulation(matched_ports=True)
    s.geometry.pec_slots.clear()
    s.set_nf2ff_contour()
    s.mesh(max_element_size=.002)
    s.solve_modes(num_modes=1,neff_guess=1.8,num_elements=64,max_refinements=0)
    s.set_incident_mode(0)
    r=s.solve(max_refinements=0)
    np.testing.assert_array_equal(r.far_field([0.,np.pi]).amplitude,np.zeros((3,2)))

import numpy as np
import pytest

from cem_common import materials
from fem_waveguide_scattering import WaveguideScatteringSolver2D


@pytest.mark.gmsh
@pytest.mark.parametrize("element_order", [1,2])
@pytest.mark.parametrize("family", ["TEM", "TE1"])
def test_matched_tem_ports_reproduce_dielectric_slab(element_order, family):
    width = .005 if family == "TEM" else .02
    s=WaveguideScatteringSolver2D(frequency=10e9,x_range=(0.,width),z_range=(-.025,.025),boundary=materials.PEC)
    s.add_rectangle(x_range=s.x_range,z_range=(-.003,.003),material=materials.Material(epsilon=4.))
    s.set_matched_ports()
    s.mesh(max_element_size=.0006,element_order=element_order)
    s.solve_modes(num_modes=1,neff_guess=1. if family == "TEM" else .66,num_elements=100,max_refinements=0)
    s.set_incident_mode(0,reference_plane=-.003)
    r=s.solve(max_refinements=0)
    cutoff = 0. if family == "TEM" else np.pi/width
    beta0, beta1 = np.sqrt(s.k0**2-cutoff**2), np.sqrt(4*s.k0**2-cutoff**2)
    assert s.modes[0].beta.real == pytest.approx(beta0,rel=.001)
    interface = (beta0-beta1)/(beta0+beta1)
    phase=np.exp(-1j*beta1*.006)
    reflection=interface*(1-phase**2)/(1-interface**2*phase**2)
    transmission=(1-interface**2)*phase/(1-interface**2*phase**2)*np.exp(1j*beta0*.006)
    assert r.S11 == pytest.approx(reflection,abs=.015 if element_order == 1 else .002)
    assert r.S21 == pytest.approx(transmission,abs=.015 if element_order == 1 else .002)
    assert r.reflection+r.transmission == pytest.approx(1.,abs=.01)
    assert r.solve_info["matched_port_mode_count"] == 1


@pytest.mark.gmsh
def test_uniform_matched_ports_have_zero_scattered_field():
    s=WaveguideScatteringSolver2D(frequency=10e9,x_range=(0.,.005),z_range=(-.02,.02),boundary=materials.PEC)
    s.set_matched_ports()
    s.mesh(max_element_size=.001)
    s.solve_modes(num_modes=1,neff_guess=1.,num_elements=25,max_refinements=0)
    s.set_incident_mode(0)
    result=s.solve(max_refinements=0)
    assert np.linalg.norm(result.E_scattered) == 0
    assert result.S21 == pytest.approx(1.,abs=1e-10)

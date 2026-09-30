from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import hankel2

from fem_waveguide_scattering.farfield import ClosedContourFields, LayeredExterior
from fem_waveguide_scattering.constants import C0, MU_0, EPSILON_0
from fem_waveguide_scattering.exceptions import ConfigurationError


def rectangle_fields(field, *, xr=(-.7,.9), zr=(-.8,.6), ky=0., exterior=None, order=90):
    exterior = exterior or LayeredExterior([], [1.], [1.], [])
    nodes, weights = leggauss(order)
    points, normals, quadrature = [], [], []
    for axis, value, normal, limits in (
        (0,xr[0],(-1,0),zr), (0,xr[1],(1,0),zr),
        (1,zr[0],(0,-1),xr), (1,zr[1],(0,1),xr),
    ):
        cuts = [limits[0], *([v for v in exterior.interfaces if limits[0] < v < limits[1]] if axis == 1 else []), limits[1]]
        for a,b in zip(cuts[:-1], cuts[1:]):
            p = np.zeros((2,order))
            p[axis] = value
            p[1-axis] = (a+b)/2 + nodes*(b-a)/2
            points.append(p)
            normals.append(np.tile(normal,(order,1)).T)
            quadrature.append(weights*(b-a)/2)
    xy = np.concatenate(points,axis=1)
    e,h = field(*xy)
    return ClosedContourFields(xr,zr,xy,np.concatenate(normals,axis=1),
        np.concatenate(quadrature),e,h,C0,ky,exterior)


def line_source(x,z, *, source=(0.,0.), ky=0., polarization="electric"):
    k = 2*np.pi
    q = np.sqrt(k*k-ky*ky)
    x,z = x-source[0], z-source[1]
    r = np.hypot(x,z)
    u = hankel2(0,q*r)
    dx, dz = -q*hankel2(1,q*r)*x/r, -q*hankel2(1,q*r)*z/r
    e = np.array((-1j*ky*dx/k**2, (1-ky**2/k**2)*u, -1j*ky*dz/k**2))
    h = np.array((-dz,np.zeros_like(u),dx))/(-1j*2*np.pi*C0*MU_0)
    if polarization == "magnetic":
        eta = np.sqrt(MU_0/EPSILON_0)
        return -eta*h, e/eta
    return e,h


@pytest.mark.parametrize("ky", [0., 1.8])
@pytest.mark.parametrize("polarization", ["electric", "magnetic"])
def test_closed_contour_matches_hankel_phase_polarization_and_power(ky,polarization):
    source = (.12,-.08)
    c = rectangle_fields(lambda x,z: line_source(x,z,source=source,ky=ky,polarization=polarization), ky=ky)
    theta = (np.arange(120)+.5)*2*np.pi/120
    ff = c.far_field(theta)
    k,q = 2*np.pi,np.sqrt((2*np.pi)**2-ky**2)
    direction = np.array((q*np.cos(theta),np.full(len(theta),ky),q*np.sin(theta)))/k
    base = np.sqrt(2/(np.pi*q))*np.exp(1j*np.pi/4)*np.exp(1j*q*(source[0]*np.cos(theta)+source[1]*np.sin(theta)))
    e = (np.array((0.,1.,0.))[:,None]-direction*ky/k)*base
    if polarization == "magnetic":
        e = -np.cross(direction.T,e.T).T
    np.testing.assert_allclose(ff.amplitude,e,rtol=1e-10,atol=1e-12)
    assert ff.integrated_power() == pytest.approx(c.outward_power,rel=1e-10)
    larger = rectangle_fields(lambda x,z: line_source(x,z,source=source,ky=ky,polarization=polarization),
                              xr=(-1.1,1.2),zr=(-1.,.85),ky=ky)
    np.testing.assert_allclose(larger.far_field(theta).amplitude,ff.amplitude,rtol=1e-10,atol=1e-12)


def test_pec_halfspace_image_field_on_closed_contour():
    exterior = LayeredExterior([0.], [1.,1.], [1.,1.], [True])
    def fields(x,z):
        e,h = line_source(x,z,source=(.25,0))
        em,hm = line_source(x,z,source=(-.25,0))
        return (e-em)*(x>0), (h-hm)*(x>0)
    c = rectangle_fields(fields,exterior=exterior)
    angles = (np.arange(100)+.5)*2*np.pi/100
    ff = c.far_field(angles)
    expected = np.sqrt(2/(np.pi*2*np.pi))*np.exp(1j*np.pi/4)*2j*np.sin(2*np.pi*.25*np.cos(angles))*(np.cos(angles)>0)
    np.testing.assert_allclose(ff.amplitude[1], expected, atol=1e-12)
    assert ff.integrated_power() == pytest.approx(c.outward_power, rel=1e-10)


@pytest.mark.parametrize("pol", ["s","p"])
def test_reciprocal_layer_fields_satisfy_interface_conditions(pol):
    medium = LayeredExterior([-.2,.3], [1.,2.5,1.44], [1.,1.2,1.], [False,False])
    for side in (-1,1):
        for cut in medium.interfaces:
            x=np.array([cut-1e-10,cut+1e-10])
            e,h,_=medium.reciprocal_fields(x,np.zeros(2),omega=2*np.pi*C0,ky=1.,kz=2.,side=side,polarization=pol)
            np.testing.assert_allclose(e[1:,0], e[1:,1], rtol=2e-8,atol=1e-9)
            np.testing.assert_allclose(h[1:,0], h[1:,1], rtol=2e-8,atol=1e-11)


def test_layered_bound_guided_field_cancels_on_all_four_sides():
    from scipy.optimize import brentq
    k, a, eps = 2*np.pi, .15, 4.
    beta=brentq(lambda b: np.sqrt(k*k*eps-b*b)*np.tan(np.sqrt(k*k*eps-b*b)*a)-np.sqrt(b*b-k*k),
                np.sqrt(k*k*eps-(np.pi/(2*a))**2)+1e-5, k*np.sqrt(eps)-1e-5)
    h,alpha = np.sqrt(k*k*eps-beta*beta),np.sqrt(beta*beta-k*k)
    def fields(x,z):
        inside=abs(x)<a
        u=np.where(inside,np.cos(h*x),np.cos(h*a)*np.exp(-alpha*(abs(x)-a)))
        dx=np.where(inside,-h*np.sin(h*x),-np.sign(x)*alpha*u)
        phase=np.exp(-1j*beta*z)
        e=np.array((np.zeros_like(u),u,np.zeros_like(u)),complex)*phase
        magnetic=np.array((1j*beta*u,np.zeros_like(u),dx))*phase/(-1j*2*np.pi*C0*MU_0)
        return e,magnetic
    medium=LayeredExterior([-a,a],[1.,eps,1.],[1.,1.,1.],[False,False])
    c=rectangle_fields(fields,exterior=medium)
    ff=c.far_field(np.linspace(-1.4,1.4,41))
    assert np.max(abs(ff.amplitude)) < 1e-11


def test_incomplete_contours_and_grazing_are_rejected():
    c=rectangle_fields(line_source)
    mask=c.normals[0]!=-1
    with pytest.raises(ConfigurationError,match="closed"):
        replace(c, coordinates=c.coordinates[:,mask],normals=c.normals[:,mask],
                weights=c.weights[mask],E=c.E[:,mask],H=c.H[:,mask])
    with pytest.raises(ConfigurationError,match="grazing"):
        c.far_field([np.pi/2])
    with pytest.raises(ConfigurationError,match="full-circle"):
        c.far_field(np.linspace(-.3,.3,10)).integrated_power()


def test_radiating_interface_source_has_correct_normalization_in_both_media():
    # Same wavenumber, different impedances. H0 centered on the interface has
    # continuous tangential E and H away from the source, in both half-spaces.
    medium=LayeredExterior([0.], [2.,1.], [.5,1.], [False])
    def fields(x,z):
        e,h=line_source(x,z)
        return e,h/np.where(x<0,.5,1.)
    c=rectangle_fields(fields,exterior=medium)
    theta=(np.arange(100)+.5)*2*np.pi/100
    ff=c.far_field(theta)
    expected=np.sqrt(2/(np.pi*2*np.pi))*np.exp(1j*np.pi/4)
    np.testing.assert_allclose(ff.amplitude[1],expected,rtol=1e-11,atol=1e-12)
    assert ff.integrated_power()==pytest.approx(c.outward_power,rel=1e-11)


def test_2d_directivity_gain_and_realized_gain_for_circular_radiator():
    contour=rectangle_fields(line_source)
    theta=(np.arange(120)+.5)*2*np.pi/120
    field=contour.far_field(theta)
    radiated=field.integrated_power()
    pattern=field.with_feed_powers(radiated_power=radiated,
        incident_power=2*radiated,reflected_power=.5*radiated)
    np.testing.assert_allclose(pattern.directivity,1.0,rtol=1e-11)
    np.testing.assert_allclose(pattern.gain,2/3,rtol=1e-11)
    np.testing.assert_allclose(pattern.realized_gain,.5,rtol=1e-11)
    assert pattern.accepted_power==pytest.approx(1.5*radiated)
    assert np.mean(pattern.directivity)==pytest.approx(1.0,abs=1e-11)


def test_zero_radiation_has_undefined_directivity_and_zero_gain():
    contour=rectangle_fields(lambda x,z:(np.zeros((3,len(x)),complex),
                                          np.zeros((3,len(x)),complex)))
    field=contour.far_field((np.arange(40)+.5)*2*np.pi/40)
    pattern=field.with_feed_powers(radiated_power=0.,incident_power=1.,reflected_power=.2)
    assert np.isnan(pattern.directivity).all()
    np.testing.assert_array_equal(pattern.gain,np.zeros(40))
    np.testing.assert_array_equal(pattern.realized_gain,np.zeros(40))

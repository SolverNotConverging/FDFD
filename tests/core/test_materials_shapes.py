"""Materials, metal presets, shapes and grid-size guidance."""

import math

import numpy as np
import pytest

import fdfd
from fdfd import (
    BackendCapabilityError,
    ConfigurationError,
    GeometryError,
    Material,
    SurfaceImpedance,
    materials,
    shapes,
)
from fdfd.core.constants import MU_0

RESISTIVITIES = {
    "aluminium": 2.650e-8,
    "copper": 1.676e-8,
    "gold": 2.192e-8,
    "molybdenum": 5.340e-8,
    "palladium": 1.054e-7,
    "silver": 1.586e-8,
    "tungsten": 5.280e-8,
    "zinc": 5.964e-8,
}


def test_materials_are_named_values_with_exp_plus_iwt_loss_sign():
    dielectric = Material(name="lossy dielectric", epsilon=2.5 - 0.03j, mu=1.0)
    assert dielectric.is_passive
    assert not Material(name="active", epsilon=2.5 + 0.03j).is_passive
    assert materials.vacuum.name == "vacuum"
    assert materials.air.is_passive
    assert SurfaceImpedance(impedance=1.0 + 1.0j).at_frequency(frequency=10e9) == 1.0 + 1.0j
    with pytest.raises(ConfigurationError):
        Material(name="", epsilon=1.0)
    with pytest.raises(BackendCapabilityError, match="off-diagonal"):
        materials.bulk_values(Material(name="tensor", epsilon=((2.0, 0.1), (0.1, 2.0))), dimension=2)


def test_metal_presets_use_reference_resistivities():
    assert materials.RESISTIVITIES == RESISTIVITIES
    for name, resistivity in RESISTIVITIES.items():
        assert getattr(materials, name).conductivity == 1 / resistivity


def test_good_conductor_impedance_has_positive_reactance_for_exp_plus_iwt():
    frequency = 10e9
    expected = math.sqrt(math.pi * frequency * MU_0 * RESISTIVITIES["copper"])
    impedance = materials.copper.at_frequency(frequency=frequency)
    assert impedance.real == pytest.approx(expected)
    assert impedance.imag == pytest.approx(expected)
    magnetic = fdfd.GoodConductor(name="magnetic", conductivity=1 / RESISTIVITIES["tungsten"], mu=4.0)
    base = materials.tungsten.at_frequency(frequency=1e9)
    assert magnetic.at_frequency(frequency=1e9) == pytest.approx(2.0 * base)
    for bad in (0.0, math.inf):
        with pytest.raises(ConfigurationError):
            materials.copper.at_frequency(frequency=bad)


def test_shapes_support_boolean_and_transformed_geometry():
    disk = shapes.Circle(center=(0.0, 0.0), radius=1.0)
    aperture = shapes.Rectangle(bounds=((-0.25, 0.25), (-2.0, 2.0)))
    split_disk = shapes.Difference(shape=disk, tool=aperture)
    moved = split_disk.translated(offset=(2.0, 0.0)).rotated(angle=90.0, center=(2.0, 0.0))
    assert split_disk.contains(0.75, 0.0)
    assert not split_disk.contains(0.0, 0.0)
    assert moved.contains(2.0, 0.75)
    assert shapes.Annulus(center=(0.0, 0.0), inner_radius=0.5, outer_radius=1.0).contains(0.75, 0.0)
    assert shapes.Ellipsoid(center=(0.0, 0.0, 0.0), radii=(1.0, 2.0, 3.0)).dimension == 3
    with pytest.raises(GeometryError):
        shapes.Polygon(points=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))


def test_out_of_domain_shapes_need_clip():
    solver = fdfd.ModeSolver2D(frequency=1e9, x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    with pytest.raises(GeometryError, match="clip"):
        solver.add_rectangle(x_range=(-0.5, 0.5), y_range=(0.0, 1.0), material=materials.PEC)
    solver.add_rectangle(x_range=(-0.5, 0.5), y_range=(0.0, 1.0), material=materials.PEC, clip=True)


def test_capabilities_are_checked_when_geometry_is_added():
    bands = fdfd.BandStructureSolver2D(x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    with pytest.raises(BackendCapabilityError):
        bands.add_circle(center=(0.5, 0.5), radius=0.2, material=materials.PEC)
    with pytest.raises(BackendCapabilityError):
        bands.add_circle(center=(0.5, 0.5), radius=0.2, material=Material(name="aniso", epsilon=(2, 3, 4)))
    scattering = fdfd.ScatteringSolver2D(frequency=1e9, x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    with pytest.raises(BackendCapabilityError):
        scattering.add_circle(center=(0.5, 0.5), radius=0.2, material=materials.PMC)
    with pytest.raises(BackendCapabilityError):
        scattering.add_circle(center=(0.5, 0.5), radius=0.2, material=materials.copper)


def test_suggest_resolution_resolves_the_shortest_wavelength():
    dielectric = Material(name="d", epsilon=9.0)
    cells = fdfd.suggest_resolution(
        frequency=10e9, ranges=((0.0, 0.1), (0.0, 0.05)), materials=(dielectric,), points_per_wavelength=10
    )
    wavelength = fdfd.core.constants.C0 / 10e9 / 3.0
    assert cells == (math.ceil(0.1 / (wavelength / 10)), math.ceil(0.05 / (wavelength / 10)))


def test_suggest_pml_strength_meets_the_target_reflection():
    strength = fdfd.suggest_pml_strength(thickness=0.01, frequency=10e9, reflection=1e-6, order=3)
    k = 2 * np.pi * 10e9 / fdfd.core.constants.C0
    assert math.exp(-2 * k * strength * 0.01 / 4) == pytest.approx(1e-6)

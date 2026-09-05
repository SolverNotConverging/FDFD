import numpy as np
import pytest

from fem_waveguide_scattering.exceptions import ConfigurationError
from fem_waveguide_scattering.geometry import GeometryModel
from fem_waveguide_scattering.materials import Material


def make_geometry() -> GeometryModel:
    geometry = GeometryModel((-2.0, 2.0), (-4.0, 4.0), Material(eps_r=1.0))
    geometry.add_rectangle(
        x=(-0.5, 0.5), z="all", material=Material(eps_r=4.0), background=True, name="core"
    )
    geometry.add_circle(center=(0.0, 0.0), radius=0.25, material=5.0, name="defect")
    return geometry


def test_actual_and_background_materials_are_distinct() -> None:
    geometry = make_geometry()
    x = np.array([1.0, 0.4, 0.0])
    z = np.zeros(3)
    eps_b, _ = geometry.material_at(x, z, profile="background")
    eps_a, _ = geometry.material_at(x, z, profile="actual")
    np.testing.assert_array_equal(eps_b, [1.0, 4.0, 4.0])
    np.testing.assert_array_equal(eps_a, [1.0, 4.0, 5.0])


def test_background_region_must_be_z_invariant() -> None:
    geometry = GeometryModel((-1.0, 1.0), (-1.0, 1.0), Material(eps_r=1.0))
    with pytest.raises(ConfigurationError, match="z-invariant"):
        geometry.add_rectangle(x=(-0.2, 0.2), z=(-0.5, 0.5), material=2.0, background=True)


def test_region_names_and_physical_tags_are_stable() -> None:
    geometry = make_geometry()
    assert geometry.physical_names == {1: "exterior", 2: "core", 3: "defect"}
    np.testing.assert_array_equal(
        geometry.region_tag_at(np.array([1.0, 0.4, 0.0]), np.zeros(3)), [1, 2, 3]
    )


def test_background_pec_sheet_and_actual_slot_profiles_are_distinct() -> None:
    geometry = GeometryModel((-1.0, 1.0), (-4.0, 4.0), Material())
    ground = geometry.add_pec(
        x=-0.25,
        z="all",
        background=True,
        name="ground",
    )
    first = geometry.add_slot(pec=ground, z=(-0.6, 0.6), name="main_slot")
    second = geometry.add_slot(pec="ground", z=(1.0, 1.4), name="side_slot")

    assert ground.x == pytest.approx(-0.25)
    assert geometry.slots_in(ground) == (first, second)
    assert [segment.z for segment in geometry.pec_segments(profile="background")] == [
        (-4.0, 4.0)
    ]
    assert [segment.z for segment in geometry.pec_segments(profile="actual")] == [
        (-4.0, -0.6),
        (0.6, 1.0),
        (1.4, 4.0),
    ]


def test_background_pec_sheet_must_be_z_invariant() -> None:
    geometry = GeometryModel((-1.0, 1.0), (-2.0, 2.0), Material())
    with pytest.raises(ConfigurationError, match="z-invariant"):
        geometry.add_pec(
            x=0.0,
            z=(-1.0, 1.0),
            background=True,
        )


def test_pec_slot_must_be_compact_disjoint_and_on_background_sheet() -> None:
    geometry = GeometryModel((-1.0, 1.0), (-2.0, 2.0), Material())
    ground = geometry.add_pec(x=0.0, background=True, name="ground")
    geometry.add_slot(pec=ground, z=(-0.5, 0.5), name="first")

    with pytest.raises(ConfigurationError, match="compact"):
        geometry.add_slot(pec=ground, z=(-2.0, -1.5))
    with pytest.raises(ConfigurationError, match="overlaps or touches"):
        geometry.add_slot(pec=ground, z=(0.5, 0.8))

    finite = geometry.add_pec(
        x=0.5,
        z=(-1.0, 1.0),
        background=False,
        name="actual_only",
    )
    with pytest.raises(ConfigurationError, match="background PEC"):
        geometry.add_slot(pec=finite, z=(-0.2, 0.2))


def test_internal_pec_cannot_duplicate_the_numerical_outer_boundary() -> None:
    geometry = GeometryModel((-1.0, 1.0), (-2.0, 2.0), Material())
    with pytest.raises(ConfigurationError, match="strictly inside x_span"):
        geometry.add_pec(
            x=-1.0,
            background=True,
            name="outer",
        )

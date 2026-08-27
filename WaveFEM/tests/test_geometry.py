import numpy as np
import pytest

from wavefem.exceptions import ConfigurationError
from wavefem.geometry import GeometryModel
from wavefem.materials import Material


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

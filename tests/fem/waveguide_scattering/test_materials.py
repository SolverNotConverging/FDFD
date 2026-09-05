from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fem_waveguide_scattering.materials import Material
from fem_waveguide_scattering import MaterialError
from fem_waveguide_scattering.materials import (
    DiagonalMaterial,
    DiagonalTensor,
    as_diagonal_material,
)


def test_default_material_is_complex_isotropic_vacuum() -> None:
    material = Material()
    assert material.eps_r == 1.0 + 0.0j
    assert material.mu_r == 1.0 + 0.0j
    assert isinstance(material.eps_r, complex)
    assert isinstance(material.mu_r, complex)
    assert material.is_lossless
    assert material.is_passive


def test_finite_python_and_numpy_scalars_are_normalized_to_complex() -> None:
    material = Material(eps_r=np.float64(12.0), mu_r=np.complex128(1.0 + 0.02j))
    assert material.eps_r == 12.0 + 0.0j
    assert material.mu_r == 1.0 + 0.02j
    assert isinstance(material.eps_r, complex)
    assert isinstance(material.mu_r, complex)


def test_passive_loss_uses_negative_imaginary_part_for_exp_plus_iwt() -> None:
    passive = Material(eps_r=2.0 - 0.1j)
    active = Material(eps_r=2.0 + 0.1j)
    assert passive.is_passive
    assert not passive.is_lossless
    assert not active.is_passive
    assert not active.is_lossless


@pytest.mark.parametrize("name", ["eps_r", "mu_r"])
@pytest.mark.parametrize(
    "value",
    [complex(np.nan, 0.0), complex(np.inf, 0.0), complex(1.0, np.nan), True, "2.0"],
)
def test_material_rejects_nonfinite_or_nonnumeric_scalars(name: str, value: object) -> None:
    kwargs = {"eps_r": 1.0, "mu_r": 1.0, name: value}
    with pytest.raises(MaterialError):
        Material(**kwargs)


@pytest.mark.parametrize(
    "tensor",
    [
        [1.0, 2.0, 3.0],
        (1.0, 2.0, 3.0),
        np.asarray([1.0, 2.0, 3.0]),
        np.eye(3),
        {"xx": 1.0, "yy": 2.0, "zz": 3.0},
    ],
)
def test_public_material_explicitly_rejects_tensor_inputs(tensor: object) -> None:
    with pytest.raises(NotImplementedError, match="tensor materials"):
        Material(eps_r=tensor)


def test_isotropic_material_expands_to_explicit_physical_diagonal_order() -> None:
    diagonal = as_diagonal_material(Material(eps_r=2.0 + 0.1j, mu_r=3.0))
    assert diagonal.eps_r == DiagonalTensor(
        xx=2.0 + 0.1j,
        yy=2.0 + 0.1j,
        zz=2.0 + 0.1j,
    )
    assert diagonal.mu_r == DiagonalTensor(xx=3.0, yy=3.0, zz=3.0)
    np.testing.assert_array_equal(
        diagonal.eps_r.as_array(),
        np.asarray([2.0 + 0.1j, 2.0 + 0.1j, 2.0 + 0.1j]),
    )
    assert diagonal.eps_r.as_array().dtype == np.complex128
    assert diagonal.eps_r.is_isotropic


def test_internal_diagonal_representation_preserves_anisotropy_for_pml() -> None:
    material = DiagonalMaterial(
        eps_r=DiagonalTensor(1.0 + 0.2j, 2.0 + 0.3j, 3.0 + 0.4j),
        mu_r=DiagonalTensor(4.0, 5.0, 6.0),
    )
    assert as_diagonal_material(material) is material
    np.testing.assert_array_equal(
        material.eps_r.as_array(),
        np.asarray([1.0 + 0.2j, 2.0 + 0.3j, 3.0 + 0.4j]),
    )
    assert not material.eps_r.is_isotropic


def test_diagonal_material_requires_explicit_tensor_objects() -> None:
    with pytest.raises(MaterialError, match="eps_r"):
        DiagonalMaterial(eps_r=1.0, mu_r=DiagonalTensor.isotropic(1.0))  # type: ignore[arg-type]
    with pytest.raises(MaterialError, match="mu_r"):
        DiagonalMaterial(eps_r=DiagonalTensor.isotropic(1.0), mu_r=1.0)  # type: ignore[arg-type]


def test_material_objects_are_immutable_and_array_conversion_returns_a_copy() -> None:
    material = Material(eps_r=2.0)
    with pytest.raises(FrozenInstanceError):
        material.eps_r = 3.0  # type: ignore[misc]

    tensor = DiagonalTensor(1.0, 2.0, 3.0)
    array = tensor.as_array()
    array[0] = 99.0
    assert tensor.xx == 1.0 + 0.0j


def test_diagonal_adapter_rejects_unrelated_objects() -> None:
    with pytest.raises(MaterialError, match="Expected Material"):
        as_diagonal_material(object())  # type: ignore[arg-type]

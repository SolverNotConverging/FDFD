from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import pi

import numpy as np
import pytest

from wavefem import C0, ConfigurationError, Frequency, resolve_frequency


def test_equivalent_frequency_inputs_resolve_to_the_same_spectral_point() -> None:
    wavelength = 1.55e-6
    frequency = C0 / wavelength
    omega = 2.0 * pi * frequency

    by_wavelength = resolve_frequency(wavelength=wavelength)
    by_frequency = resolve_frequency(frequency=frequency)
    by_omega = resolve_frequency(omega=omega)

    assert by_wavelength.omega == pytest.approx(omega)
    assert by_frequency.omega == pytest.approx(omega)
    assert by_omega.omega == pytest.approx(omega)
    assert by_wavelength.frequency == pytest.approx(frequency)
    assert by_wavelength.wavelength == pytest.approx(wavelength)
    assert by_wavelength.k0 == pytest.approx(2.0 * pi / wavelength)


def test_named_constructors_document_the_input_quantity() -> None:
    assert Frequency.from_wavelength(2.0).wavelength == pytest.approx(2.0)
    assert Frequency.from_frequency(3.0).frequency == pytest.approx(3.0)
    assert Frequency.from_omega(4.0).omega == pytest.approx(4.0)
    assert Frequency(omega=5.0).angular_frequency == pytest.approx(5.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"wavelength": 1.0, "frequency": 2.0},
        {"wavelength": 1.0, "omega": 2.0},
        {"frequency": 1.0, "omega": 2.0},
        {"wavelength": 1.0, "frequency": 2.0, "omega": 3.0},
    ],
)
def test_resolver_requires_exactly_one_independent_input(kwargs: dict[str, float]) -> None:
    with pytest.raises(ConfigurationError, match="exactly one"):
        resolve_frequency(**kwargs)


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, np.nan, np.inf, -np.inf, True, 1.0 + 0.0j, "1.0"],
)
@pytest.mark.parametrize("name", ["wavelength", "frequency", "omega"])
def test_frequency_inputs_reject_nonpositive_nonfinite_or_nonreal_values(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ConfigurationError):
        resolve_frequency(**{name: value})


def test_numpy_real_scalars_are_accepted() -> None:
    point = resolve_frequency(frequency=np.float64(193.4e12))
    assert point.frequency == pytest.approx(193.4e12)


def test_frequency_is_immutable() -> None:
    point = Frequency.from_frequency(1.0)
    with pytest.raises(FrozenInstanceError):
        point.omega = 2.0  # type: ignore[misc]

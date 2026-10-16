"""Scalar validation for the resident CUDA eigensolver."""

from numbers import Integral, Real

import numpy as np

from .errors import ConfigurationError


def finite_real(value, name):
    """Return a finite real scalar, without parsing text or discarding phase."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a finite real scalar.")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be a finite real scalar.") from exc
    if not np.isfinite(result):
        raise ConfigurationError(f"{name} must be a finite real scalar.")
    return result


def positive_real(value, name):
    """Return a finite, strictly positive real scalar."""
    result = finite_real(value, name)
    if result <= 0:
        raise ConfigurationError(f"{name} must be finite and positive.")
    return result


def positive_integer(value, name, minimum=1):
    """Return an exact integer at least minimum; integral floats are accepted."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
    if isinstance(value, Integral):
        result = int(value)
    else:
        numeric = finite_real(value, name)
        if not numeric.is_integer():
            raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
        result = int(numeric)
    if result < minimum:
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
    return result

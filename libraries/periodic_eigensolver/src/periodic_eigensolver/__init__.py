"""Shared refined shift-and-invert Arnoldi solver for periodic pencils."""

from .refined import (
    ArnoldiResult,
    native_backend_available,
    solve_generalized,
)

__all__ = [
    "ArnoldiResult",
    "native_backend_available",
    "solve_generalized",
]

__version__ = "1.0.0"

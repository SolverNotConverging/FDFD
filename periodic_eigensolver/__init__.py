"""Shared refined shift-and-invert Arnoldi solver for periodic pencils."""

from .refined import (
    ArnoldiResult,
    native_backend_available,
    refined_shift_invert_arnoldi,
    resolve_backend,
    resolve_kernel_backend,
    solve_generalized,
)

__all__ = [
    "ArnoldiResult",
    "native_backend_available",
    "refined_shift_invert_arnoldi",
    "resolve_backend",
    "resolve_kernel_backend",
    "solve_generalized",
]

__version__ = "0.2.0"

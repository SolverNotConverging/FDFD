"""Compatibility facade for the shared periodic eigensolver package."""

from periodic_eigensolver.refined import (
    _arnoldi_factorization,
    _default_arnoldi_ncv,
    _normalised_vector,
    _refined_candidates,
    _relative_residual_norm,
    _restart_vector,
    native_backend_available,
    refined_shift_invert_arnoldi,
    resolve_kernel_backend,
)

__all__ = [
    "native_backend_available",
    "refined_shift_invert_arnoldi",
    "resolve_kernel_backend",
]

"""Shared refined shift-and-invert Arnoldi solver for periodic pencils."""

from pathlib import Path as _Path
__path__.append(str(_Path(__file__).parent / "src"))


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

__version__ = "1.1.0"

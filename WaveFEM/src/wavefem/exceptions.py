"""WaveFEM exception hierarchy."""

from __future__ import annotations


class WaveFEMError(Exception):
    """Base class for actionable WaveFEM errors."""


class ConfigurationError(WaveFEMError):
    """The requested simulation configuration is incomplete or inconsistent."""


class MaterialError(ConfigurationError):
    """A material value or constitutive representation is invalid."""


class MeshError(WaveFEMError):
    """Mesh generation, import, or physical-region tagging failed."""


class ModeSolverError(WaveFEMError):
    """The guided-mode eigenproblem could not produce valid requested modes."""


class ModeProjectionError(WaveFEMError):
    """Fields could not be projected reliably onto the requested lead modes."""


class SolverError(WaveFEMError):
    """A finite-element linear or eigenvalue solve failed."""


__all__ = [
    "ConfigurationError",
    "MaterialError",
    "MeshError",
    "ModeProjectionError",
    "ModeSolverError",
    "SolverError",
    "WaveFEMError",
]

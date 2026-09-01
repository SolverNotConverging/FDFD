"""Finite-element electrostatic solution containers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .meshing import FEMMesh


@dataclass(frozen=True, slots=True)
class ElectrostaticResult:
    mesh: FEMMesh
    potential: NDArray[np.float64]
    electric_field: NDArray[np.float64]
    displacement_field: NDArray[np.float64]
    reaction: NDArray[np.float64]
    conductor_charges: Mapping[str, float]
    energy: float
    residual_norm: float

    def __post_init__(self) -> None:
        for name in ("potential", "electric_field", "displacement_field", "reaction"):
            values = np.asarray(getattr(self, name), dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains a non-finite FEM result.")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "conductor_charges", MappingProxyType(dict(self.conductor_charges)))

    @property
    def coordinates(self) -> NDArray[np.float64]:
        return self.mesh.nodes

    @property
    def elements(self) -> NDArray[np.int64]:
        return self.mesh.elements

    def conductor_charge(self, name: str) -> float:
        try:
            return self.conductor_charges[name]
        except KeyError as exc:
            raise KeyError(f"No fixed-potential region named {name!r} exists in this result.") from exc


__all__ = ["ElectrostaticResult"]

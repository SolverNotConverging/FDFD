"""Finite-element electrostatic solution containers."""

from __future__ import annotations

from cem_common.contracts import ResultMixin

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .meshing import FEMMesh


@dataclass(frozen=True, slots=True)
class ElectrostaticResult(ResultMixin):
    def plot(self, *, component: str | None = None, quantity: str = "real"):
        """Return a potential, field, or mesh figure without opening a window."""
        return self._result_api().plot_result(self, component=component, quantity=quantity)

    mesh: FEMMesh
    potential: NDArray[np.float64]
    electric_field: NDArray[np.float64]
    displacement_field: NDArray[np.float64]
    reaction: NDArray[np.float64]
    conductor_charges: Mapping[str, float]
    energy: float
    residual_norm: float
    element_electric_field: NDArray[np.float64] | None = None
    element_displacement_field: NDArray[np.float64] | None = None
    adaptive_history: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        for name in ("potential", "electric_field", "displacement_field", "reaction"):
            values = np.asarray(getattr(self, name), dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains a non-finite FEM result.")
            object.__setattr__(self, name, values)
        for name in ("element_electric_field", "element_displacement_field"):
            value = getattr(self, name)
            if value is None:
                continue  # Compatibility with manually constructed older results.
            values = np.asarray(value, dtype=float)
            expected = (len(self.mesh.elements), self.mesh.nodes.shape[1])
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain finite values with shape {expected}.")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "conductor_charges", MappingProxyType(dict(self.conductor_charges)))
        object.__setattr__(self, "adaptive_history", tuple(
            MappingProxyType(dict(step)) for step in self.adaptive_history
        ))

    @property
    def metadata(self):
        return MappingProxyType({"time_convention": "static", "field_locations": {"potential": "node", "electric_field": "node", "element_electric_field": "cell"}})

    @property
    def solve_info(self):
        return MappingProxyType({"algebraic_residual": self.residual_norm, "adaptive_history": self.adaptive_history, "adaptive_residual": self.adaptive_residual, "adaptive_converged": self.adaptive_converged})

    @property
    def adaptive_residual(self) -> float | None:
        return float(self.adaptive_history[-1]["residual"]) if self.adaptive_history else None

    @property
    def adaptive_converged(self) -> bool:
        return bool(self.adaptive_history and self.adaptive_history[-1]["status"] == "tolerance")

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

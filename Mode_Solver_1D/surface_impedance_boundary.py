"""Compile opaque 1D cell regions into exact scalar SIBC Yee rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceImpedanceDefinition:
    """One immutable scalar surface-impedance law at the solve frequency."""

    key: tuple
    impedance: complex
    label: str
    preset: str | None = None


@dataclass(frozen=True)
class ImpedanceAmpereRow:
    """One clipped node-centred Ampere row at an exposed interface."""

    electric_index: int
    retained_cell_index: int
    opaque_cell_index: int
    retained_dual_length: float
    relative_permittivity_yy: complex
    relative_permittivity_zz: complex
    magnetic_coefficient: float


@dataclass(frozen=True)
class CompiledImpedanceBoundary:
    """Retained Yee masks and rows replacing the ordinary Ampere derivative."""

    electric_retained: tuple[np.ndarray, np.ndarray, np.ndarray]
    magnetic_retained: tuple[np.ndarray, np.ndarray, np.ndarray]
    rows: tuple[ImpedanceAmpereRow, ...]


def _nodes_incident_to_cells(cell_mask: np.ndarray) -> np.ndarray:
    nodes = np.zeros(cell_mask.size + 1, dtype=bool)
    nodes[:-1] |= cell_mask
    nodes[1:] |= cell_mask
    return nodes


def validate_impedance_pml_separation(
        opaque_cells: np.ndarray,
        pml_cells: np.ndarray,
) -> None:
    """Reject impedance cells or their exposed interfaces in the PML."""
    opaque = np.asarray(opaque_cells, dtype=bool)
    pml = np.asarray(pml_cells, dtype=bool)
    if opaque.ndim != 1:
        raise ValueError("Surface-impedance cell mask must be one-dimensional.")
    if pml.shape != opaque.shape:
        raise ValueError("PML cell mask does not match the impedance cell mask.")

    overlap = opaque & pml
    if np.any(overlap):
        cell = int(np.flatnonzero(overlap)[0])
        raise ValueError(f"Surface-impedance cell {cell} overlaps the PML.")

    if opaque.size > 1:
        interface = opaque[:-1] ^ opaque[1:]
        touches_pml = interface & (pml[:-1] | pml[1:])
        if np.any(touches_pml):
            left = int(np.flatnonzero(touches_pml)[0])
            raise ValueError(
                "Surface-impedance interface crosses the PML between cells "
                f"{left} and {left + 1}."
            )


def _definition_impedance(
        owner_index: int,
        definitions: tuple[SurfaceImpedanceDefinition, ...],
) -> complex:
    if owner_index < 0 or owner_index >= len(definitions):
        raise ValueError(f"Invalid surface-impedance owner index {owner_index}.")
    definition = definitions[owner_index]
    impedance = complex(definition.impedance)
    if not np.isfinite(impedance) or impedance == 0:
        raise ValueError(
            f"Surface impedance {definition.label!r} is not finite and nonzero."
        )
    if impedance.real < 0:
        raise ValueError(f"Surface impedance {definition.label!r} is active.")
    return impedance


def _effective_relative_permittivity(
        bulk_value: complex,
        surface_term: complex,
) -> complex:
    value = complex(bulk_value + surface_term)
    if not np.isfinite(value):
        raise ValueError(
            "Surface-impedance boundary produced a non-finite electric coefficient."
        )
    tolerance = 512 * np.finfo(float).eps * max(
        1.0,
        abs(bulk_value),
        abs(surface_term),
    )
    if abs(value) <= tolerance:
        raise ValueError(
            "Surface-impedance boundary produced a singular electric coefficient."
        )
    return value


def compile_impedance_boundary(
        *,
        owner: np.ndarray,
        definitions: tuple[SurfaceImpedanceDefinition, ...],
        cell_eps_r_yy: np.ndarray,
        cell_eps_r_zz: np.ndarray,
        dx: float,
        frequency: float,
        epsilon0: float,
        pml_cells: np.ndarray | None = None,
) -> CompiledImpedanceBoundary:
    """Compile an opaque owner line into retained masks and clipped rows."""
    owner = np.asarray(owner, dtype=np.int32)
    if owner.ndim != 1:
        raise ValueError("Surface-impedance owner grid must be one-dimensional.")
    if np.any(owner < -1):
        raise ValueError("Surface-impedance owner entries must be -1 or model indices.")
    if np.any(owner >= len(definitions)):
        raise ValueError(
            f"Invalid surface-impedance owner index {int(np.max(owner))}."
        )

    eps_yy = np.asarray(cell_eps_r_yy, dtype=complex)
    eps_zz = np.asarray(cell_eps_r_zz, dtype=complex)
    if eps_yy.shape != owner.shape or eps_zz.shape != owner.shape:
        raise ValueError("Cell permittivity arrays do not match the impedance owner grid.")
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("Surface-impedance grid spacing must be finite and positive.")
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency must be finite and positive for surface impedance.")
    if not np.isfinite(epsilon0) or epsilon0 <= 0:
        raise ValueError("epsilon0 must be finite and positive.")

    opaque = owner >= 0
    retained_cells = ~opaque
    if not np.any(retained_cells):
        raise ValueError("Surface-impedance geometry leaves no retained field cells.")

    if pml_cells is None:
        pml_cells = np.zeros(owner.shape, dtype=bool)
    validate_impedance_pml_separation(opaque, pml_cells)

    retained_nodes = _nodes_incident_to_cells(retained_cells)
    electric_retained = (
        retained_cells.copy(),
        retained_nodes.copy(),
        retained_nodes.copy(),
    )
    magnetic_retained = (
        retained_nodes.copy(),
        retained_cells.copy(),
        retained_cells.copy(),
    )

    angular_frequency = 2 * np.pi * frequency
    dual_length = 0.5 * dx
    rows: list[ImpedanceAmpereRow] = []
    for node in range(1, owner.size):
        left_opaque = opaque[node - 1]
        right_opaque = opaque[node]
        if left_opaque == right_opaque:
            continue

        if left_opaque:
            opaque_cell = node - 1
            retained_cell = node
            magnetic_coefficient = 1.0 / dual_length
        else:
            retained_cell = node - 1
            opaque_cell = node
            magnetic_coefficient = -1.0 / dual_length

        impedance = _definition_impedance(
            int(owner[opaque_cell]),
            definitions,
        )
        surface_term = 1.0 / (
            1j * angular_frequency * epsilon0 * dual_length * impedance
        )
        rows.append(
            ImpedanceAmpereRow(
                electric_index=node,
                retained_cell_index=retained_cell,
                opaque_cell_index=opaque_cell,
                retained_dual_length=dual_length,
                relative_permittivity_yy=_effective_relative_permittivity(
                    eps_yy[retained_cell],
                    surface_term,
                ),
                relative_permittivity_zz=_effective_relative_permittivity(
                    eps_zz[retained_cell],
                    surface_term,
                ),
                magnetic_coefficient=magnetic_coefficient,
            )
        )

    return CompiledImpedanceBoundary(
        electric_retained=electric_retained,
        magnetic_retained=magnetic_retained,
        rows=tuple(rows),
    )


__all__ = [
    "CompiledImpedanceBoundary",
    "ImpedanceAmpereRow",
    "SurfaceImpedanceDefinition",
    "compile_impedance_boundary",
    "validate_impedance_pml_separation",
]

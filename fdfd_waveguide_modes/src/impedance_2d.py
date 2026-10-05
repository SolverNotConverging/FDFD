"""Cell-topology compiler for opaque scalar surface-impedance boundaries.

The public geometry API marks cell-centred opaque regions.  This module turns
the final ownership grid into independent retained Yee-component masks and
integral Ampere rows on the exposed boundary.  No artificial conductor
permittivity or film thickness is introduced.
"""

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
class MagneticLineTerm:
    """One signed magnetic line contribution to an integral Ampere row."""

    component: int
    index: tuple[int, int]
    length: float


@dataclass(frozen=True)
class ImpedanceAmpereRow:
    """One clipped electric row on an exposed impedance boundary."""

    electric_component: int
    electric_index: tuple[int, int]
    retained_dual_area: float
    relative_permittivity: complex
    magnetic_terms: tuple[MagneticLineTerm, ...]


@dataclass(frozen=True)
class CompiledImpedanceBoundary:
    """Retained Yee masks and the rows replacing ordinary Ampere curls."""

    electric_retained: tuple[np.ndarray, np.ndarray, np.ndarray]
    magnetic_retained: tuple[np.ndarray, np.ndarray, np.ndarray]
    rows: tuple[ImpedanceAmpereRow, ...]


def _component_incidence_masks(cell_mask: np.ndarray):
    """Return component masks having at least one incident marked cell."""

    nx, ny = cell_mask.shape

    on_ex = np.zeros((nx, ny + 1), dtype=bool)
    on_ex[:, :ny] |= cell_mask
    on_ex[:, 1:] |= cell_mask

    on_ey = np.zeros((nx + 1, ny), dtype=bool)
    on_ey[:nx, :] |= cell_mask
    on_ey[1:, :] |= cell_mask

    on_ez = np.zeros((nx + 1, ny + 1), dtype=bool)
    on_ez[:nx, :ny] |= cell_mask
    on_ez[1:, :ny] |= cell_mask
    on_ez[:nx, 1:] |= cell_mask
    on_ez[1:, 1:] |= cell_mask
    return on_ex, on_ey, on_ez


def _validate_topology(opaque: np.ndarray) -> None:
    """Reject diagonal-only contacts at a longitudinal Yee edge."""

    if opaque.shape[0] < 2 or opaque.shape[1] < 2:
        return

    lower_left = opaque[:-1, :-1]
    lower_right = opaque[1:, :-1]
    upper_left = opaque[:-1, 1:]
    upper_right = opaque[1:, 1:]
    diagonal = (
        lower_left & upper_right & ~lower_right & ~upper_left
    ) | (
        lower_right & upper_left & ~lower_left & ~upper_right
    )
    if np.any(diagonal):
        i, j = np.argwhere(diagonal)[0]
        raise ValueError(
            "Surface-impedance topology is non-manifold at Ez edge "
            f"({i + 1}, {j + 1}): opaque cells touch only diagonally."
        )


def _validate_pml_separation(opaque: np.ndarray, pml_cells: np.ndarray) -> None:
    """Reject impedance cells and exposed interfaces in a PML."""

    if np.any(opaque & pml_cells):
        i, j = np.argwhere(opaque & pml_cells)[0]
        raise ValueError(
            f"Surface-impedance cell ({i}, {j}) overlaps the PML."
        )

    if opaque.shape[0] > 1:
        interface = opaque[:-1, :] ^ opaque[1:, :]
        touches_pml = interface & (pml_cells[:-1, :] | pml_cells[1:, :])
        if np.any(touches_pml):
            i, j = np.argwhere(touches_pml)[0]
            raise ValueError(
                "Surface-impedance interface crosses the PML between cells "
                f"({i}, {j}) and ({i + 1}, {j})."
            )

    if opaque.shape[1] > 1:
        interface = opaque[:, :-1] ^ opaque[:, 1:]
        touches_pml = interface & (pml_cells[:, :-1] | pml_cells[:, 1:])
        if np.any(touches_pml):
            i, j = np.argwhere(touches_pml)[0]
            raise ValueError(
                "Surface-impedance interface crosses the PML between cells "
                f"({i}, {j}) and ({i}, {j + 1})."
            )


def validate_impedance_pml_separation(
    opaque_cells: np.ndarray,
    pml_cells: np.ndarray,
) -> None:
    """Reject overlap or a shared interface between impedance cells and PML."""

    opaque = np.asarray(opaque_cells, dtype=bool)
    pml = np.asarray(pml_cells, dtype=bool)
    if opaque.ndim != 2:
        raise ValueError("Surface-impedance cell mask must be two-dimensional.")
    if pml.shape != opaque.shape:
        raise ValueError("PML cell mask does not match the impedance cell mask.")
    _validate_pml_separation(opaque, pml)


def _effective_relative_permittivity(
    *,
    relative_mass: complex,
    area: float,
    ports: list[tuple[float, complex]],
    angular_frequency: float,
    epsilon0: float,
) -> complex:
    """Map an integral SIBC Ampere load to the solver's electric coefficient."""

    surface_load = sum(length / impedance for length, impedance in ports)
    bulk_term = relative_mass / area
    surface_term = surface_load / (1j * angular_frequency * epsilon0 * area)
    value = bulk_term + surface_term
    if not np.isfinite(value):
        raise ValueError("Surface-impedance boundary produced a non-finite electric coefficient.")
    tolerance = 512 * np.finfo(float).eps * max(
        1.0,
        abs(bulk_term),
        abs(surface_term),
    )
    if abs(value) <= tolerance:
        raise ValueError(
            "Surface-impedance boundary produced a singular electric coefficient."
        )
    return complex(value)


def _definition_impedance(
    owner_index: int,
    definitions: tuple[SurfaceImpedanceDefinition, ...],
) -> complex:
    if owner_index < 0 or owner_index >= len(definitions):
        raise ValueError(f"Invalid surface-impedance owner index {owner_index}.")
    value = complex(definitions[owner_index].impedance)
    if not np.isfinite(value) or value == 0:
        raise ValueError(
            f"Surface impedance {definitions[owner_index].label!r} is not finite and nonzero."
        )
    if value.real < 0:
        raise ValueError(
            f"Surface impedance {definitions[owner_index].label!r} is active."
        )
    return value


def compile_impedance_boundary(
    *,
    owner: np.ndarray,
    definitions: tuple[SurfaceImpedanceDefinition, ...],
    cell_eps_r_xx: np.ndarray,
    cell_eps_r_yy: np.ndarray,
    cell_eps_r_zz: np.ndarray,
    dx: float,
    dy: float,
    frequency: float,
    epsilon0: float,
    pml_cells: np.ndarray | None = None,
) -> CompiledImpedanceBoundary:
    """Compile an opaque cell-owner map into exact scalar SIBC Yee rows."""

    owner = np.asarray(owner, dtype=np.int32)
    if owner.ndim != 2:
        raise ValueError("Surface-impedance owner grid must be two-dimensional.")
    nx, ny = owner.shape
    if np.any(owner < -1):
        raise ValueError("Surface-impedance owner entries must be -1 or model indices.")
    if np.any(owner >= len(definitions)):
        invalid = int(np.max(owner))
        raise ValueError(f"Invalid surface-impedance owner index {invalid}.")
    expected_shape = (nx, ny)
    eps_xx = np.asarray(cell_eps_r_xx, dtype=complex)
    eps_yy = np.asarray(cell_eps_r_yy, dtype=complex)
    eps_zz = np.asarray(cell_eps_r_zz, dtype=complex)
    if any(values.shape != expected_shape for values in (eps_xx, eps_yy, eps_zz)):
        raise ValueError("Cell permittivity arrays do not match the impedance owner grid.")
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency must be finite and positive for surface impedance.")
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(dy) or dy <= 0:
        raise ValueError("Surface-impedance grid spacings must be finite and positive.")
    if not np.isfinite(epsilon0) or epsilon0 <= 0:
        raise ValueError("epsilon0 must be finite and positive.")

    opaque = owner >= 0
    retained_cells = ~opaque
    if not np.any(retained_cells):
        raise ValueError("Surface-impedance geometry leaves no retained field cells.")
    _validate_topology(opaque)

    if pml_cells is None:
        pml_cells = np.zeros(expected_shape, dtype=bool)
    else:
        pml_cells = np.asarray(pml_cells, dtype=bool)
        if pml_cells.shape != expected_shape:
            raise ValueError("PML cell mask does not match the impedance owner grid.")
    validate_impedance_pml_separation(opaque, pml_cells)

    free_ex, free_ey, free_ez = _component_incidence_masks(retained_cells)
    solid_ex, solid_ey, solid_ez = _component_incidence_masks(opaque)
    boundary_ex = free_ex & solid_ex
    boundary_ey = free_ey & solid_ey
    boundary_ez = free_ez & solid_ez

    electric_retained = (free_ex, free_ey, free_ez)
    magnetic_retained = (free_ey.copy(), free_ex.copy(), retained_cells.copy())
    rows: list[ImpedanceAmpereRow] = []
    angular_frequency = 2 * np.pi * frequency

    # Ex lies between the cell below and the cell above.  The arbitrary unit
    # propagation length cancels between the dual area and surface port.
    for i, j in np.argwhere(boundary_ex):
        incident = []
        if j > 0:
            incident.append((i, j - 1, "below"))
        if j < ny:
            incident.append((i, j, "above"))
        free = [cell for cell in incident if owner[cell[0], cell[1]] < 0]
        solid = [cell for cell in incident if owner[cell[0], cell[1]] >= 0]
        if len(free) != 1 or len(solid) != 1:
            raise ValueError(f"Invalid Ex impedance topology at ({i}, {j}).")
        fi, fj, side = free[0]
        oi, oj, _ = solid[0]
        area = 0.5 * dy
        impedance = _definition_impedance(int(owner[oi, oj]), definitions)
        relative_permittivity = _effective_relative_permittivity(
            relative_mass=eps_xx[fi, fj] * area,
            area=area,
            ports=[(1.0, impedance)],
            angular_frequency=angular_frequency,
            epsilon0=epsilon0,
        )
        line_sign = 1.0 if side == "above" else -1.0
        rows.append(
            ImpedanceAmpereRow(
                0,
                (int(i), int(j)),
                area,
                relative_permittivity,
                (MagneticLineTerm(2, (int(fi), int(fj)), line_sign),),
            )
        )

    # Ey lies between the cell to the left and the cell to the right.
    for i, j in np.argwhere(boundary_ey):
        incident = []
        if i > 0:
            incident.append((i - 1, j, "left"))
        if i < nx:
            incident.append((i, j, "right"))
        free = [cell for cell in incident if owner[cell[0], cell[1]] < 0]
        solid = [cell for cell in incident if owner[cell[0], cell[1]] >= 0]
        if len(free) != 1 or len(solid) != 1:
            raise ValueError(f"Invalid Ey impedance topology at ({i}, {j}).")
        fi, fj, side = free[0]
        oi, oj, _ = solid[0]
        area = 0.5 * dx
        impedance = _definition_impedance(int(owner[oi, oj]), definitions)
        relative_permittivity = _effective_relative_permittivity(
            relative_mass=eps_yy[fi, fj] * area,
            area=area,
            ports=[(1.0, impedance)],
            angular_frequency=angular_frequency,
            epsilon0=epsilon0,
        )
        line_sign = 1.0 if side == "left" else -1.0
        rows.append(
            ImpedanceAmpereRow(
                1,
                (int(i), int(j)),
                area,
                relative_permittivity,
                (MagneticLineTerm(2, (int(fi), int(fj)), line_sign),),
            )
        )

    quarter_area = 0.25 * dx * dy
    for i, j in np.argwhere(boundary_ez):
        quadrants: dict[tuple[int, int], tuple[int, int, int]] = {}
        for di in (-1, 0):
            for dj in (-1, 0):
                ci, cj = int(i + di), int(j + dj)
                if 0 <= ci < nx and 0 <= cj < ny:
                    quadrants[(di, dj)] = (ci, cj, int(owner[ci, cj]))

        free_quadrants = [cell for cell in quadrants.values() if cell[2] < 0]
        if not free_quadrants:
            raise ValueError(f"Invalid Ez impedance topology at ({i}, {j}).")

        term_weights: dict[tuple[int, tuple[int, int]], float] = {}

        def add_term(component: int, index: tuple[int, int], length: float) -> None:
            key = (component, index)
            term_weights[key] = term_weights.get(key, 0.0) + length

        relative_mass = 0.0 + 0.0j
        for di, dj in quadrants:
            ci, cj, cell_owner = quadrants[(di, dj)]
            if cell_owner >= 0:
                continue
            relative_mass += eps_zz[ci, cj] * quarter_area
            add_term(1, (ci, int(j)), 0.5 * dy if di == 0 else -0.5 * dy)
            add_term(0, (int(i), cj), -0.5 * dx if dj == 0 else 0.5 * dx)

        ports: list[tuple[float, complex]] = []

        def add_port(first: tuple[int, int], second: tuple[int, int], length: float) -> None:
            if first not in quadrants or second not in quadrants:
                return
            first_owner = quadrants[first][2]
            second_owner = quadrants[second][2]
            if (first_owner < 0) == (second_owner < 0):
                return
            opaque_owner = second_owner if first_owner < 0 else first_owner
            ports.append((length, _definition_impedance(opaque_owner, definitions)))

        add_port((-1, -1), (0, -1), 0.5 * dy)
        add_port((-1, 0), (0, 0), 0.5 * dy)
        add_port((-1, -1), (-1, 0), 0.5 * dx)
        add_port((0, -1), (0, 0), 0.5 * dx)
        if not ports:
            raise ValueError(f"Ez impedance edge ({i}, {j}) has no exposed face segment.")

        area = len(free_quadrants) * quarter_area
        relative_permittivity = _effective_relative_permittivity(
            relative_mass=relative_mass,
            area=area,
            ports=ports,
            angular_frequency=angular_frequency,
            epsilon0=epsilon0,
        )
        terms = tuple(
            MagneticLineTerm(component, index, length)
            for (component, index), length in sorted(term_weights.items())
            if length != 0.0
        )
        if not terms:
            raise ValueError(f"Ez impedance edge ({i}, {j}) has no magnetic circulation.")
        rows.append(
            ImpedanceAmpereRow(
                2,
                (int(i), int(j)),
                area,
                relative_permittivity,
                terms,
            )
        )

    return CompiledImpedanceBoundary(
        electric_retained=tuple(mask.copy() for mask in electric_retained),
        magnetic_retained=tuple(mask.copy() for mask in magnetic_retained),
        rows=tuple(rows),
    )


__all__ = [
    "CompiledImpedanceBoundary",
    "ImpedanceAmpereRow",
    "MagneticLineTerm",
    "SurfaceImpedanceDefinition",
    "compile_impedance_boundary",
    "validate_impedance_pml_separation",
]

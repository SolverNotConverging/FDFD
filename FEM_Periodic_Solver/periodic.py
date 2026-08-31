"""Sparse equality constraints for periodic finite-element envelopes.

The propagation phase is carried analytically by the shifted curl/derivative,
so all seam constraints in this module are plain equality constraints.  The
edge helper additionally accounts for the orientation of first-kind Nedelec
degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, spmatrix

from .exceptions import ConfigurationError

IntArray = NDArray[np.int64]


def _indices(values: ArrayLike, size: int, name: str) -> IntArray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind == "b":
        raise ConfigurationError(f"{name} must be a one-dimensional integer array.")
    try:
        numeric = np.asarray(raw, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must contain integers.") from exc
    if not np.isfinite(numeric).all() or np.any(numeric.imag != 0.0) or np.any(numeric.real != np.floor(numeric.real)):
        raise ConfigurationError(f"{name} must contain finite integers.")
    result = np.asarray(numeric.real, dtype=np.int64)
    if result.size and (result.min() < 0 or result.max() >= size):
        raise ConfigurationError(f"{name} contains an index outside 0..{size - 1}.")
    return result


def node_representatives(
    node_count: int,
    slave_nodes: ArrayLike,
    master_nodes: ArrayLike,
) -> IntArray:
    """Return the master-root representative of every mesh node."""

    if isinstance(node_count, bool) or int(node_count) != node_count or node_count < 1:
        raise ConfigurationError("node_count must be a positive integer.")
    count = int(node_count)
    slaves = _indices(slave_nodes, count, "slave_nodes")
    masters = _indices(master_nodes, count, "master_nodes")
    if slaves.shape != masters.shape:
        raise ConfigurationError("slave_nodes and master_nodes must have the same shape.")
    if np.unique(slaves).size != slaves.size:
        raise ConfigurationError("Each periodic slave node may occur only once.")
    if np.any(slaves == masters):
        raise ConfigurationError("A periodic node cannot be paired with itself.")

    parent = np.arange(count, dtype=np.int64)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            following = int(parent[value])
            parent[value] = root
            value = following
        return root

    # Direct each equivalence class toward the nominated master.  This is
    # deterministic and preserves Gmsh's master/slave convention.
    for slave, master in zip(slaves, masters, strict=True):
        slave_root = find(int(slave))
        master_root = find(int(master))
        if slave_root != master_root:
            parent[slave_root] = master_root
    representatives = np.fromiter((find(index) for index in range(count)), dtype=np.int64, count=count)
    representatives.setflags(write=False)
    return representatives


@dataclass(frozen=True, slots=True)
class PeriodicProlongation:
    """A sparse map from independent coefficients to full mesh coefficients."""

    matrix: csr_matrix
    representatives: IntArray
    independent_representatives: IntArray
    signs: NDArray[np.int8]

    @property
    def full_size(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def reduced_size(self) -> int:
        return int(self.matrix.shape[1])

    def expand(self, coefficients: ArrayLike) -> NDArray[np.complex128]:
        values = np.asarray(coefficients, dtype=np.complex128)
        if values.shape != (self.reduced_size,):
            raise ValueError(
                f"coefficients must have shape ({self.reduced_size},); received {values.shape}."
            )
        return np.asarray(self.matrix @ values, dtype=np.complex128)

    def reduce_matrix(self, matrix: spmatrix) -> csr_matrix:
        if matrix.shape != (self.full_size, self.full_size):
            raise ValueError(
                f"matrix must have shape ({self.full_size}, {self.full_size}); received {matrix.shape}."
            )
        return (self.matrix.getH() @ matrix @ self.matrix).tocsr()

    def equality_error(self, full_coefficients: ArrayLike) -> float:
        values = np.asarray(full_coefficients, dtype=np.complex128)
        if values.shape != (self.full_size,):
            raise ValueError(f"full_coefficients must have shape ({self.full_size},).")
        error = 0.0
        scale = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
        for index, representative in enumerate(self.representatives):
            error = max(
                error,
                float(abs(values[index] - int(self.signs[index]) * values[int(representative)])),
            )
        return error / scale


def build_node_prolongation(
    node_count: int,
    slave_nodes: ArrayLike,
    master_nodes: ArrayLike,
    *,
    constrained_nodes: ArrayLike = (),
) -> PeriodicProlongation:
    """Build an unsigned scalar-P1 prolongation for periodic node pairs."""

    representatives = node_representatives(node_count, slave_nodes, master_nodes)
    constrained = _indices(constrained_nodes, int(node_count), "constrained_nodes")
    constrained_roots = set(int(representatives[index]) for index in constrained)
    independent = np.asarray(
        sorted(set(int(value) for value in representatives) - constrained_roots),
        dtype=np.int64,
    )
    if independent.size == 0:
        raise ConfigurationError("Periodic and boundary constraints leave no independent nodes.")
    columns = {int(root): column for column, root in enumerate(independent)}
    rows: list[int] = []
    column_indices: list[int] = []
    for row, root in enumerate(representatives):
        column = columns.get(int(root))
        if column is not None:
            rows.append(row)
            column_indices.append(column)
    data = np.ones(len(rows), dtype=np.complex128)
    matrix = csr_matrix(
        (data, (np.asarray(rows), np.asarray(column_indices))),
        shape=(int(node_count), independent.size),
        dtype=np.complex128,
    )
    signs = np.ones(int(node_count), dtype=np.int8)
    signs.setflags(write=False)
    independent.setflags(write=False)
    return PeriodicProlongation(matrix, representatives, independent, signs)


def build_signed_edge_prolongation(
    edges: ArrayLike,
    slave_nodes: ArrayLike,
    master_nodes: ArrayLike,
    *,
    node_count: int | None = None,
    constrained_edges: ArrayLike = (),
) -> PeriodicProlongation:
    """Build a signed first-kind Nedelec edge prolongation.

    ``edges`` accepts either ``(2, E)`` (scikit-fem convention) or ``(E, 2)``.
    An edge is identified with the master trace only when both endpoints occur
    in ``slave_nodes``; incident volume edges stay independent.  Reversing the
    mapped endpoints relative to the actual master-edge orientation contributes
    ``-1`` to the prolongation row.
    """

    raw = np.asarray(edges)
    if raw.ndim != 2 or 2 not in raw.shape:
        raise ConfigurationError("edges must have shape (2, E) or (E, 2).")
    oriented = raw.T if raw.shape[0] == 2 else raw
    if oriented.shape[1] != 2:
        raise ConfigurationError("edges must contain exactly two endpoints.")
    if node_count is None:
        if oriented.size == 0:
            raise ConfigurationError("node_count is required for an empty edge array.")
        node_count = int(np.max(oriented)) + 1
    endpoint_values = _indices(oriented.ravel(), int(node_count), "edges").reshape(-1, 2)
    if np.any(endpoint_values[:, 0] == endpoint_values[:, 1]):
        raise ConfigurationError("edges may not have identical endpoints.")
    slaves = _indices(slave_nodes, int(node_count), "slave_nodes")
    masters = _indices(master_nodes, int(node_count), "master_nodes")
    if slaves.shape != masters.shape:
        raise ConfigurationError("slave_nodes and master_nodes must have the same shape.")
    if np.unique(slaves).size != slaves.size:
        raise ConfigurationError("Each periodic slave node may occur only once.")
    slave_to_master = {
        int(slave): int(master)
        for slave, master in zip(slaves, masters, strict=True)
    }

    # Use the actual edge orientation supplied by the mesh, not sorted node
    # numbers, as the sign reference.  Only trace edges (both endpoints on the
    # slave face) are periodic.  Mapping an incident volume edge with one
    # slave endpoint would incorrectly constrain an interior Nedelec DOF.
    edge_lookup: dict[tuple[int, int], int] = {}
    for index, (first, second) in enumerate(endpoint_values):
        key = tuple(sorted((int(first), int(second))))
        if key in edge_lookup:
            raise ConfigurationError("edges contains duplicate endpoint pairs.")
        edge_lookup[key] = index

    direct_roots = np.arange(endpoint_values.shape[0], dtype=np.int64)
    direct_signs = np.ones(endpoint_values.shape[0], dtype=np.int8)
    for index, (first_raw, second_raw) in enumerate(endpoint_values):
        first, second = int(first_raw), int(second_raw)
        if first not in slave_to_master or second not in slave_to_master:
            continue
        mapped = (slave_to_master[first], slave_to_master[second])
        if mapped[0] == mapped[1]:
            raise ConfigurationError("Periodic node pairing collapsed a trace edge.")
        try:
            master_index = edge_lookup[tuple(sorted(mapped))]
        except KeyError as exc:
            raise ConfigurationError(
                "A slave trace edge has no corresponding master edge in the mesh."
            ) from exc
        master_edge = tuple(int(value) for value in endpoint_values[master_index])
        if mapped == master_edge:
            orientation = 1
        elif mapped == master_edge[::-1]:
            orientation = -1
        else:  # pragma: no cover - protected by the unordered lookup
            raise ConfigurationError("The mapped master edge endpoints are inconsistent.")
        direct_roots[index] = master_index
        direct_signs[index] = orientation

    # Resolve possible chains (for example, edges on intersecting periodic
    # faces in a future multi-axis extension) while accumulating orientation.
    edge_roots = np.empty(endpoint_values.shape[0], dtype=np.int64)
    edge_signs = np.empty(endpoint_values.shape[0], dtype=np.int8)
    for index in range(endpoint_values.shape[0]):
        root = index
        sign = 1
        visited: set[int] = set()
        while int(direct_roots[root]) != root:
            if root in visited:
                raise ConfigurationError("Periodic edge mapping contains a cycle.")
            visited.add(root)
            sign *= int(direct_signs[root])
            root = int(direct_roots[root])
        edge_roots[index] = root
        edge_signs[index] = sign

    constrained = _indices(constrained_edges, len(endpoint_values), "constrained_edges")
    constrained_roots = set(int(edge_roots[index]) for index in constrained)
    independent = np.asarray(
        sorted(set(int(root) for root in edge_roots) - constrained_roots),
        dtype=np.int64,
    )
    if independent.size == 0:
        raise ConfigurationError("Periodic and boundary constraints leave no independent edges.")
    columns = {int(root): column for column, root in enumerate(independent)}
    rows: list[int] = []
    column_indices: list[int] = []
    data: list[complex] = []
    for row, root in enumerate(edge_roots):
        column = columns.get(int(root))
        if column is not None:
            rows.append(row)
            column_indices.append(column)
            data.append(complex(int(edge_signs[row])))
    matrix = csr_matrix(
        (np.asarray(data, dtype=np.complex128), (np.asarray(rows), np.asarray(column_indices))),
        shape=(len(endpoint_values), independent.size),
        dtype=np.complex128,
    )
    edge_roots.setflags(write=False)
    edge_signs.setflags(write=False)
    independent.setflags(write=False)
    return PeriodicProlongation(matrix, edge_roots, independent, edge_signs)


__all__ = [
    "PeriodicProlongation",
    "build_node_prolongation",
    "build_signed_edge_prolongation",
    "node_representatives",
]

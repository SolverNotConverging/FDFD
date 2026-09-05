"""Versioned, lazy HDF5 persistence for periodic FEM modes.

The schema is intentionally independent of Python object pickling.  Its index is
small enough to open eagerly; meshes, material states, coefficient vectors, and
field hyperslabs are read only when a case or mode is requested.
"""

from __future__ import annotations

from fem_common import MeshSnapshot
from fem_common.persistence import write_envelope, validate_envelope, write_value, read_value

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
from uuid import uuid4

import h5py
import numpy as np
from numpy.typing import NDArray

from .constants import C_0
from .exceptions import PersistenceError
from .results import PeriodicMode, PeriodicModeSet, PeriodicSampledFields


SCHEMA_FORMAT = "cem-fem-results"
SCHEMA_MAJOR = 1
SCHEMA_MINOR = 0
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
_FILTERS = {
    "compression": "gzip",
    "compression_opts": 4,
    "shuffle": True,
    "fletcher32": True,
}


@dataclass(frozen=True, slots=True)
class H5ValidationReport:
    """Summary returned after a successful schema validation."""

    path: Path
    schema_major: int
    schema_minor: int
    case_count: int
    mode_count: int
    deep: bool

    def __bool__(self) -> bool:
        return True


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _scalar_text_attribute(group: h5py.Group, name: str) -> str:
    """Read a scalar HDF5 string attribute without coercing numeric values."""

    if name not in group.attrs:
        raise PersistenceError(f"{group.name} is missing text attribute {name!r}.")
    raw = np.asarray(group.attrs[name])
    if raw.shape != ():
        raise PersistenceError(f"{group.name}/{name} attribute must be scalar text.")
    value = raw.item()
    if not isinstance(value, (str, bytes, np.str_, np.bytes_)):
        raise PersistenceError(f"{group.name}/{name} attribute must use an HDF5 string type.")
    return _text(value)


def _read_complex(dataset: h5py.Dataset | NDArray[Any]) -> NDArray[np.complex128]:
    raw = np.asarray(dataset[...] if isinstance(dataset, h5py.Dataset) else dataset)
    name = dataset.name if isinstance(dataset, h5py.Dataset) else "selected complex data"
    if raw.dtype.kind == "c" and raw.dtype.itemsize == np.dtype(np.complex128).itemsize:
        result = np.asarray(raw, dtype=np.complex128)
    elif raw.dtype.fields is not None and set(raw.dtype.fields) == {"r", "i"}:
        member_dtypes = [raw.dtype.fields[member][0] for member in ("r", "i")]
        if any(dtype.kind != "f" or dtype.itemsize != 8 for dtype in member_dtypes):
            raise PersistenceError(
                f"{name} compound complex members r/i must both be float64."
            )
        result = np.asarray(raw["r"], dtype=np.float64) + 1j * np.asarray(
            raw["i"], dtype=np.float64
        )
    else:
        raise PersistenceError(
            f"{name} must use complex128 or an exact compound {{r, i}} encoding."
        )
    if not np.isfinite(result).all():
        raise PersistenceError(f"{name} contains a non-finite complex value.")
    return result


def _read_real(dataset: h5py.Dataset) -> NDArray[np.float64]:
    raw = np.asarray(dataset[...])
    if raw.dtype.kind != "f":
        raise PersistenceError(f"{dataset.name} must contain floating-point values.")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise PersistenceError(f"{dataset.name} contains a non-finite value.")
    return result


def _read_int(dataset: h5py.Dataset) -> NDArray[np.int64]:
    raw = np.asarray(dataset[...])
    if not np.issubdtype(raw.dtype, np.integer):
        raise PersistenceError(f"{dataset.name} must contain integers.")
    return np.asarray(raw, dtype=np.int64)


def _read_strings(dataset: h5py.Dataset) -> tuple[str, ...]:
    raw = np.asarray(dataset.asstr()[...])
    if raw.ndim != 1:
        raise PersistenceError(f"{dataset.name} must be one-dimensional text.")
    return tuple(str(item) for item in raw.tolist())


def _require_group(parent: h5py.Group | h5py.File, name: str) -> h5py.Group:
    if name not in parent or not isinstance(parent[name], h5py.Group):
        raise PersistenceError(f"Missing HDF5 group {parent.name.rstrip('/')}/{name}.")
    return parent[name]


def _require_dataset(parent: h5py.Group, name: str) -> h5py.Dataset:
    if name not in parent or not isinstance(parent[name], h5py.Dataset):
        raise PersistenceError(f"Missing HDF5 dataset {parent.name}/{name}.")
    return parent[name]


def _write_text(group: h5py.Group, name: str, values: Sequence[str]) -> h5py.Dataset:
    return group.create_dataset(name, data=np.asarray(tuple(values), dtype=object), dtype=_STRING_DTYPE)


def _write_filtered(
    group: h5py.Group,
    name: str,
    values: Any,
    *,
    chunks: tuple[int, ...] | None = None,
) -> h5py.Dataset:
    data = np.asarray(values)
    if data.ndim == 0 or any(length == 0 for length in data.shape):
        return group.create_dataset(name, data=data)
    if chunks is None:
        chunks = tuple(max(1, min(int(length), 16_384)) for length in data.shape)
    return group.create_dataset(name, data=data, chunks=chunks, **_FILTERS)


def _mode_chunks(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    if len(shape) == 2:
        # 49,152 complex128 values are 768 KiB.
        return (1, min(shape[1], 49_152))
    if len(shape) == 3:
        # 16,384 complex three-vectors are 768 KiB.
        return (1, min(shape[1], 16_384), shape[2])
    raise PersistenceError(f"Cannot construct mode-first chunks for shape {shape!r}.")


def _pad_points(points: NDArray[np.float64], dimension: int) -> NDArray[np.float64]:
    points = np.asarray(points, dtype=np.float64)
    if dimension == 3:
        if points.ndim != 2 or points.shape[1] != 3:
            raise PersistenceError("3D points must have shape (N, 3).")
        return np.ascontiguousarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise PersistenceError("2D points must have shape (N, 2) in (x, z) order.")
    result = np.zeros((points.shape[0], 3), dtype=np.float64)
    result[:, 0] = points[:, 0]
    result[:, 2] = points[:, 1]
    return result


def _metadata_array(
    mode_set: PeriodicModeSet,
    fields: PeriodicSampledFields,
    key: str,
    dtype: Any,
) -> NDArray[Any] | None:
    value = fields.metadata.get(key, mode_set.metadata.get(key))
    return None if value is None else np.asarray(value, dtype=dtype)


def _cell_materials(
    mode_set: PeriodicModeSet,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64]]:
    fields = mode_set[0].fields
    cells = fields.mesh_cells.shape[0]
    epsilon = _metadata_array(mode_set, fields, "cell_epsilon_r", np.complex128)
    mu = _metadata_array(mode_set, fields, "cell_mu_r", np.complex128)
    pml = _metadata_array(mode_set, fields, "cell_pml_fraction", np.float64)

    if epsilon is None:
        epsilon = np.ones((cells, 3), dtype=np.complex128)
        if fields.material is not None:
            counts = np.bincount(fields.sample_element_indices, minlength=cells)
            sample = np.asarray(fields.material, dtype=np.complex128)
            if sample.ndim == 1:
                sample = np.repeat(sample[:, None], 3, axis=1)
            if sample.shape != (fields.coordinates.shape[0], 3):
                raise PersistenceError("Sample material data must have shape (Nsamples, 3).")
            for component in range(3):
                sums = np.bincount(
                    fields.sample_element_indices,
                    weights=sample[:, component].real,
                    minlength=cells,
                ) + 1j * np.bincount(
                    fields.sample_element_indices,
                    weights=sample[:, component].imag,
                    minlength=cells,
                )
                populated = counts > 0
                epsilon[populated, component] = sums[populated] / counts[populated]
    if mu is None:
        mu = np.ones((cells, 3), dtype=np.complex128)
    if pml is None:
        pml = np.zeros(cells, dtype=np.float64)

    epsilon = np.asarray(epsilon, dtype=np.complex128)
    mu = np.asarray(mu, dtype=np.complex128)
    pml = np.asarray(pml, dtype=np.float64)
    if epsilon.shape != (cells, 3) or mu.shape != (cells, 3) or pml.shape != (cells,):
        raise PersistenceError("Cell epsilon, mu, and PML arrays have inconsistent shapes.")
    if not np.isfinite(epsilon).all() or not np.isfinite(mu).all() or not np.isfinite(pml).all():
        raise PersistenceError("Cell material state contains a non-finite value.")
    if np.any((pml < 0.0) | (pml > 1.0)):
        raise PersistenceError("Cell PML fractions must lie in [0, 1].")
    return epsilon, mu, pml


def _field_vectors(mode: PeriodicMode) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    samples = mode.fields.coordinates.shape[0]
    zeros = np.zeros(samples, dtype=np.complex128)
    electric = np.column_stack(
        [np.asarray(mode.fields.values.get(name, zeros)) for name in ("Ex", "Ey", "Ez")]
    )
    magnetic = np.column_stack(
        [np.asarray(mode.fields.values.get(name, zeros)) for name in ("Hx", "Hy", "Hz")]
    )
    return np.asarray(electric, dtype=np.complex128), np.asarray(magnetic, dtype=np.complex128)


def _hash_record(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.view(np.uint8))
        else:
            digest.update(repr(value).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _validate_3d_edge_topology(
    points: NDArray[np.float64],
    cells: NDArray[np.int64],
    optional: dict[str, NDArray[Any]],
) -> None:
    edge_nodes = optional.get("edge_nodes")
    cell_edges = optional.get("cell_edges")
    cell_edge_signs = optional.get("cell_edge_signs")
    if edge_nodes is None or edge_nodes.ndim != 2 or edge_nodes.shape[1] != 2:
        raise PersistenceError("3D archives require canonical edge_nodes with shape (N, 2).")
    if cell_edges is None or cell_edges.shape != (cells.shape[0], 6):
        raise PersistenceError("3D archives require cell_edges with shape (N_tetrahedra, 6).")
    if cell_edge_signs is None or cell_edge_signs.shape != cell_edges.shape:
        raise PersistenceError("3D archives require cell_edge_signs matching cell_edges.")
    if np.any(edge_nodes[:, 0] >= edge_nodes[:, 1]):
        raise PersistenceError("edge_nodes must use strict ascending canonical orientation.")
    if edge_nodes.size and (np.any(edge_nodes < 0) or np.any(edge_nodes >= points.shape[0])):
        raise PersistenceError("edge_nodes contains an invalid node index.")
    if len({tuple(edge) for edge in edge_nodes.tolist()}) != edge_nodes.shape[0]:
        raise PersistenceError("edge_nodes contains a duplicate canonical edge.")
    if cell_edges.size and (np.any(cell_edges < 0) or np.any(cell_edges >= edge_nodes.shape[0])):
        raise PersistenceError("cell_edges contains an invalid canonical edge index.")
    if not np.all(np.isin(cell_edge_signs, (-1, 1))):
        raise PersistenceError("cell_edge_signs must contain only -1 or +1.")

    local_pairs = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
    for row, cell in enumerate(cells):
        for column, ((first, second), edge_index) in enumerate(
            zip(local_pairs, cell_edges[row], strict=True)
        ):
            endpoints = tuple(int(value) for value in edge_nodes[int(edge_index)])
            expected_endpoints = tuple(
                sorted((int(cell[first]), int(cell[second])))
            )
            if endpoints != expected_endpoints:
                raise PersistenceError(
                    "cell_edges columns must follow the declared local Nedelec edge order."
                )
            expected_sign = 1 if int(cell[first]) < int(cell[second]) else -1
            if int(cell_edge_signs[row, column]) != expected_sign:
                raise PersistenceError(
                    "cell_edge_signs disagrees with canonical edge orientation."
                )

    node_pairs = optional.get("periodic_node_pairs")
    edge_pairs = optional.get("periodic_edge_pairs")
    if edge_pairs is not None:
        if node_pairs is None:
            raise PersistenceError("periodic_edge_pairs require periodic_node_pairs.")
        if edge_pairs.ndim != 2 or edge_pairs.shape[1] not in (2, 3):
            raise PersistenceError("periodic_edge_pairs must have two or three columns.")
        if edge_pairs.size and (
            np.any(edge_pairs[:, :2] < 0)
            or np.any(edge_pairs[:, :2] >= edge_nodes.shape[0])
        ):
            raise PersistenceError("A periodic edge pair is invalid.")
        if (
            np.unique(edge_pairs[:, 0]).size != edge_pairs.shape[0]
            or np.unique(edge_pairs[:, 1]).size != edge_pairs.shape[0]
        ):
            raise PersistenceError("Periodic edge_pairs must form a one-to-one map.")
        signs = (
            edge_pairs[:, 2]
            if edge_pairs.shape[1] == 3
            else np.ones(edge_pairs.shape[0], dtype=np.int64)
        )
        if not np.all(np.isin(signs, (-1, 1))):
            raise PersistenceError("Periodic edge signs must contain only -1 or +1.")
        node_map = {
            int(slave): int(master) for slave, master in node_pairs.tolist()
        }
        for (slave_edge, master_edge), sign in zip(
            edge_pairs[:, :2], signs, strict=True
        ):
            slave_endpoints = edge_nodes[int(slave_edge)]
            try:
                mapped = tuple(node_map[int(node)] for node in slave_endpoints)
            except KeyError as exc:
                raise PersistenceError(
                    "A periodic slave edge endpoint is absent from periodic_node_pairs."
                ) from exc
            expected_master = tuple(sorted(mapped))
            actual_master = tuple(int(value) for value in edge_nodes[int(master_edge)])
            expected_sign = 1 if mapped[0] < mapped[1] else -1
            if actual_master != expected_master or int(sign) != expected_sign:
                raise PersistenceError(
                    "Periodic edge pair/sign disagrees with the periodic node map."
                )


def _validate_periodic_node_pairs(
    points: NDArray[np.float64], node_pairs: NDArray[np.int64] | None
) -> None:
    if node_pairs is None:
        return
    if node_pairs.ndim != 2 or node_pairs.shape[1] != 2:
        raise PersistenceError("periodic_node_pairs must have shape (N, 2).")
    if node_pairs.size and (
        np.any(node_pairs < 0)
        or np.any(node_pairs >= points.shape[0])
        or np.any(node_pairs[:, 0] == node_pairs[:, 1])
    ):
        raise PersistenceError("A periodic node pair is invalid.")
    if (
        np.unique(node_pairs[:, 0]).size != node_pairs.shape[0]
        or np.unique(node_pairs[:, 1]).size != node_pairs.shape[0]
    ):
        raise PersistenceError("Periodic node_pairs must form a one-to-one map.")


def _mesh_record(mode_set: PeriodicModeSet) -> dict[str, Any]:
    fields = mode_set[0].fields
    for mode in mode_set[1:]:
        other = mode.fields
        if (
            other.dimension != fields.dimension
            or not np.array_equal(other.mesh_points, fields.mesh_points)
            or not np.array_equal(other.mesh_cells, fields.mesh_cells)
            or not np.array_equal(other.coordinates, fields.coordinates)
            or not np.array_equal(other.sample_element_indices, fields.sample_element_indices)
        ):
            raise PersistenceError("All modes in one case must share one mesh and sample grid.")
    points = _pad_points(fields.mesh_points, fields.dimension)
    samples = _pad_points(fields.coordinates, fields.dimension)
    cells = np.asarray(fields.mesh_cells, dtype=np.int64)
    owners = np.asarray(fields.sample_element_indices, dtype=np.int64)
    element_tags = _metadata_array(mode_set, fields, "mesh_element_tags", np.int64)
    if element_tags is None:
        element_tags = np.zeros(cells.shape[0], dtype=np.int64)
    if element_tags.shape != (cells.shape[0],):
        raise PersistenceError("mesh_element_tags must contain one value per cell.")
    optional: dict[str, NDArray[Any]] = {}
    for key, dtype in (
        ("periodic_node_pairs", np.int64),
        ("periodic_edge_pairs", np.int64),
        ("edge_nodes", np.int64),
        ("cell_edges", np.int64),
        ("cell_edge_signs", np.int8),
    ):
        value = _metadata_array(mode_set, fields, key, dtype)
        if value is not None:
            optional[key] = value
    _validate_periodic_node_pairs(points, optional.get("periodic_node_pairs"))
    if fields.dimension == 3:
        _validate_3d_edge_topology(points, cells, optional)
    raw_boundaries = mode_set.metadata.get("boundary_facets", {})
    boundaries: dict[str, NDArray[np.int64]] = {}
    if hasattr(raw_boundaries, "items"):
        for name, values in raw_boundaries.items():
            facets = np.asarray(values, dtype=np.int64)
            if facets.ndim != 2 or facets.shape[1] != fields.dimension:
                raise PersistenceError(
                    f"Boundary {name!r} must have shape (N, {fields.dimension})."
                )
            if np.any(facets < 0) or np.any(facets >= points.shape[0]):
                raise PersistenceError(f"Boundary {name!r} contains an invalid node index.")
            if facets.shape[0]:
                boundaries[str(name)] = facets
    reference_z = float(np.min(points[:, 2]))
    raw_physical_names = mode_set.metadata.get("physical_names", {})
    physical_names = (
        {int(key): str(value) for key, value in raw_physical_names.items()}
        if hasattr(raw_physical_names, "items")
        else {}
    )
    return {
        "dimension": fields.dimension,
        "topology": "triangle3" if fields.dimension == 2 else "tetra4",
        "points": points,
        "cells": cells,
        "samples": samples,
        "owners": owners,
        "element_tags": element_tags,
        "period": float(mode_set.period),
        "reference_z": reference_z,
        "optional": optional,
        "boundaries": boundaries,
        "physical_names": physical_names,
    }


def _mesh_key(record: dict[str, Any]) -> str:
    optional = record["optional"]
    values: list[Any] = [
        record["dimension"], record["topology"], record["period"], record["reference_z"],
        record["points"], record["cells"], record["samples"], record["owners"],
        record["element_tags"],
    ]
    for name in sorted(optional):
        values.extend((name, optional[name]))
    for name in sorted(record["boundaries"]):
        values.extend((name, record["boundaries"][name]))
    for identifier, name in sorted(record["physical_names"].items()):
        values.extend(("physical_name", int(identifier), str(name)))
    return _hash_record(*values)


def _validate_case(mode_set: PeriodicModeSet) -> None:
    if not isinstance(mode_set, PeriodicModeSet):
        raise TypeError("Every saved case must be a PeriodicModeSet.")
    if not np.isfinite(mode_set.frequency) or mode_set.frequency <= 0.0:
        raise PersistenceError("Case frequency must be finite and positive.")
    coefficient_size = mode_set[0].coefficients.size
    for mode in mode_set:
        if mode.residual is None:
            raise PersistenceError("Every persisted solver mode must have a residual.")
        if mode.coefficients.size != coefficient_size:
            raise PersistenceError("All coefficient vectors in a case must have equal length.")


def _prepare_cases(mode_sets: Sequence[PeriodicModeSet]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    meshes: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    mesh_indices: dict[str, int] = {}
    material_indices: dict[str, int] = {}
    for mode_set in mode_sets:
        _validate_case(mode_set)
        mesh = _mesh_record(mode_set)
        expected_dofs = (
            mesh["points"].shape[0]
            if mode_set.dimension == 2
            else mesh["optional"]["edge_nodes"].shape[0]
        )
        if any(mode.coefficients.size != expected_dofs for mode in mode_set):
            space = "mesh nodes" if mode_set.dimension == 2 else "canonical mesh edges"
            raise PersistenceError(
                f"full_expanded coefficients must contain one degree of freedom per {space}."
            )
        mesh_key = _mesh_key(mesh)
        mesh_index = mesh_indices.get(mesh_key)
        if mesh_index is None:
            mesh_index = len(meshes)
            mesh_indices[mesh_key] = mesh_index
            meshes.append(mesh)
        epsilon, mu, pml = _cell_materials(mode_set)
        material_key = _hash_record(mesh_index, epsilon, mu, pml)
        material_index = material_indices.get(material_key)
        if material_index is None:
            material_index = len(materials)
            material_indices[material_key] = material_index
            materials.append(
                {"mesh_index": mesh_index, "epsilon": epsilon, "mu": mu, "pml": pml}
            )
        cases.append(
            {
                "mode_set": mode_set,
                "mesh_index": mesh_index,
                "material_index": material_index,
            }
        )
    return meshes, materials, cases


def _write_archive(path: Path, mode_sets: Sequence[PeriodicModeSet]) -> None:
    meshes, materials, cases = _prepare_cases(mode_sets)
    mode_offsets = np.zeros(len(cases) + 1, dtype=np.int64)
    for index, case in enumerate(cases):
        mode_offsets[index + 1] = mode_offsets[index] + len(case["mode_set"])
    all_modes = [mode for case in cases for mode in case["mode_set"]]

    # Cap schema-v1 object formats at the viewer's documented HDF5 1.10 floor.
    with h5py.File(path, "x", libver=("v110", "v110"), track_order=True) as archive:
        write_envelope(archive, family="periodic_modes", kind="modes" if len(cases) == 1 else "sweep",
            dimension=mode_sets[0].dimension if len({case.dimension for case in mode_sets}) == 1 else 0, representation="periodic-envelope")
        archive.attrs["format"] = SCHEMA_FORMAT
        archive.attrs["schema_major"] = SCHEMA_MAJOR
        archive.attrs["schema_minor"] = SCHEMA_MINOR
        archive.attrs["kind"] = "single" if len(cases) == 1 else "sweep"
        archive.attrs["case_count"] = len(cases)
        archive.attrs["time_convention"] = "exp(+i*omega*t)"
        archive.attrs["field_representation"] = "periodic-envelope"
        archive.attrs["length_unit"] = "m"
        archive.attrs["frequency_unit"] = "Hz"
        archive.attrs["producer"] = "fem_periodic_modes"
        archive.attrs["producer_version"] = "1.0.0"
        archive.attrs["complex_storage"] = "compound-r-i"

        index = archive.create_group("index", track_order=True)
        index.create_dataset("frequency_hz", data=[case["mode_set"].frequency for case in cases])
        index.create_dataset("mode_offsets", data=mode_offsets)
        index.create_dataset("mesh_index", data=[case["mesh_index"] for case in cases])
        index.create_dataset(
            "material_state_index", data=[case["material_index"] for case in cases]
        )
        index.create_dataset("gamma_per_m", data=np.asarray([mode.gamma for mode in all_modes]))
        index.create_dataset("neff", data=np.asarray([mode.neff for mode in all_modes]))
        index.create_dataset(
            "neff_folded", data=np.asarray([mode.folded_neff for mode in all_modes])
        )
        index.create_dataset(
            "bloch_multiplier", data=np.asarray([mode.bloch_multiplier for mode in all_modes])
        )
        index.create_dataset(
            "alpha_per_m", data=np.asarray([mode.gamma.real for mode in all_modes], dtype=float)
        )
        index.create_dataset(
            "beta_per_m", data=np.asarray([mode.beta.real for mode in all_modes], dtype=float)
        )
        index.create_dataset(
            "beta_folded_per_m",
            data=np.asarray([mode.folded_beta.real for mode in all_modes], dtype=float),
        )
        index.create_dataset(
            "residual", data=np.asarray([mode.residual for mode in all_modes], dtype=float)
        )
        index.create_dataset(
            "pml_fraction", data=np.asarray([mode.pml_fraction for mode in all_modes], dtype=float)
        )
        _write_text(index, "polarization", [mode.polarization or "unknown" for mode in all_modes])
        _write_text(index, "direction", [mode.direction for mode in all_modes])
        _write_text(index, "normalization", [mode.normalization for mode in all_modes])
        if any(mode.gauss_residual is not None for mode in all_modes):
            index.create_dataset(
                "gauss_residual",
                data=np.asarray(
                    [0.0 if mode.gauss_residual is None else mode.gauss_residual for mode in all_modes],
                    dtype=float,
                ),
            )
            index.create_dataset(
                "gauss_available",
                data=np.asarray([mode.gauss_residual is not None for mode in all_modes], dtype=np.uint8),
            )

        mesh_group = archive.create_group("meshes", track_order=True)
        for mesh_index, mesh in enumerate(meshes):
            group = mesh_group.create_group(f"{mesh_index:06d}", track_order=True)
            group.attrs["dimension"] = mesh["dimension"]
            group.attrs["topology"] = mesh["topology"]
            group.attrs["periodic_axis"] = "z"
            group.attrs["period_m"] = mesh["period"]
            group.attrs["reference_z_m"] = mesh["reference_z"]
            group.attrs["edge_orientation"] = "ascending-node-index"
            _write_filtered(group, "points", mesh["points"])
            _write_filtered(group, "cells", mesh["cells"])
            _write_filtered(group, "cell_region_id", mesh["element_tags"])
            optional = mesh["optional"]
            for name in ("edge_nodes", "cell_edges"):
                if name in optional:
                    _write_filtered(group, name, optional[name])
            if "cell_edge_signs" in optional:
                _write_filtered(group, "cell_edge_sign", optional["cell_edge_signs"])
            if mesh["boundaries"]:
                boundary = group.create_group("boundary")
                boundary_names = sorted(mesh["boundaries"])
                facets = np.concatenate([mesh["boundaries"][name] for name in boundary_names])
                tags = np.concatenate(
                    [
                        np.full(mesh["boundaries"][name].shape[0], tag, dtype=np.int64)
                        for tag, name in enumerate(boundary_names, 1)
                    ]
                )
                _write_filtered(boundary, "facets", facets)
                _write_filtered(boundary, "tag", tags)
                names = boundary.create_group("names")
                names.create_dataset("tag", data=np.arange(1, len(boundary_names) + 1))
                _write_text(names, "name", boundary_names)
            if "periodic_node_pairs" in optional:
                periodic = group.create_group("periodic")
                _write_filtered(periodic, "node_pairs", optional["periodic_node_pairs"])
                affine = np.eye(4, dtype=np.float64)
                affine[2, 3] = mesh["period"]
                periodic.create_dataset("affine", data=affine)
                if "periodic_edge_pairs" in optional:
                    edge_pairs = np.asarray(optional["periodic_edge_pairs"], dtype=np.int64)
                    if edge_pairs.ndim != 2 or edge_pairs.shape[1] not in (2, 3):
                        raise PersistenceError("periodic_edge_pairs must have two or three columns.")
                    _write_filtered(periodic, "edge_pairs", edge_pairs[:, :2])
                    signs = edge_pairs[:, 2] if edge_pairs.shape[1] == 3 else np.ones(edge_pairs.shape[0])
                    _write_filtered(periodic, "edge_sign", np.asarray(signs, dtype=np.int8))
            if mesh["physical_names"]:
                names = group.create_group("physical_names")
                ids = sorted(int(value) for value in mesh["physical_names"])
                names.create_dataset("id", data=np.asarray(ids, dtype=np.int64))
                _write_text(names, "name", [str(mesh["physical_names"][value]) for value in ids])
            sample_group = group.create_group("samples")
            _write_filtered(sample_group, "points", mesh["samples"])
            _write_filtered(sample_group, "owner_cell", mesh["owners"])

        material_group = archive.create_group("material_states", track_order=True)
        for material_index, material in enumerate(materials):
            group = material_group.create_group(f"{material_index:06d}", track_order=True)
            group.attrs["mesh_index"] = material["mesh_index"]
            cells = material["epsilon"].shape[0]
            _write_filtered(
                group, "epsilon_r", material["epsilon"], chunks=(min(cells, 16_384), 3)
            )
            _write_filtered(group, "mu_r", material["mu"], chunks=(min(cells, 16_384), 3))
            _write_filtered(group, "pml_fraction", material["pml"], chunks=(min(cells, 16_384),))

        case_group = archive.create_group("cases", track_order=True)
        for case_index, case_record in enumerate(cases):
            mode_set: PeriodicModeSet = case_record["mode_set"]
            group = case_group.create_group(f"{case_index:06d}", track_order=True)
            write_value(group, "result_metadata", mode_set.metadata)
            write_value(group, "mesh_snapshot", mode_set.mesh_data)
            group.attrs["frequency_hz"] = mode_set.frequency
            group.attrs["omega"] = 2.0 * np.pi * mode_set.frequency
            group.attrs["k0"] = 2.0 * np.pi * mode_set.frequency / C_0
            group.attrs["mesh_index"] = case_record["mesh_index"]
            group.attrs["material_state_index"] = case_record["material_index"]
            group.attrs["mode_count"] = len(mode_set)
            group.attrs["backend"] = str(
                mode_set.metadata.get("backend", mode_set.metadata.get("eigensolver", "unknown"))
            )

            coefficients = group.create_group("coefficients")
            coefficients.attrs["space"] = (
                "P1-scalar-nodal" if mode_set.dimension == 2 else "Nedelec-N1-canonical-edges"
            )
            coefficients.attrs["full_expanded"] = 1
            primary = [
                "Ey" if mode.polarization == "TE" and mode_set.dimension == 2 else
                "Hy" if mode.polarization == "TM" and mode_set.dimension == 2 else "E"
                for mode in mode_set
            ]
            _write_text(coefficients, "primary_unknown", primary)
            coefficient_values = np.stack([mode.coefficients for mode in mode_set])
            _write_filtered(
                coefficients,
                "values",
                coefficient_values,
                chunks=_mode_chunks(coefficient_values.shape, coefficient_values.dtype.itemsize),
            )

            visualization = group.create_group("visualization")
            electric, magnetic = zip(*(_field_vectors(mode) for mode in mode_set), strict=True)
            electric_values = np.stack(electric)
            magnetic_values = np.stack(magnetic)
            _write_filtered(
                visualization,
                "E",
                electric_values,
                chunks=_mode_chunks(electric_values.shape, electric_values.dtype.itemsize),
            )
            _write_filtered(
                visualization,
                "H",
                magnetic_values,
                chunks=_mode_chunks(magnetic_values.shape, magnetic_values.dtype.itemsize),
            )
            metadata = group.create_group("mode_metadata")
            metadata.create_dataset(
                "has_power", data=np.asarray([mode.power is not None for mode in mode_set], dtype=np.uint8)
            )
            metadata.create_dataset(
                "power",
                data=np.asarray([0.0j if mode.power is None else mode.power for mode in mode_set]),
            )
            archive.flush()


def _atomic_save(mode_sets: Sequence[PeriodicModeSet], path: str | os.PathLike[str]) -> Path:
    destination = Path(path).expanduser().resolve()
    if not destination.parent.is_dir():
        raise PersistenceError(f"HDF5 destination directory does not exist: {destination.parent}")
    if destination.is_dir():
        raise PersistenceError(f"HDF5 destination is a directory: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        _write_archive(temporary, mode_sets)
        # Windows requires a writable CRT descriptor for ``_commit``/fsync.
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def save_periodic_h5(mode_set: PeriodicModeSet, path: str | os.PathLike[str]) -> Path:
    """Atomically save one periodic FEM result case."""

    return _atomic_save((mode_set,), path)


def save_periodic_sweep_h5(
    mode_sets: Iterable[PeriodicModeSet], path: str | os.PathLike[str]
) -> Path:
    """Atomically save a nonempty frequency/parameter sweep with deduplicated states."""

    cases = tuple(mode_sets)
    if not cases:
        raise PersistenceError("A periodic sweep must contain at least one mode set.")
    return _atomic_save(cases, path)


def _mode_selector(selector: int | slice | Iterable[int] | None, count: int) -> tuple[int, ...]:
    if selector is None:
        return tuple(range(count))
    if isinstance(selector, slice):
        return tuple(range(count))[selector]
    if isinstance(selector, (int, np.integer)):
        values = (int(selector),)
    else:
        values = tuple(int(value) for value in selector)
    normalized: list[int] = []
    for value in values:
        index = value + count if value < 0 else value
        if index < 0 or index >= count:
            raise IndexError(f"Mode index {value} is outside a case with {count} modes.")
        if index not in normalized:
            normalized.append(index)
    if not normalized:
        raise ValueError("modes cannot select an empty result.")
    return tuple(normalized)


class PeriodicH5Archive:
    """Lazy reader whose constructor touches only root metadata and ``/index``."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise PersistenceError(f"Periodic HDF5 file does not exist: {self.path}")
        self._mesh_cache: dict[int, dict[str, Any]] = {}
        self._material_cache: dict[int, dict[str, Any]] = {}
        with h5py.File(self.path, "r") as archive:
            validate_envelope(archive, family="periodic_modes")
            if _text(archive.attrs.get("format", "")) != SCHEMA_FORMAT:
                raise PersistenceError("The file is not a fem-periodic-modes archive.")
            self.schema_major = int(archive.attrs.get("schema_major", -1))
            self.schema_minor = int(archive.attrs.get("schema_minor", -1))
            if self.schema_major != SCHEMA_MAJOR or self.schema_minor < 0:
                raise PersistenceError(
                    f"Unsupported HDF5 schema {self.schema_major}.{self.schema_minor}."
                )
            self.kind = _text(archive.attrs.get("kind", ""))
            self.case_count = int(archive.attrs.get("case_count", -1))
            if self.kind not in ("single", "sweep") or self.case_count < 1:
                raise PersistenceError("Invalid archive kind or case_count.")
            if self.kind == "single" and self.case_count != 1:
                raise PersistenceError("A single archive must contain exactly one case.")
            if _text(archive.attrs.get("time_convention", "")) != "exp(+i*omega*t)":
                raise PersistenceError("Unsupported time convention.")
            if _text(archive.attrs.get("field_representation", "")) != "periodic-envelope":
                raise PersistenceError("Unsupported field representation.")
            index = _require_group(archive, "index")
            self.frequency_hz = _read_real(_require_dataset(index, "frequency_hz"))
            self.mode_offsets = _read_int(_require_dataset(index, "mode_offsets"))
            self.mesh_indices = _read_int(_require_dataset(index, "mesh_index"))
            self.material_indices = _read_int(_require_dataset(index, "material_state_index"))
            self.gamma = _read_complex(_require_dataset(index, "gamma_per_m"))
            self.neff = _read_complex(_require_dataset(index, "neff"))
            self.folded_neff = _read_complex(_require_dataset(index, "neff_folded"))
            self.bloch_multiplier = _read_complex(_require_dataset(index, "bloch_multiplier"))
            self.alpha = _read_real(_require_dataset(index, "alpha_per_m"))
            self.beta = _read_real(_require_dataset(index, "beta_per_m"))
            self.folded_beta = _read_real(_require_dataset(index, "beta_folded_per_m"))
            self.residual = _read_real(_require_dataset(index, "residual"))
            self.pml_fraction = _read_real(_require_dataset(index, "pml_fraction"))
            self.polarization = _read_strings(_require_dataset(index, "polarization"))
            self.direction = _read_strings(_require_dataset(index, "direction"))
            self.normalization = _read_strings(_require_dataset(index, "normalization"))
            self.gauss_residual = (
                _read_real(index["gauss_residual"]) if "gauss_residual" in index else None
            )
            if "gauss_available" in index:
                if self.gauss_residual is None:
                    raise PersistenceError(
                        "gauss_available requires a matching gauss_residual dataset."
                    )
                raw_gauss_available = _read_int(index["gauss_available"])
                if np.any((raw_gauss_available != 0) & (raw_gauss_available != 1)):
                    raise PersistenceError("gauss_available must contain only zero or one.")
                self.gauss_available = np.asarray(raw_gauss_available, dtype=bool)
            else:
                self.gauss_available = (
                    None
                    if self.gauss_residual is None
                    else np.ones(self.gauss_residual.shape, dtype=bool)
                )
        self._validate_index()
        for array in (
            self.frequency_hz, self.mode_offsets, self.mesh_indices, self.material_indices,
            self.gamma, self.neff, self.folded_neff, self.bloch_multiplier, self.alpha,
            self.beta, self.folded_beta, self.residual, self.pml_fraction,
        ):
            array.setflags(write=False)
        if self.gauss_residual is not None:
            self.gauss_residual.setflags(write=False)
        if self.gauss_available is not None:
            self.gauss_available.setflags(write=False)

    def _validate_index(self) -> None:
        if self.frequency_hz.shape != (self.case_count,) or np.any(self.frequency_hz <= 0.0):
            raise PersistenceError("frequency_hz is inconsistent with case_count.")
        if self.mode_offsets.shape != (self.case_count + 1,):
            raise PersistenceError("mode_offsets has the wrong length.")
        if self.mode_offsets[0] != 0 or np.any(np.diff(self.mode_offsets) < 0):
            raise PersistenceError("mode_offsets must start at zero and be nondecreasing.")
        if self.mesh_indices.shape != (self.case_count,) or self.material_indices.shape != (
            self.case_count,
        ):
            raise PersistenceError("Case mesh/material indexes have the wrong length.")
        if np.any(self.mesh_indices < 0) or np.any(self.material_indices < 0):
            raise PersistenceError("Mesh and material indexes cannot be negative.")
        total = int(self.mode_offsets[-1])
        arrays = (
            self.gamma, self.neff, self.folded_neff, self.bloch_multiplier, self.alpha,
            self.beta, self.folded_beta, self.residual, self.pml_fraction,
        )
        strings = (self.polarization, self.direction, self.normalization)
        if any(value.shape != (total,) for value in arrays) or any(
            len(value) != total for value in strings
        ):
            raise PersistenceError("One or more modal index datasets have the wrong length.")
        if self.gauss_residual is not None and self.gauss_residual.shape != (total,):
            raise PersistenceError("gauss_residual has the wrong length.")
        if self.gauss_available is not None and self.gauss_available.shape != (total,):
            raise PersistenceError("gauss_available has the wrong length.")
        if np.any(self.residual < 0.0) or np.any(
            (self.pml_fraction < 0.0) | (self.pml_fraction > 1.0)
        ):
            raise PersistenceError("Residual or PML fraction is outside its valid range.")

    @property
    def mode_count(self) -> int:
        return int(self.mode_offsets[-1])

    def __enter__(self) -> PeriodicH5Archive:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release cached arrays; no HDF5 handle is kept open between operations."""

        self._mesh_cache.clear()
        self._material_cache.clear()

    def _read_mesh(self, archive: h5py.File, index: int) -> dict[str, Any]:
        if index in self._mesh_cache:
            return self._mesh_cache[index]
        meshes = _require_group(archive, "meshes")
        group = _require_group(meshes, f"{index:06d}")
        dimension = int(group.attrs.get("dimension", -1))
        topology = _text(group.attrs.get("topology", ""))
        if (dimension, topology) not in ((2, "triangle3"), (3, "tetra4")):
            raise PersistenceError("Mesh dimension/topology is unsupported.")
        if _text(group.attrs.get("periodic_axis", "")) != "z":
            raise PersistenceError("Only z-periodic meshes are supported.")
        points = _read_real(_require_dataset(group, "points"))
        cells = _read_int(_require_dataset(group, "cells"))
        if points.ndim != 2 or points.shape[1] != 3 or not points.shape[0]:
            raise PersistenceError("Mesh points must have shape (N, 3).")
        if cells.ndim != 2 or cells.shape[1] != dimension + 1 or not cells.shape[0]:
            raise PersistenceError("Mesh cells have an invalid shape.")
        if np.any(cells < 0) or np.any(cells >= points.shape[0]):
            raise PersistenceError("Mesh cells contain an out-of-range node index.")
        samples_group = _require_group(group, "samples")
        samples = _read_real(_require_dataset(samples_group, "points"))
        owners = _read_int(_require_dataset(samples_group, "owner_cell"))
        if samples.ndim != 2 or samples.shape[1] != 3 or owners.shape != (samples.shape[0],):
            raise PersistenceError("Visualization samples have inconsistent shapes.")
        if np.any(owners < 0) or np.any(owners >= cells.shape[0]):
            raise PersistenceError("A visualization sample has an invalid owner cell.")
        optional = {
            name: _read_int(group[name])
            for name in ("cell_region_id", "edge_nodes", "cell_edges")
            if name in group
        }
        if "cell_region_id" in optional and optional["cell_region_id"].shape != (
            cells.shape[0],
        ):
            raise PersistenceError("cell_region_id must contain one value per mesh cell.")
        if "cell_edge_sign" in group:
            optional["cell_edge_signs"] = _read_int(group["cell_edge_sign"])
        boundaries: dict[str, NDArray[np.int64]] = {}
        if "boundary" in group:
            boundary = _require_group(group, "boundary")
            facets = _read_int(_require_dataset(boundary, "facets"))
            tags = _read_int(_require_dataset(boundary, "tag"))
            if facets.ndim != 2 or facets.shape[1] != dimension or tags.shape != (facets.shape[0],):
                raise PersistenceError("Boundary facets/tags have inconsistent shapes.")
            if facets.size and (np.any(facets < 0) or np.any(facets >= points.shape[0])):
                raise PersistenceError("A boundary facet contains an invalid node index.")
            tag_names: dict[int, str] = {}
            if "names" in boundary:
                names = _require_group(boundary, "names")
                name_tags = _read_int(_require_dataset(names, "tag"))
                name_values = _read_strings(_require_dataset(names, "name"))
                tag_names = dict(zip(name_tags.tolist(), name_values, strict=True))
            for tag in np.unique(tags):
                boundaries[tag_names.get(int(tag), f"boundary_{int(tag)}")] = facets[tags == tag]
        if "periodic" in group:
            periodic = _require_group(group, "periodic")
            node_pairs = _read_int(_require_dataset(periodic, "node_pairs"))
            if node_pairs.ndim != 2 or node_pairs.shape[1] != 2:
                raise PersistenceError("Periodic node_pairs must have shape (N, 2).")
            if node_pairs.size and (
                np.any(node_pairs < 0)
                or np.any(node_pairs >= points.shape[0])
                or np.any(node_pairs[:, 0] == node_pairs[:, 1])
            ):
                raise PersistenceError("A periodic node pair is invalid.")
            _validate_periodic_node_pairs(points, node_pairs)
            optional["periodic_node_pairs"] = node_pairs
            if "affine" in periodic:
                affine = _read_real(periodic["affine"])
                if affine.shape != (4, 4):
                    raise PersistenceError("Periodic affine must have shape (4, 4).")
            if "edge_pairs" in periodic:
                pairs = _read_int(periodic["edge_pairs"])
                signs = _read_int(_require_dataset(periodic, "edge_sign"))
                if pairs.ndim != 2 or pairs.shape[1] != 2 or signs.shape != (pairs.shape[0],):
                    raise PersistenceError("Periodic edge pairs/signs have inconsistent shapes.")
                edge_count = optional.get("edge_nodes", np.empty((0, 2))).shape[0]
                if pairs.size and (np.any(pairs < 0) or np.any(pairs >= edge_count)):
                    raise PersistenceError("A periodic edge pair is invalid.")
                if not np.all(np.isin(signs, (-1, 1))):
                    raise PersistenceError("Periodic edge signs must contain only -1 or +1.")
                optional["periodic_edge_pairs"] = np.column_stack((pairs, signs))
        physical_names: dict[int, str] = {}
        if "physical_names" in group:
            names_group = _require_group(group, "physical_names")
            identifiers = _read_int(_require_dataset(names_group, "id"))
            names = _read_strings(_require_dataset(names_group, "name"))
            if identifiers.ndim != 1 or identifiers.shape[0] != len(names):
                raise PersistenceError("physical_names ids/names have inconsistent shapes.")
            if np.unique(identifiers).size != identifiers.size:
                raise PersistenceError("physical_names contains a duplicate id.")
            physical_names = {
                int(identifier): name
                for identifier, name in zip(identifiers.tolist(), names, strict=True)
            }
        if dimension == 3:
            _validate_3d_edge_topology(points, cells, optional)
        record = {
            "dimension": dimension,
            "topology": topology,
            "period": float(group.attrs.get("period_m", np.nan)),
            "reference_z": float(group.attrs.get("reference_z_m", np.nan)),
            "points": points,
            "cells": cells,
            "samples": samples,
            "owners": owners,
            "optional": optional,
            "boundaries": boundaries,
            "physical_names": physical_names,
        }
        if (
            not np.isfinite(record["period"])
            or record["period"] <= 0.0
            or not np.isfinite(record["reference_z"])
        ):
            raise PersistenceError("Mesh period/reference_z metadata is invalid.")
        self._mesh_cache[index] = record
        return record

    def _read_material(self, archive: h5py.File, index: int) -> dict[str, Any]:
        if index in self._material_cache:
            return self._material_cache[index]
        states = _require_group(archive, "material_states")
        group = _require_group(states, f"{index:06d}")
        epsilon = _read_complex(_require_dataset(group, "epsilon_r"))
        mu = _read_complex(_require_dataset(group, "mu_r"))
        pml = _read_real(_require_dataset(group, "pml_fraction"))
        if epsilon.ndim != 2 or epsilon.shape[1] != 3 or mu.shape != epsilon.shape:
            raise PersistenceError("Material epsilon_r/mu_r must share shape (Ncells, 3).")
        if pml.shape != (epsilon.shape[0],) or np.any((pml < 0.0) | (pml > 1.0)):
            raise PersistenceError("Material PML fractions are invalid.")
        record = {
            "mesh_index": int(group.attrs.get("mesh_index", -1)),
            "epsilon": epsilon,
            "mu": mu,
            "pml": pml,
        }
        self._material_cache[index] = record
        return record

    def load_case(
        self,
        case: int = 0,
        *,
        modes: int | slice | Iterable[int] | None = None,
    ) -> PeriodicModeSet:
        """Load selected zero-based modes from one case using mode hyperslabs."""

        case = int(case)
        if case < 0:
            case += self.case_count
        if case < 0 or case >= self.case_count:
            raise IndexError(f"Case index {case} is outside {self.case_count} cases.")
        start = int(self.mode_offsets[case])
        count = int(self.mode_offsets[case + 1] - start)
        selected = _mode_selector(modes, count)
        with h5py.File(self.path, "r") as archive:
            mesh_index = int(self.mesh_indices[case])
            material_index = int(self.material_indices[case])
            mesh = self._read_mesh(archive, mesh_index)
            material = self._read_material(archive, material_index)
            if material["mesh_index"] != mesh_index or material["epsilon"].shape[0] != mesh["cells"].shape[0]:
                raise PersistenceError("The case mesh and material state are inconsistent.")
            cases = _require_group(archive, "cases")
            group = _require_group(cases, f"{case:06d}")
            coefficients = _require_group(group, "coefficients")
            visualization = _require_group(group, "visualization")
            coefficient_values = _require_dataset(coefficients, "values")
            electric_values = _require_dataset(visualization, "E")
            magnetic_values = _require_dataset(visualization, "H")
            if coefficient_values.ndim != 2 or coefficient_values.shape[0] != count:
                raise PersistenceError("Coefficient data has the wrong modal shape.")
            full_expanded = np.asarray(coefficients.attrs.get("full_expanded", 0))
            if (
                full_expanded.shape != ()
                or not np.issubdtype(full_expanded.dtype, np.integer)
                or int(full_expanded) != 1
            ):
                raise PersistenceError("Coefficients must declare full_expanded=1.")
            expected_space = (
                "P1-scalar-nodal"
                if mesh["dimension"] == 2
                else "Nedelec-N1-canonical-edges"
            )
            space = _scalar_text_attribute(coefficients, "space")
            if space != expected_space:
                raise PersistenceError(
                    f"Coefficient space {space!r} is incompatible with the case mesh."
                )
            expected_dofs = (
                mesh["points"].shape[0]
                if mesh["dimension"] == 2
                else mesh["optional"]["edge_nodes"].shape[0]
            )
            if coefficient_values.shape[1] != expected_dofs:
                raise PersistenceError(
                    "Expanded coefficient count does not match the mesh degree-of-freedom count."
                )
            expected_field_shape = (count, mesh["samples"].shape[0], 3)
            if electric_values.shape != expected_field_shape or magnetic_values.shape != expected_field_shape:
                raise PersistenceError("E/H visualization datasets have the wrong shape.")
            if "primary_unknown" in coefficients:
                primary = _read_strings(_require_dataset(coefficients, "primary_unknown"))
                if len(primary) != count:
                    raise PersistenceError("primary_unknown has the wrong length.")
            elif "primary_unknown" in coefficients.attrs:
                primary = (_scalar_text_attribute(coefficients, "primary_unknown"),) * count
            else:
                raise PersistenceError("Coefficients are missing primary_unknown metadata.")
            powers: NDArray[np.complex128] | None = None
            has_power: NDArray[np.bool_] | None = None
            if "mode_metadata" in group:
                metadata = _require_group(group, "mode_metadata")
                has_power_dataset = "has_power" in metadata
                power_dataset = "power" in metadata
                if has_power_dataset != power_dataset:
                    raise PersistenceError(
                        "mode_metadata must provide has_power and power together."
                    )
                if has_power_dataset:
                    raw_has_power = np.asarray(
                        _require_dataset(metadata, "has_power")[...]
                    )
                    if (
                        raw_has_power.shape != (count,)
                        or not np.issubdtype(raw_has_power.dtype, np.integer)
                        or not np.all(np.isin(raw_has_power, (0, 1)))
                    ):
                        raise PersistenceError(
                            "mode_metadata/has_power must be an integer 0/1 vector of length M."
                        )
                    powers = _read_complex(_require_dataset(metadata, "power"))
                    if powers.shape != (count,):
                        raise PersistenceError(
                            "mode_metadata/power must be a complex128 vector of length M."
                        )
                    has_power = np.asarray(raw_has_power, dtype=bool)
            points = mesh["points"][:, (0, 2)] if mesh["dimension"] == 2 else mesh["points"]
            samples = mesh["samples"][:, (0, 2)] if mesh["dimension"] == 2 else mesh["samples"]
            field_material = material["epsilon"][mesh["owners"]]
            entries: list[PeriodicMode] = []
            for local in selected:
                global_index = start + local
                electric = _read_complex(electric_values[local:local + 1])[0]
                magnetic = _read_complex(magnetic_values[local:local + 1])[0]
                coefficient = _read_complex(coefficient_values[local:local + 1])[0]
                field_metadata: dict[str, Any] = {
                    "sampling": "archive-visualization",
                    "time_convention": "exp(+1j*omega*t - 1j*k0*neff*z)",
                    "cell_epsilon_r": material["epsilon"],
                    "cell_mu_r": material["mu"],
                    "cell_pml_fraction": material["pml"],
                    "mesh_element_tags": mesh["optional"].get(
                        "cell_region_id", np.zeros(mesh["cells"].shape[0], dtype=np.int64)
                    ),
                    "physical_names": mesh["physical_names"],
                }
                field_metadata.update(
                    {name: value for name, value in mesh["optional"].items() if name != "cell_region_id"}
                )
                field_metadata["boundary_facets"] = mesh["boundaries"]
                fields = PeriodicSampledFields(
                    samples,
                    {
                        "Ex": electric[:, 0], "Ey": electric[:, 1], "Ez": electric[:, 2],
                        "Hx": magnetic[:, 0], "Hy": magnetic[:, 1], "Hz": magnetic[:, 2],
                    },
                    dimension=mesh["dimension"],
                    mesh_points=points,
                    mesh_cells=mesh["cells"],
                    sample_element_indices=mesh["owners"],
                    material=field_material,
                    metadata=field_metadata,
                )
                frequency = float(self.frequency_hz[case])
                k0 = float(group.attrs.get("k0", 2.0 * np.pi * frequency / C_0))
                power = None
                if powers is not None and has_power is not None and has_power[local]:
                    power = complex(powers[local])
                entries.append(
                    PeriodicMode(
                        neff=complex(self.neff[global_index]),
                        k0=k0,
                        period=mesh["period"],
                        fields=fields,
                        coefficients=coefficient,
                        index=local,
                        polarization=self.polarization[global_index],
                        power=power,
                        direction=self.direction[global_index],
                        normalization=self.normalization[global_index],
                        residual=float(self.residual[global_index]),
                        gauss_residual=(
                            None
                            if self.gauss_residual is None
                            or (self.gauss_available is not None and not self.gauss_available[global_index])
                            else float(self.gauss_residual[global_index])
                        ),
                        pml_fraction=float(self.pml_fraction[global_index]),
                        metadata={
                            "archive_case_index": case,
                            "archive_mode_index": local,
                            "primary_unknown": primary[local],
                            "coefficient_space": _text(coefficients.attrs.get("space", "unknown")),
                        },
                    )
                )
            result_metadata: dict[str, Any] = {
                "archive_path": str(self.path),
                "archive_case_index": case,
                "backend": _text(group.attrs.get("backend", "unknown")),
                "mesh_element_tags": field_metadata["mesh_element_tags"],
                "physical_names": mesh["physical_names"],
            }
            result_metadata.update(
                {name: value for name, value in mesh["optional"].items() if name != "cell_region_id"}
            )
            result_metadata["boundary_facets"] = mesh["boundaries"]
            result_metadata.update(read_value(group["result_metadata"], {}))
            result = PeriodicModeSet(
                entries,
                frequency=float(self.frequency_hz[case]),
                period=mesh["period"],
                dimension=mesh["dimension"],
                metadata=result_metadata,
            )
            object.__setattr__(result, "_mesh_snapshot", MeshSnapshot(points, mesh["cells"],
                ("x", "z") if mesh["dimension"] == 2 else ("x", "y", "z"), {},
                {"boundary_facets": mesh["boundaries"], "physical_names": mesh["physical_names"]}))
            snapshot = read_value(group["mesh_snapshot"], {"MeshSnapshot": MeshSnapshot}) if "mesh_snapshot" in group else None
            if snapshot is not None:
                object.__setattr__(result, "_mesh_snapshot", snapshot)
            return result


def open_periodic_h5(path: str | os.PathLike[str]) -> PeriodicH5Archive:
    """Open only the archive root and index, returning a lazy reader."""

    return PeriodicH5Archive(path)


def load_periodic_h5(
    path: str | os.PathLike[str],
    modes: int | slice | Iterable[int] | None = None,
) -> PeriodicModeSet | tuple[PeriodicModeSet, ...]:
    """Load one archive; sweeps return one immutable mode set per case."""

    with open_periodic_h5(path) as archive:
        loaded = tuple(archive.load_case(index, modes=modes) for index in range(archive.case_count))
    return loaded[0] if len(loaded) == 1 else loaded


def _validate_filter_contract(dataset: h5py.Dataset) -> None:
    if (
        dataset.compression != "gzip"
        or dataset.compression_opts != 4
        or not dataset.shuffle
        or not dataset.fletcher32
        or dataset.chunks is None
        or dataset.chunks[0] != 1
    ):
        raise PersistenceError(f"{dataset.name} violates the mode-first filter contract.")
    chunk_bytes = int(np.prod(dataset.chunks)) * dataset.dtype.itemsize
    if chunk_bytes > 1024 * 1024:
        raise PersistenceError(f"{dataset.name} uses a chunk larger than 1 MiB.")


def validate_periodic_h5(
    path: str | os.PathLike[str], *, deep: bool = False
) -> H5ValidationReport:
    """Validate schema/index data and optionally every referenced heavy dataset."""

    with open_periodic_h5(path) as archive:
        if deep:
            with h5py.File(archive.path, "r") as handle:
                for case in range(archive.case_count):
                    group = _require_group(_require_group(handle, "cases"), f"{case:06d}")
                    coefficients = _require_group(group, "coefficients")
                    visualization = _require_group(group, "visualization")
                    _validate_filter_contract(_require_dataset(coefficients, "values"))
                    _validate_filter_contract(_require_dataset(visualization, "E"))
                    _validate_filter_contract(_require_dataset(visualization, "H"))
                    archive.load_case(case)
        return H5ValidationReport(
            path=archive.path,
            schema_major=archive.schema_major,
            schema_minor=archive.schema_minor,
            case_count=archive.case_count,
            mode_count=archive.mode_count,
            deep=bool(deep),
        )


def _viewer_candidates(executable_name: str) -> tuple[Path, ...]:
    """Return override, checkout-build, and installed viewer candidates."""

    candidates: list[Path] = []
    configured = os.environ.get("FEM_PERIODIC_MODE_VIEWER_EXECUTABLE")
    if configured:
        candidates.append(Path(configured).expanduser())
    repository = next((parent for parent in Path(__file__).resolve().parents
                       if (parent / "apps" / "fem_periodic_mode_viewer").is_dir()), None)
    build_roots = []
    if repository is not None:
        source_root = repository / "apps" / "fem_periodic_mode_viewer"
        build_roots = [source_root / "build", *sorted(source_root.glob("build*")),
            *sorted(repository.glob("build*")), *sorted((repository / "outputs").glob("build*"))]
    configurations = ("Release", "RelWithDebInfo", "Debug", "MinSizeRel")
    for build_root in build_roots:
        for binary_root in (build_root, build_root / "apps" / "fem_periodic_mode_viewer"):
            candidates.append(binary_root / executable_name)
            candidates.append(
                binary_root
                / "fem-periodic-mode-viewer.app"
                / "Contents"
                / "MacOS"
                / "fem-periodic-mode-viewer"
            )
            for configuration in configurations:
                configured_root = binary_root / configuration
                candidates.append(configured_root / executable_name)
                candidates.append(
                    configured_root
                    / "fem-periodic-mode-viewer.app"
                    / "Contents"
                    / "MacOS"
                    / "fem-periodic-mode-viewer"
                )
    located = shutil.which(executable_name) or shutil.which("fem-periodic-mode-viewer")
    if located:
        candidates.append(Path(located))
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        candidates.append(
            Path(os.environ["LOCALAPPDATA"])
            / "FEMPeriodicModeViewer"
            / "bin"
            / executable_name
        )
    return tuple(candidates)


def _build_runtime_environment(executable: Path) -> dict[str, str] | None:
    """Return the MinGW DLL/plugin environment recorded by CMake."""

    if os.name != "nt":
        return None
    for directory in (executable.parent, *executable.parents):
        cache = directory / "CMakeCache.txt"
        if not cache.is_file():
            continue
        try:
            lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        compiler_value = next(
            (
                line.split("=", 1)[1]
                for line in lines
                if line.startswith("CMAKE_CXX_COMPILER:") and "=" in line
            ),
            None,
        )
        if compiler_value is None:
            return None
        runtime = Path(compiler_value).expanduser().resolve().parent
        if not any((runtime / name).is_file() for name in ("Qt6Core.dll", "libstdc++-6.dll")):
            return None
        environment = os.environ.copy()
        existing = environment.get("PATH", "")
        entries = [entry for entry in existing.split(os.pathsep) if entry]
        runtime_text = str(runtime)
        runtime_key = os.path.normcase(os.path.normpath(runtime_text))
        # A Conda IDE environment commonly includes MinGW late in PATH.  It
        # must still be moved to the front or Windows loads Conda's Qt DLLs
        # first and the MinGW viewer dies with STATUS_ENTRYPOINT_NOT_FOUND.
        entries = [
            entry
            for entry in entries
            if os.path.normcase(os.path.normpath(entry)) != runtime_key
        ]
        environment["PATH"] = os.pathsep.join((runtime_text, *entries))
        plugins = runtime.parent / "share" / "qt6" / "plugins"
        if plugins.is_dir():
            environment["QT_PLUGIN_PATH"] = str(plugins)
            platform_plugins = plugins / "platforms"
            if platform_plugins.is_dir():
                environment["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_plugins)
        return environment
    return None


def _confirm_viewer_started(process: object, executable: Path) -> None:
    """Raise instead of silently returning when the native process dies."""

    wait = getattr(process, "wait", None)
    if not callable(wait):
        return
    try:
        return_code = wait(timeout=0.35)
    except subprocess.TimeoutExpired:
        return
    if return_code is not None:
        raise PersistenceError(
            f"The native FEM periodic viewer exited before opening a window "
            f"(exit code {return_code}): {executable}. Rebuild or install its "
            "matching Qt/HDF5 runtime."
        )


def launch_viewer(
    path: str | os.PathLike[str] | None = None,
    *,
    _remove_on_exit: bool = False,
) -> subprocess.Popen[bytes]:
    """Open a result or result directory in the standalone native viewer.

    A file is schema-validated before launch.  Passing a directory lets the
    native GUI present its HDF5 selector; omitting ``path`` uses the current
    working directory.
    """

    archive_path = Path.cwd() if path is None else Path(path).expanduser()
    archive_path = archive_path.resolve()
    if archive_path.is_file():
        validate_periodic_h5(archive_path)
    elif not archive_path.is_dir():
        raise PersistenceError(f"Viewer path does not exist: {archive_path}")
    if _remove_on_exit and not archive_path.is_file():
        raise PersistenceError("Only a temporary HDF5 file can be removed on viewer exit.")
    executable_name = "fem-periodic-mode-viewer.exe" if os.name == "nt" else "fem-periodic-mode-viewer"
    candidates = _viewer_candidates(executable_name)
    executable = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if executable is None:
        raise PersistenceError(
            "fem-periodic-mode-viewer was not found. Install it on PATH or set "
            "FEM_PERIODIC_MODE_VIEWER_EXECUTABLE."
        )
    arguments = [str(executable)]
    if _remove_on_exit:
        arguments.append("--remove-source-on-exit")
    arguments.append(str(archive_path))
    environment = _build_runtime_environment(executable)
    if environment is None:
        process = subprocess.Popen(arguments)
    else:
        process = subprocess.Popen(arguments, env=environment)
    _confirm_viewer_started(process, executable)
    return process


__all__ = [
    "H5ValidationReport",
    "PeriodicH5Archive",
    "_viewer_candidates",
    "launch_viewer",
    "load_periodic_h5",
    "open_periodic_h5",
    "save_periodic_h5",
    "save_periodic_sweep_h5",
    "validate_periodic_h5",
]

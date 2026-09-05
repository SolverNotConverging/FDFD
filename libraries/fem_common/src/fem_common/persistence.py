"""Versioned envelopes and safe data-only HDF5 storage.

Readers supply an explicit type registry. Archives never select Python modules,
execute callbacks, or deserialize executable objects.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
import os
import tempfile

import h5py
import numpy as np

from .errors import PersistenceError

FORMAT = "cem-fem-results"
SCHEMA = "1.0"
EM_CONVENTION = "exp(+i*omega*t)"


def write_envelope(handle, *, family, kind, dimension, representation, static=False):
    handle.attrs.update(format=FORMAT, schema=SCHEMA, solver_family=family,
        result_kind=kind, units="SI", dimension=int(dimension),
        time_convention="static" if static else EM_CONVENTION,
        field_representation=representation)


def validate_envelope(handle, *, family, static=False):
    expected = {"format": FORMAT, "schema": SCHEMA, "solver_family": family,
                "units": "SI", "time_convention": "static" if static else EM_CONVENTION}
    for name, value in expected.items():
        actual = handle.attrs.get(name)
        if not isinstance(actual, str) or actual != value:
            raise PersistenceError(f"Incompatible FEM archive: {name} must be {value!r}; "
                                   f"received {handle.attrs.get(name)!r}. Legacy archives are unsupported.")
    if "field_representation" not in handle.attrs or "dimension" not in handle.attrs or "result_kind" not in handle.attrs:
        raise PersistenceError("Incomplete FEM archive envelope.")
    representations = {"waveguide_modes": "sampled-fields; exp(-i*beta*z)",
        "periodic_modes": "periodic-envelope", "waveguide_scattering": "sampled-fields; exp(-i*ky*y)",
        "electrostatics": "nodal-potential; nodal-and-cell-fields"}
    if handle.attrs["field_representation"] != representations[family]:
        raise PersistenceError("Incompatible spatial field representation.")
    dimension = handle.attrs["dimension"]
    dimensions = {"waveguide_modes": (1, 2), "periodic_modes": (0, 2, 3),
                  "waveguide_scattering": (2,), "electrostatics": (1, 2)}
    if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)) or dimension not in dimensions[family]:
        raise PersistenceError("Invalid archive mesh dimension.")


@contextmanager
def atomic_h5(path):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as handle:
            yield handle
            handle.flush()
        os.replace(temporary, target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_value(parent, name, value):
    """Store arrays, scalars, mappings, sequences, and data records only."""
    group = parent.create_group(name)
    if value is None:
        group.attrs["type"] = "none"
    elif isinstance(value, (str, bytes)):
        group.attrs.update(type="text", value=value.decode() if isinstance(value, bytes) else value)
    elif isinstance(value, (bool, int, float, complex, np.number, np.ndarray)):
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise PersistenceError(f"Unsupported array dtype {array.dtype} at {group.name}.")
        group.attrs["type"] = "array" if isinstance(value, np.ndarray) else "scalar"
        group.create_dataset("value", data=array, **({"compression": "gzip", "shuffle": True} if array.ndim and array.size > 256 else {}))
    elif isinstance(value, Mapping):
        group.attrs["type"] = "mapping"
        for index, (key, item) in enumerate(value.items()):
            entry = group.create_group(str(index))
            write_value(entry, "key", key)
            write_value(entry, "value", item)
    elif isinstance(value, (tuple, list, set, frozenset)):
        group.attrs["type"] = "sequence"
        for index, item in enumerate(value):
            write_value(group, str(index), item)
    elif is_dataclass(value):
        group.attrs.update(type="record", record=type(value).__name__)
        for descriptor in fields(value):
            write_value(group, descriptor.name, getattr(value, descriptor.name))
    else:
        raise PersistenceError(f"Unsupported non-data value {type(value).__name__} at {group.name}.")


def read_value(group, registry):
    tag = group.attrs.get("type")
    if tag == "none": return None
    if tag == "text": return str(group.attrs["value"])
    if tag == "array": return np.asarray(group["value"])
    if tag == "scalar": return group["value"][()].item()
    if tag == "sequence": return tuple(read_value(group[str(i)], registry) for i in range(len(group)))
    if tag == "mapping":
        return {read_value(entry["key"], registry): read_value(entry["value"], registry) for entry in group.values()}
    if tag == "record":
        name = group.attrs.get("record")
        if name not in registry:
            raise PersistenceError(f"Unsupported record type {name!r}.")
        return registry[name](**{key: read_value(value, registry) for key, value in group.items()})
    raise PersistenceError(f"Unsupported data tag {tag!r} at {group.name}.")

"""Data-only HDF5 storage for results.

Every result file carries the same header attributes, written and checked
here. Readers supply an explicit registry of record types; files never name
Python modules or contain executable objects.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path

import h5py
import numpy as np

from .constants import TIME_CONVENTION
from .errors import PersistenceError

FORMAT = "fdfd-result"
SCHEMA = "2.0"


def write_header(handle, *, family, kind):
    """Write the attributes shared by every fdfd result file."""
    from .. import __version__

    handle.attrs.update(
        format=FORMAT,
        schema=SCHEMA,
        solver_family=family,
        result_kind=kind,
        time_convention=TIME_CONVENTION,
        units="SI",
        fdfd_version=__version__,
    )


def read_header(handle):
    """Validate the shared header and return ``(solver_family, result_kind)``."""
    expected = {"format": FORMAT, "schema": SCHEMA, "time_convention": TIME_CONVENTION, "units": "SI"}
    for key, value in expected.items():
        actual = handle.attrs.get(key)
        if actual != value:
            raise PersistenceError(
                f"Not an fdfd {SCHEMA} result file: {key} is {actual!r}, expected {value!r}. "
                "Files written by fdfd 1.x are not supported."
            )
    try:
        return str(handle.attrs["solver_family"]), str(handle.attrs["result_kind"])
    except KeyError as exc:
        raise PersistenceError(f"Result file header is missing {exc}.") from exc


@contextmanager
def atomic_h5(path):
    """Write an HDF5 file through a temporary file that replaces ``path`` on success."""
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
    """Store arrays, scalars, mappings, sequences, and dataclass records only."""
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
        compression = {"compression": "gzip", "shuffle": True} if array.ndim and array.size > 256 else {}
        group.create_dataset("value", data=array, **compression)
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
    """Inverse of :func:`write_value`; records are rebuilt only from ``registry``."""
    tag = group.attrs.get("type")
    if tag == "none":
        return None
    if tag == "text":
        return str(group.attrs["value"])
    if tag == "array":
        return np.asarray(group["value"])
    if tag == "scalar":
        return group["value"][()].item()
    if tag == "sequence":
        return tuple(read_value(group[str(i)], registry) for i in range(len(group)))
    if tag == "mapping":
        entries = sorted(group.values(), key=lambda entry: int(entry.name.rsplit("/", 1)[-1]))
        return {read_value(entry["key"], registry): read_value(entry["value"], registry) for entry in entries}
    if tag == "record":
        name = group.attrs.get("record")
        if name not in registry:
            raise PersistenceError(f"Unsupported record type {name!r}.")
        return registry[name](**{key: read_value(value, registry) for key, value in group.items()})
    raise PersistenceError(f"Unsupported data tag {tag!r} at {group.name}.")

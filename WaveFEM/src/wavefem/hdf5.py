"""Versioned HDF5 persistence for WaveFEM scattering data.

Schema version 1 stores either one result or an ordered frequency sweep.  The
loader deliberately returns lightweight frozen records instead of recreating
live FEM objects, so files remain useful without a mesh, solver backend, or
the exact WaveFEM class version that produced them.  Each result may include
an additive scene subgroup containing a full-domain material mesh and line
overlays; older schema-v1 files without that subgroup remain valid.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from operator import index as integer_index
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import C0
from .exceptions import ConfigurationError
from .scene import Scene2D, SceneLine


SCHEMA_NAME = "wavefem"
SCHEMA_VERSION = 1
_COMPRESSION = "gzip"
_COMPRESSION_LEVEL = 4
_WINDOWS_REPLACE_RETRY_DELAYS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)
_WINDOWS_SHARING_VIOLATIONS = frozenset((5, 32, 33))
_POWER_NAMES = (
    "reflected_power",
    "transmitted_power",
    "radiated_power",
    "absorbed_power",
    "incident_power",
)
_MODE_COMPONENT_NAMES = (
    "x_nodes",
    "E_x",
    "E_y",
    "E_z",
    "H_x",
    "H_y",
    "H_z",
    "H_x_left",
    "H_x_right",
)
_MODE_METADATA_NAMES = (
    "beta",
    "neff",
    "power",
    "complex_power",
    "ky",
    "omega",
    "direction",
    "classification",
    "normalization",
    "residual",
    "divergence_residual",
)
_RESULT_METADATA_NAMES = (
    "ndofs",
    "solve_info",
    "mesh_info",
    "projection_condition_numbers",
    "reference_planes",
    "port_betas",
    "metadata",
)


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
PortKey = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class H5ModeData:
    """Portable sampled mode data loaded from a schema-v1 file.

    ``x`` is in metres.  ``E`` and ``H`` have shape ``(3, n)`` in Cartesian
    ``(x, y, z)`` order.  Scalar modal quantities such as ``beta``, ``neff``,
    ``omega``, direction, normalization, and residuals live in ``metadata``.
    Original mixed-space coefficient arrays live in ``raw_components``.
    """

    x: FloatArray
    E: ComplexArray
    H: ComplexArray
    metadata: Mapping[str, Any]
    raw_components: Mapping[str, NDArray[Any]]


@dataclass(frozen=True, slots=True)
class H5ResultData:
    """Portable fields, observables, modes, metadata, and optional scene.

    ``scene`` is a validated :class:`~wavefem.scene.Scene2D` when the file
    contains visualization data and ``None`` for legacy schema-v1 results.
    """

    frequency_hz: float | None
    ky: float | None
    coordinates: FloatArray
    E_incident: ComplexArray
    E_scattered: ComplexArray
    E_total: ComplexArray
    H_incident: ComplexArray
    H_scattered: ComplexArray
    H_total: ComplexArray
    s_parameters: Mapping[PortKey, complex]
    powers: Mapping[str, float]
    modes: tuple[H5ModeData, ...]
    metadata: Mapping[str, Any]
    scene: Scene2D | None = None


@dataclass(frozen=True, slots=True)
class H5FileData:
    """Top-level contents returned by :func:`load_h5`."""

    path: Path
    kind: Literal["single", "sweep"]
    frequencies_hz: FloatArray
    results: tuple[H5ResultData, ...]


@dataclass(frozen=True, slots=True)
class _PreparedMode:
    x: FloatArray
    E: ComplexArray
    H: ComplexArray
    metadata: Mapping[str, Any]
    raw_components: Mapping[str, NDArray[Any]]


@dataclass(frozen=True, slots=True)
class _PreparedResult:
    frequency_hz: float | None
    ky: float | None
    coordinates: FloatArray
    E_incident: ComplexArray
    E_scattered: ComplexArray
    E_total: ComplexArray
    H_incident: ComplexArray
    H_scattered: ComplexArray
    H_total: ComplexArray
    s_parameters: Mapping[PortKey, complex]
    powers: Mapping[str, float]
    modes: tuple[_PreparedMode, ...]
    metadata: Mapping[str, Any]
    scene: Scene2D | None


def _require_h5py() -> Any:
    try:
        import h5py
    except (ImportError, OSError) as exc:  # pragma: no cover - environment-specific
        raise ConfigurationError(
            "HDF5 persistence requires a working h5py installation. "
            "Install h5py in the active WaveFEM environment and ensure its "
            "native HDF5 libraries are loadable."
        ) from exc
    return h5py


def _path(value: os.PathLike[str] | str) -> Path:
    try:
        result = Path(value).expanduser().resolve()
    except (TypeError, ValueError, OSError) as exc:
        raise ConfigurationError("HDF5 path must be a valid filesystem path.") from exc
    if result.exists() and result.is_dir():
        raise ConfigurationError(f"HDF5 destination is a directory: {result}")
    if not result.parent.is_dir():
        raise ConfigurationError(
            f"HDF5 destination directory does not exist: {result.parent}"
        )
    return result


def _real_scalar(value: object, name: str, *, positive: bool = False) -> float:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be a finite real scalar.") from exc
    if array.shape != () or np.iscomplexobj(array) or isinstance(value, (bool, str, bytes)):
        raise ConfigurationError(f"{name} must be a finite real scalar.")
    try:
        result = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be a finite real scalar.") from exc
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise ConfigurationError(f"{name} must be a finite {qualifier}real scalar.")
    return result


def _complex_scalar(value: object, name: str) -> complex:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be a finite complex scalar.") from exc
    if array.shape != () or isinstance(value, (bool, str, bytes)):
        raise ConfigurationError(f"{name} must be a finite complex scalar.")
    try:
        result = complex(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be a finite complex scalar.") from exc
    if not np.isfinite((result.real, result.imag)).all():
        raise ConfigurationError(f"{name} must be a finite complex scalar.")
    return result


def _real_array(value: object, name: str) -> FloatArray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
        raise ConfigurationError(f"{name} must contain real values.")
    try:
        result = np.asarray(np.real(raw), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must contain real values.") from exc
    if not np.isfinite(result).all():
        raise ConfigurationError(f"{name} contains a non-finite value.")
    return result


def _complex_array(value: object, name: str) -> ComplexArray:
    try:
        result = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must contain complex-valued samples.") from exc
    if not np.isfinite(result).all():
        raise ConfigurationError(f"{name} contains a non-finite value.")
    return result


def _mapping(value: object, name: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping.")
    return value


def _prepare_s_parameters(value: object) -> Mapping[PortKey, complex]:
    source = _mapping(value, "result.s_parameters")
    prepared: dict[PortKey, complex] = {}
    for raw_key, raw_value in source.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 3:
            raise ConfigurationError(
                "Each S-parameter key must be a (side, out_mode, in_mode) tuple."
            )
        side = raw_key[0]
        if not isinstance(side, str) or side.lower() not in ("left", "right"):
            raise ConfigurationError("S-parameter side must be 'left' or 'right'.")
        indices: list[int] = []
        for item, label in zip(raw_key[1:], ("out_mode", "in_mode"), strict=True):
            if isinstance(item, bool):
                raise ConfigurationError(f"S-parameter {label} must be nonnegative.")
            try:
                converted = integer_index(item)
            except TypeError as exc:
                raise ConfigurationError(
                    f"S-parameter {label} must be a nonnegative integer."
                ) from exc
            if converted < 0:
                raise ConfigurationError(
                    f"S-parameter {label} must be a nonnegative integer."
                )
            indices.append(converted)
        key = (side.lower(), indices[0], indices[1])
        if key in prepared:
            raise ConfigurationError(f"Duplicate normalized S-parameter key {key!r}.")
        prepared[key] = _complex_scalar(raw_value, f"S-parameter {key!r}")
    return MappingProxyType(dict(sorted(prepared.items())))


def _prepare_powers(result: object) -> Mapping[str, float]:
    powers: dict[str, float] = {}
    for name in _POWER_NAMES:
        if not hasattr(result, name):
            raise ConfigurationError(f"Result is missing required attribute {name!r}.")
        value = _real_scalar(getattr(result, name), f"result.{name}")
        if value < 0.0:
            raise ConfigurationError(f"result.{name} must be nonnegative.")
        powers[name] = value
    if powers["incident_power"] <= 0.0:
        raise ConfigurationError("result.incident_power must be positive.")
    return MappingProxyType(powers)


def _materialize_modes(value: object, name: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)):
        raise ConfigurationError(f"{name} must be an iterable of mode objects.")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ConfigurationError(f"{name} must be an iterable of mode objects.") from exc


def _select_modes(
    result: object,
    modes: object,
    *,
    fallback_to_result: bool,
) -> tuple[object, ...]:
    selected = _materialize_modes(modes, "modes")
    if selected or not fallback_to_result or not hasattr(result, "modes"):
        return selected
    return _materialize_modes(getattr(result, "modes"), "result.modes")


def _prepare_mode(mode: object, index: int) -> _PreparedMode:
    for name in ("x", "E", "H"):
        if not hasattr(mode, name):
            raise ConfigurationError(f"Mode {index} is missing required attribute {name!r}.")
    x = _real_array(getattr(mode, "x"), f"mode {index}.x")
    if x.ndim != 1 or x.size == 0:
        raise ConfigurationError(f"mode {index}.x must be a nonempty 1D array.")
    electric = _complex_array(getattr(mode, "E"), f"mode {index}.E")
    magnetic = _complex_array(getattr(mode, "H"), f"mode {index}.H")
    expected = (3, x.size)
    if electric.shape != expected or magnetic.shape != expected:
        raise ConfigurationError(
            f"Mode {index} E and H must have shape {expected}; received "
            f"{electric.shape} and {magnetic.shape}."
        )

    metadata: dict[str, Any] = {
        "python_type": f"{type(mode).__module__}.{type(mode).__qualname__}"
    }
    for name in _MODE_METADATA_NAMES:
        if hasattr(mode, name):
            metadata[name] = getattr(mode, name)

    raw: dict[str, NDArray[Any]] = {}
    for name in _MODE_COMPONENT_NAMES:
        if not hasattr(mode, name):
            continue
        value = getattr(mode, name)
        if value is None:
            continue
        if name == "x_nodes":
            array: NDArray[Any] = _real_array(value, f"mode {index}.{name}")
        else:
            array = _complex_array(value, f"mode {index}.{name}")
        if array.ndim != 1:
            raise ConfigurationError(f"mode {index}.{name} must be a 1D array.")
        raw[name] = array
    return _PreparedMode(
        x=x,
        E=electric,
        H=magnetic,
        metadata=MappingProxyType(metadata),
        raw_components=MappingProxyType(raw),
    )


def _candidate_frequency_hz(result: object, modes: Sequence[object]) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []
    if hasattr(result, "frequency_hz") and getattr(result, "frequency_hz") is not None:
        candidates.append(
            (
                "result.frequency_hz",
                _real_scalar(getattr(result, "frequency_hz"), "result.frequency_hz", positive=True),
            )
        )
    if hasattr(result, "omega") and getattr(result, "omega") is not None:
        omega = _real_scalar(getattr(result, "omega"), "result.omega", positive=True)
        candidates.append(("result.omega", omega / (2.0 * np.pi)))
    solve_info = getattr(result, "solve_info", None)
    if isinstance(solve_info, Mapping):
        if solve_info.get("frequency_hz") is not None:
            candidates.append(
                (
                    "result.solve_info['frequency_hz']",
                    _real_scalar(
                        solve_info["frequency_hz"],
                        "result.solve_info['frequency_hz']",
                        positive=True,
                    ),
                )
            )
        elif solve_info.get("omega") is not None:
            omega = _real_scalar(
                solve_info["omega"], "result.solve_info['omega']", positive=True
            )
            candidates.append(("result.solve_info['omega']", omega / (2.0 * np.pi)))
        elif solve_info.get("length_scale") is not None:
            length_scale = _real_scalar(
                solve_info["length_scale"],
                "result.solve_info['length_scale']",
                positive=True,
            )
            candidates.append(
                ("result.solve_info['length_scale']", C0 / (2.0 * np.pi * length_scale))
            )
    for index, mode in enumerate(modes):
        if hasattr(mode, "omega"):
            omega = _real_scalar(
                getattr(mode, "omega"), f"mode {index}.omega", positive=True
            )
            candidates.append((f"mode {index}.omega", omega / (2.0 * np.pi)))
    return candidates


def _candidate_ky(result: object, modes: Sequence[object]) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []
    if hasattr(result, "ky") and getattr(result, "ky") is not None:
        candidates.append(
            ("result.ky", _real_scalar(getattr(result, "ky"), "result.ky"))
        )
    solve_info = getattr(result, "solve_info", None)
    if isinstance(solve_info, Mapping) and solve_info.get("ky") is not None:
        candidates.append(
            (
                "result.solve_info['ky']",
                _real_scalar(solve_info["ky"], "result.solve_info['ky']"),
            )
        )
    for index, mode in enumerate(modes):
        if hasattr(mode, "ky"):
            candidates.append(
                (f"mode {index}.ky", _real_scalar(getattr(mode, "ky"), f"mode {index}.ky"))
            )
    return candidates


def _consistent_optional(
    candidates: Sequence[tuple[str, float]],
    name: str,
) -> float | None:
    if not candidates:
        return None
    source, reference = candidates[0]
    for other_source, value in candidates[1:]:
        if not np.isclose(value, reference, rtol=1e-10, atol=1e-12):
            raise ConfigurationError(
                f"Inconsistent {name}: {source}={reference!r}, "
                f"but {other_source}={value!r}."
            )
    return float(reference)


def _result_metadata(result: object) -> Mapping[str, Any]:
    metadata: dict[str, Any] = {
        "python_type": f"{type(result).__module__}.{type(result).__qualname__}"
    }
    for name in _RESULT_METADATA_NAMES:
        if hasattr(result, name):
            metadata[name] = getattr(result, name)
    return MappingProxyType(metadata)


def _prepare_scene(result: object) -> Scene2D | None:
    """Normalize optional duck-typed scene data without coupling the writer."""

    if not hasattr(result, "scene") or getattr(result, "scene") is None:
        return None
    source = getattr(result, "scene")
    try:
        prepared_lines = tuple(
            SceneLine(
                kind=getattr(line, "kind"),
                endpoints=getattr(line, "endpoints"),
                label=getattr(line, "label", ""),
            )
            for line in getattr(source, "lines")
        )
        return Scene2D(
            points=getattr(source, "points"),
            triangles=getattr(source, "triangles"),
            eps_r=getattr(source, "eps_r"),
            x_span=getattr(source, "x_span"),
            z_span=getattr(source, "z_span"),
            lines=prepared_lines,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"result.scene is invalid: {exc}") from exc


def _prepare_result(
    result: object,
    modes: object,
    *,
    fallback_to_result_modes: bool,
    forced_frequency_hz: float | None = None,
) -> _PreparedResult:
    if result is None:
        raise ConfigurationError("result must be a scattering-result-like object.")
    selected_modes = _select_modes(
        result, modes, fallback_to_result=fallback_to_result_modes
    )
    mode_data = tuple(_prepare_mode(mode, index) for index, mode in enumerate(selected_modes))

    if not hasattr(result, "coordinates"):
        raise ConfigurationError("Result is missing required attribute 'coordinates'.")
    coordinates = _real_array(getattr(result, "coordinates"), "result.coordinates")
    if coordinates.ndim != 2 or coordinates.shape[0] != 2 or coordinates.shape[1] == 0:
        raise ConfigurationError("result.coordinates must have shape (2, npoints), npoints > 0.")
    npoints = coordinates.shape[1]

    fields: dict[str, ComplexArray] = {}
    for name in ("E_incident", "E_scattered", "H_incident", "H_scattered"):
        if not hasattr(result, name):
            raise ConfigurationError(f"Result is missing required attribute {name!r}.")
        field = _complex_array(getattr(result, name), f"result.{name}")
        if field.shape != (3, npoints):
            raise ConfigurationError(
                f"result.{name} must have shape (3, {npoints}); received {field.shape}."
            )
        fields[name] = field
    electric_total = fields["E_incident"] + fields["E_scattered"]
    magnetic_total = fields["H_incident"] + fields["H_scattered"]

    frequency_candidates = _candidate_frequency_hz(result, selected_modes)
    if forced_frequency_hz is not None:
        forced = _real_scalar(forced_frequency_hz, "frequency_hz", positive=True)
        frequency_candidates.insert(0, ("frequencies_hz", forced))
    frequency_hz = _consistent_optional(frequency_candidates, "frequency")
    ky = _consistent_optional(_candidate_ky(result, selected_modes), "ky")

    return _PreparedResult(
        frequency_hz=frequency_hz,
        ky=ky,
        coordinates=coordinates,
        E_incident=fields["E_incident"],
        E_scattered=fields["E_scattered"],
        E_total=np.asarray(electric_total, dtype=np.complex128),
        H_incident=fields["H_incident"],
        H_scattered=fields["H_scattered"],
        H_total=np.asarray(magnetic_total, dtype=np.complex128),
        s_parameters=_prepare_s_parameters(getattr(result, "s_parameters", None)),
        powers=_prepare_powers(result),
        modes=mode_data,
        metadata=_result_metadata(result),
        scene=_prepare_scene(result),
    )


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return {"__wavefem_type__": "float", "value": repr(value)}
    if isinstance(value, complex):
        return {
            "__wavefem_type__": "complex",
            "real": _json_ready(float(value.real)),
            "imag": _json_ready(float(value.imag)),
        }
    if isinstance(value, Path):
        return {"__wavefem_type__": "path", "value": str(value)}
    if isinstance(value, bytes):
        return {"__wavefem_type__": "bytes", "hex": value.hex()}
    if isinstance(value, np.ndarray):
        return {
            "__wavefem_type__": "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": _json_ready(value.tolist()),
        }
    if isinstance(value, Mapping):
        return {
            "__wavefem_type__": "mapping",
            "items": [
                [_json_ready(key), _json_ready(item)] for key, item in value.items()
            ],
        }
    if isinstance(value, tuple):
        return {"__wavefem_type__": "tuple", "items": [_json_ready(item) for item in value]}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return {
        "__wavefem_type__": "repr",
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": repr(value),
    }


def _json_restore(value: object) -> object:
    if isinstance(value, list):
        return [_json_restore(item) for item in value]
    if not isinstance(value, dict) or "__wavefem_type__" not in value:
        return value
    kind = value["__wavefem_type__"]
    if kind == "complex":
        return complex(
            float(_json_restore(value["real"])),
            float(_json_restore(value["imag"])),
        )
    if kind == "float":
        return float(str(value["value"]))
    if kind == "path":
        return Path(str(value["value"]))
    if kind == "bytes":
        return bytes.fromhex(str(value["hex"]))
    if kind == "tuple":
        return tuple(_json_restore(item) for item in value["items"])
    if kind == "mapping":
        restored: dict[Any, Any] = {}
        for raw_key, raw_item in value["items"]:
            key = _json_restore(raw_key)
            try:
                restored[key] = _json_restore(raw_item)
            except TypeError as exc:
                raise ValueError("HDF5 metadata contains an unhashable mapping key.") from exc
        return restored
    if kind == "ndarray":
        data = _json_restore(value["data"])
        try:
            result = np.asarray(data, dtype=np.dtype(str(value["dtype"])))
            return result.reshape(tuple(int(item) for item in value["shape"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("HDF5 metadata contains an invalid encoded array.") from exc
    if kind == "repr":
        return {
            "python_type": str(value.get("python_type", "unknown")),
            "repr": str(value.get("value", "")),
        }
    raise ValueError(f"HDF5 metadata uses unknown tagged type {kind!r}.")


def _json_dump(value: object) -> str:
    return json.dumps(
        _json_ready(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_load(value: object, name: str) -> object:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a UTF-8 JSON string.")
    try:
        return _json_restore(json.loads(value))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains invalid WaveFEM metadata JSON.") from exc


def _dataset_options(array: NDArray[Any]) -> dict[str, Any]:
    if array.ndim == 0 or array.size == 0:
        return {}
    return {
        "compression": _COMPRESSION,
        "compression_opts": _COMPRESSION_LEVEL,
        "shuffle": True,
    }


def _write_array(group: Any, name: str, value: ArrayLike) -> None:
    array = np.asarray(value)
    group.create_dataset(name, data=array, **_dataset_options(array))


def _write_mode(group: Any, mode: _PreparedMode) -> None:
    _write_array(group, "x", np.asarray(mode.x, dtype=np.float64))
    _write_array(group, "E", np.asarray(mode.E, dtype=np.complex128))
    _write_array(group, "H", np.asarray(mode.H, dtype=np.complex128))
    group.attrs["metadata_json"] = _json_dump(mode.metadata)
    raw_group = group.create_group("raw_components")
    for name, array in mode.raw_components.items():
        _write_array(raw_group, name, array)


def _write_scene(group: Any, scene: Scene2D) -> None:
    group.attrs["format"] = "wavefem-scene"
    group.attrs["version"] = 1
    group.attrs["coordinate_order"] = "x,z"
    _write_array(group, "points", np.asarray(scene.points, dtype=np.float64))
    _write_array(group, "triangles", np.asarray(scene.triangles, dtype=np.int64))
    _write_array(group, "eps_r", np.asarray(scene.eps_r, dtype=np.complex128))
    _write_array(group, "x_span", np.asarray(scene.x_span, dtype=np.float64))
    _write_array(group, "z_span", np.asarray(scene.z_span, dtype=np.float64))

    line_group = group.create_group("lines")
    _write_array(
        line_group,
        "kind",
        np.asarray([line.kind.encode("ascii") for line in scene.lines], dtype="S9"),
    )
    endpoints = (
        np.stack([line.endpoints for line in scene.lines])
        if scene.lines
        else np.empty((0, 2, 2), dtype=np.float64)
    )
    _write_array(line_group, "endpoints", np.asarray(endpoints, dtype=np.float64))
    encoded_labels = [line.label.encode("utf-8") for line in scene.lines]
    label_width = max((len(label) for label in encoded_labels), default=1)
    _write_array(
        line_group,
        "label",
        np.asarray(encoded_labels, dtype=f"S{label_width}"),
    )


def _write_result(group: Any, result: _PreparedResult, index: int) -> None:
    group.attrs["index"] = index
    if result.frequency_hz is not None:
        group.attrs["frequency_hz"] = result.frequency_hz
    if result.ky is not None:
        group.attrs["ky"] = result.ky
    group.attrs["metadata_json"] = _json_dump(result.metadata)

    _write_array(group, "coordinates", np.asarray(result.coordinates, dtype=np.float64))
    fields = group.create_group("fields")
    for name in (
        "E_incident",
        "E_scattered",
        "E_total",
        "H_incident",
        "H_scattered",
        "H_total",
    ):
        _write_array(fields, name, np.asarray(getattr(result, name), dtype=np.complex128))

    s_group = group.create_group("s_parameters")
    records = list(result.s_parameters.items())
    _write_array(
        s_group,
        "side",
        np.asarray([key[0].encode("ascii") for key, _ in records], dtype="S5"),
    )
    _write_array(
        s_group,
        "out_mode",
        np.asarray([key[1] for key, _ in records], dtype=np.int64),
    )
    _write_array(
        s_group,
        "in_mode",
        np.asarray([key[2] for key, _ in records], dtype=np.int64),
    )
    _write_array(
        s_group,
        "value",
        np.asarray([value for _, value in records], dtype=np.complex128),
    )

    power_group = group.create_group("powers")
    for name, value in result.powers.items():
        power_group.attrs[name] = value

    mode_group = group.create_group("modes")
    mode_group.attrs["count"] = len(result.modes)
    for mode_index, mode in enumerate(result.modes):
        _write_mode(mode_group.create_group(f"{mode_index:06d}"), mode)
    if result.scene is not None:
        _write_scene(group.create_group("scene"), result.scene)


def _write_file(
    handle: Any,
    *,
    kind: Literal["single", "sweep"],
    frequencies_hz: FloatArray,
    results: Sequence[_PreparedResult],
) -> None:
    handle.attrs["format"] = SCHEMA_NAME
    handle.attrs["schema_version"] = SCHEMA_VERSION
    handle.attrs["kind"] = kind
    handle.attrs["result_count"] = len(results)
    handle.attrs["complex_storage"] = "native"
    _write_array(handle, "frequencies_hz", np.asarray(frequencies_hz, dtype=np.float64))
    result_group = handle.create_group("results")
    for index, result in enumerate(results):
        _write_result(result_group.create_group(f"{index:06d}"), result, index)


def _replace_file(source: Path, destination: Path) -> None:
    """Atomically replace a file, tolerating transient Windows share locks.

    ``h5py.File`` is fully closed before this helper is called.  Windows virus
    scanners and indexers can nevertheless open the just-created temporary
    file during that handoff and briefly cause ``os.replace`` to report
    ``ERROR_ACCESS_DENIED``, ``ERROR_SHARING_VIOLATION``, or
    ``ERROR_LOCK_VIOLATION``.  Retry only those specific errors; every other
    failure remains immediate.
    """

    for delay in (*_WINDOWS_REPLACE_RETRY_DELAYS, None):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if (
                delay is None
                or getattr(exc, "winerror", None)
                not in _WINDOWS_SHARING_VIOLATIONS
            ):
                raise
            time.sleep(delay)


def _atomic_write(
    destination: Path,
    *,
    kind: Literal["single", "sweep"],
    frequencies_hz: FloatArray,
    results: Sequence[_PreparedResult],
) -> None:
    h5py = _require_h5py()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            _write_file(
                handle,
                kind=kind,
                frequencies_hz=frequencies_hz,
                results=results,
            )
            handle.flush()
        _replace_file(temporary, destination)
    except OSError as exc:
        raise ConfigurationError(
            f"Could not write HDF5 file {destination}: {exc}"
        ) from exc
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def save_result_h5(
    result: object,
    path: os.PathLike[str] | str,
    *,
    modes: Iterable[object] = (),
) -> Path:
    """Save one scattering result using WaveFEM HDF5 schema version 1.

    Duck-typed ``frequency_hz``, ``ky``, and ``modes`` attributes are used
    when present.  Frequency can also be recovered from legacy
    ``solve_info['length_scale']`` metadata or supplied modes.  Unknown values
    remain explicitly absent instead of being guessed.
    """

    destination = _path(path)
    prepared = _prepare_result(
        result,
        modes,
        fallback_to_result_modes=True,
    )
    frequency = np.asarray(
        [np.nan if prepared.frequency_hz is None else prepared.frequency_hz],
        dtype=np.float64,
    )
    _atomic_write(
        destination,
        kind="single",
        frequencies_hz=frequency,
        results=(prepared,),
    )
    return destination


def save_sweep_h5(
    frequencies_hz: ArrayLike,
    results: Sequence[object],
    path: os.PathLike[str] | str,
    *,
    modes_per_result: Sequence[Iterable[object]] | None = None,
) -> Path:
    """Save an ordered nonempty frequency sweep using schema version 1."""

    destination = _path(path)
    frequency_array = _real_array(frequencies_hz, "frequencies_hz")
    if frequency_array.ndim != 1 or frequency_array.size == 0:
        raise ConfigurationError("frequencies_hz must be a nonempty 1D array.")
    if np.any(frequency_array <= 0.0):
        raise ConfigurationError("frequencies_hz values must be strictly positive.")
    if np.any(np.diff(frequency_array) <= 0.0):
        raise ConfigurationError("frequencies_hz must be strictly increasing.")
    try:
        result_items = tuple(results)
    except TypeError as exc:
        raise ConfigurationError("results must be a sequence of result objects.") from exc
    if len(result_items) != frequency_array.size:
        raise ConfigurationError(
            "frequencies_hz and results must contain the same number of entries."
        )

    if modes_per_result is None:
        mode_items: tuple[object, ...] = tuple(() for _ in result_items)
        fallback = True
    else:
        try:
            mode_items = tuple(modes_per_result)
        except TypeError as exc:
            raise ConfigurationError(
                "modes_per_result must contain one mode iterable per result."
            ) from exc
        if len(mode_items) != len(result_items):
            raise ConfigurationError(
                "modes_per_result must contain one entry per result."
            )
        fallback = False

    prepared = tuple(
        _prepare_result(
            result,
            mode_group,
            fallback_to_result_modes=fallback,
            forced_frequency_hz=float(frequency),
        )
        for frequency, result, mode_group in zip(
            frequency_array, result_items, mode_items, strict=True
        )
    )
    _atomic_write(
        destination,
        kind="sweep",
        frequencies_hz=frequency_array,
        results=prepared,
    )
    return destination


def _attribute_text(value: object, name: str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"HDF5 attribute {name!r} is not valid UTF-8.") from exc
    if isinstance(value, str):
        return value
    raise ValueError(f"HDF5 attribute {name!r} must be text.")


def _require_member(group: Any, name: str) -> Any:
    if name not in group:
        raise ValueError(f"HDF5 schema-v1 object {group.name!r} is missing {name!r}.")
    return group[name]


def _loaded_array(dataset: Any, name: str, *, complex_values: bool) -> NDArray[Any]:
    try:
        array = np.asarray(dataset[...])
    except Exception as exc:
        raise ValueError(f"Could not read HDF5 dataset {name!r}.") from exc
    if complex_values:
        if not np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(f"HDF5 dataset {name!r} must use native complex storage.")
        result: NDArray[Any] = np.asarray(array, dtype=np.complex128)
    else:
        if np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(f"HDF5 dataset {name!r} must be real-valued.")
        try:
            result = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"HDF5 dataset {name!r} must be numeric.") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"HDF5 dataset {name!r} contains a non-finite value.")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


def _metadata_attribute(group: Any) -> Mapping[str, Any]:
    if "metadata_json" not in group.attrs:
        raise ValueError(f"HDF5 group {group.name!r} is missing metadata_json.")
    restored = _json_load(group.attrs["metadata_json"], f"{group.name}.metadata_json")
    if not isinstance(restored, Mapping):
        raise ValueError(f"HDF5 metadata for {group.name!r} must decode to a mapping.")
    return MappingProxyType(dict(restored))


def _load_mode(group: Any) -> H5ModeData:
    x = _loaded_array(_require_member(group, "x"), f"{group.name}/x", complex_values=False)
    electric = _loaded_array(
        _require_member(group, "E"), f"{group.name}/E", complex_values=True
    )
    magnetic = _loaded_array(
        _require_member(group, "H"), f"{group.name}/H", complex_values=True
    )
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"HDF5 mode {group.name!r} x must be a nonempty 1D array.")
    if electric.shape != (3, x.size) or magnetic.shape != (3, x.size):
        raise ValueError(
            f"HDF5 mode {group.name!r} E/H arrays must have shape (3, {x.size})."
        )
    raw_group = _require_member(group, "raw_components")
    raw: dict[str, NDArray[Any]] = {}
    for name in raw_group:
        raw[name] = _loaded_array(
            raw_group[name],
            f"{raw_group.name}/{name}",
            complex_values=name != "x_nodes",
        )
        if raw[name].ndim != 1:
            raise ValueError(
                f"HDF5 raw mode component {raw_group.name}/{name} must be 1D."
            )
    return H5ModeData(
        x=np.asarray(x, dtype=np.float64),
        E=np.asarray(electric, dtype=np.complex128),
        H=np.asarray(magnetic, dtype=np.complex128),
        metadata=_metadata_attribute(group),
        raw_components=MappingProxyType(raw),
    )


def _load_s_parameters(group: Any) -> Mapping[PortKey, complex]:
    sides_raw = np.asarray(_require_member(group, "side")[...])
    out_modes = np.asarray(_require_member(group, "out_mode")[...])
    in_modes = np.asarray(_require_member(group, "in_mode")[...])
    values = _loaded_array(
        _require_member(group, "value"), f"{group.name}/value", complex_values=True
    )
    lengths = {sides_raw.size, out_modes.size, in_modes.size, values.size}
    if len(lengths) != 1 or any(array.ndim != 1 for array in (sides_raw, out_modes, in_modes, values)):
        raise ValueError(f"HDF5 S-parameter records in {group.name!r} are inconsistent.")
    result: dict[PortKey, complex] = {}
    for raw_side, raw_out, raw_in, raw_value in zip(
        sides_raw, out_modes, in_modes, values, strict=True
    ):
        try:
            side = bytes(raw_side).decode("ascii").lower()
        except (TypeError, UnicodeDecodeError) as exc:
            raise ValueError("HDF5 S-parameter side is not ASCII text.") from exc
        if side not in ("left", "right"):
            raise ValueError(f"HDF5 S-parameter side {side!r} is invalid.")
        try:
            out_mode, in_mode = integer_index(raw_out), integer_index(raw_in)
        except TypeError as exc:
            raise ValueError("HDF5 S-parameter mode indices must be integers.") from exc
        if out_mode < 0 or in_mode < 0:
            raise ValueError("HDF5 S-parameter mode indices must be nonnegative.")
        key = (side, out_mode, in_mode)
        if key in result:
            raise ValueError(f"HDF5 file contains duplicate S-parameter key {key!r}.")
        result[key] = complex(raw_value)
    return MappingProxyType(result)


def _loaded_integer_array(dataset: Any, name: str) -> NDArray[np.int64]:
    try:
        raw = np.asarray(dataset[...])
    except Exception as exc:
        raise ValueError(f"Could not read HDF5 dataset {name!r}.") from exc
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"HDF5 dataset {name!r} must use integer storage.")
    result = np.array(raw, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _loaded_text_array(dataset: Any, name: str) -> tuple[str, ...]:
    try:
        raw = np.asarray(dataset[...])
    except Exception as exc:
        raise ValueError(f"Could not read HDF5 dataset {name!r}.") from exc
    if raw.ndim != 1:
        raise ValueError(f"HDF5 dataset {name!r} must be one-dimensional text.")
    result: list[str] = []
    for item in raw:
        try:
            if isinstance(item, (bytes, np.bytes_)):
                result.append(bytes(item).decode("utf-8"))
            elif isinstance(item, str):
                result.append(item)
            else:
                raise TypeError
        except (TypeError, UnicodeDecodeError) as exc:
            raise ValueError(f"HDF5 dataset {name!r} must contain UTF-8 text.") from exc
    return tuple(result)


def _load_scene(group: Any) -> Scene2D:
    if _attribute_text(group.attrs.get("format"), f"{group.name}.format") != "wavefem-scene":
        raise ValueError(f"HDF5 scene {group.name!r} has an invalid format attribute.")
    try:
        version = integer_index(group.attrs["version"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"HDF5 scene {group.name!r} has no valid version.") from exc
    if version != 1:
        raise ValueError(f"Unsupported HDF5 scene version {version}; expected version 1.")
    if _attribute_text(
        group.attrs.get("coordinate_order"), f"{group.name}.coordinate_order"
    ) != "x,z":
        raise ValueError("HDF5 scene coordinate_order must be 'x,z'.")

    points = _loaded_array(
        _require_member(group, "points"), f"{group.name}/points", complex_values=False
    )
    triangles = _loaded_integer_array(
        _require_member(group, "triangles"), f"{group.name}/triangles"
    )
    eps_r = _loaded_array(
        _require_member(group, "eps_r"), f"{group.name}/eps_r", complex_values=True
    )
    x_span = _loaded_array(
        _require_member(group, "x_span"), f"{group.name}/x_span", complex_values=False
    )
    z_span = _loaded_array(
        _require_member(group, "z_span"), f"{group.name}/z_span", complex_values=False
    )
    line_group = _require_member(group, "lines")
    kinds = _loaded_text_array(
        _require_member(line_group, "kind"), f"{line_group.name}/kind"
    )
    endpoints = _loaded_array(
        _require_member(line_group, "endpoints"),
        f"{line_group.name}/endpoints",
        complex_values=False,
    )
    labels = _loaded_text_array(
        _require_member(line_group, "label"), f"{line_group.name}/label"
    )
    if endpoints.shape != (len(kinds), 2, 2) or len(labels) != len(kinds):
        raise ValueError(f"HDF5 scene lines in {line_group.name!r} are inconsistent.")
    try:
        lines = tuple(
            SceneLine(kind, endpoints[index], labels[index])
            for index, kind in enumerate(kinds)
        )
        return Scene2D(
            points=points,
            triangles=triangles,
            eps_r=eps_r,
            x_span=x_span,
            z_span=z_span,
            lines=lines,
        )
    except ValueError as exc:
        raise ValueError(f"HDF5 scene {group.name!r} is invalid: {exc}") from exc


def _load_result(group: Any, root_frequency: float) -> H5ResultData:
    coordinates = _loaded_array(
        _require_member(group, "coordinates"),
        f"{group.name}/coordinates",
        complex_values=False,
    )
    if coordinates.ndim != 2 or coordinates.shape[0] != 2 or coordinates.shape[1] == 0:
        raise ValueError(
            f"HDF5 result {group.name!r} coordinates must have shape (2, npoints)."
        )
    npoints = coordinates.shape[1]
    field_group = _require_member(group, "fields")
    fields: dict[str, ComplexArray] = {}
    for name in (
        "E_incident",
        "E_scattered",
        "E_total",
        "H_incident",
        "H_scattered",
        "H_total",
    ):
        array = _loaded_array(
            _require_member(field_group, name),
            f"{field_group.name}/{name}",
            complex_values=True,
        )
        if array.shape != (3, npoints):
            raise ValueError(
                f"HDF5 field {field_group.name}/{name} must have shape (3, {npoints})."
            )
        fields[name] = np.asarray(array, dtype=np.complex128)
    if not np.allclose(
        fields["E_total"], fields["E_incident"] + fields["E_scattered"], rtol=1e-13, atol=1e-15
    ):
        raise ValueError("HDF5 E_total is inconsistent with incident + scattered fields.")
    if not np.allclose(
        fields["H_total"], fields["H_incident"] + fields["H_scattered"], rtol=1e-13, atol=1e-15
    ):
        raise ValueError("HDF5 H_total is inconsistent with incident + scattered fields.")

    powers_group = _require_member(group, "powers")
    powers: dict[str, float] = {}
    for name in _POWER_NAMES:
        if name not in powers_group.attrs:
            raise ValueError(f"HDF5 powers group is missing {name!r}.")
        value = powers_group.attrs[name]
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"HDF5 power {name!r} must be real.") from exc
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"HDF5 power {name!r} must be finite and nonnegative.")
        powers[name] = numeric
    if powers["incident_power"] <= 0.0:
        raise ValueError("HDF5 incident_power must be positive.")

    mode_group = _require_member(group, "modes")
    try:
        mode_count = integer_index(mode_group.attrs["count"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"HDF5 mode group {mode_group.name!r} has no valid count.") from exc
    modes: list[H5ModeData] = []
    for index in range(mode_count):
        modes.append(_load_mode(_require_member(mode_group, f"{index:06d}")))
    if len(mode_group) != mode_count:
        raise ValueError(f"HDF5 mode count for {mode_group.name!r} is inconsistent.")

    if "frequency_hz" in group.attrs:
        frequency_hz: float | None = float(group.attrs["frequency_hz"])
    elif np.isfinite(root_frequency):
        frequency_hz = float(root_frequency)
    else:
        frequency_hz = None
    if frequency_hz is not None and (not np.isfinite(frequency_hz) or frequency_hz <= 0.0):
        raise ValueError("HDF5 result frequency_hz must be finite and positive.")
    if (
        frequency_hz is not None
        and np.isfinite(root_frequency)
        and not np.isclose(frequency_hz, root_frequency, rtol=1e-12, atol=0.0)
    ):
        raise ValueError(
            "HDF5 result frequency_hz is inconsistent with frequencies_hz."
        )
    ky = float(group.attrs["ky"]) if "ky" in group.attrs else None
    if ky is not None and not np.isfinite(ky):
        raise ValueError("HDF5 result ky must be finite.")

    return H5ResultData(
        frequency_hz=frequency_hz,
        ky=ky,
        coordinates=np.asarray(coordinates, dtype=np.float64),
        E_incident=fields["E_incident"],
        E_scattered=fields["E_scattered"],
        E_total=fields["E_total"],
        H_incident=fields["H_incident"],
        H_scattered=fields["H_scattered"],
        H_total=fields["H_total"],
        s_parameters=_load_s_parameters(_require_member(group, "s_parameters")),
        powers=MappingProxyType(powers),
        modes=tuple(modes),
        metadata=_metadata_attribute(group),
        scene=_load_scene(group["scene"]) if "scene" in group else None,
    )


def load_h5(path: os.PathLike[str] | str) -> H5FileData:
    """Load and validate one WaveFEM schema-v1 HDF5 file."""

    source = _path(path)
    if not source.is_file():
        raise ValueError(f"HDF5 file does not exist: {source}")
    h5py = _require_h5py()
    try:
        with h5py.File(source, "r") as handle:
            format_name = _attribute_text(handle.attrs.get("format"), "format")
            if format_name != SCHEMA_NAME:
                raise ValueError(
                    f"Not a WaveFEM HDF5 file: format={format_name!r}."
                )
            try:
                version = integer_index(handle.attrs["schema_version"])
            except (KeyError, TypeError) as exc:
                raise ValueError("WaveFEM HDF5 schema_version is missing or invalid.") from exc
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported WaveFEM HDF5 schema version {version}; "
                    f"this reader supports version {SCHEMA_VERSION}."
                )
            kind_text = _attribute_text(handle.attrs.get("kind"), "kind")
            if kind_text not in ("single", "sweep"):
                raise ValueError(f"WaveFEM HDF5 kind {kind_text!r} is invalid.")
            kind: Literal["single", "sweep"] = kind_text
            try:
                result_count = integer_index(handle.attrs["result_count"])
            except (KeyError, TypeError) as exc:
                raise ValueError("WaveFEM HDF5 result_count is missing or invalid.") from exc
            if result_count <= 0 or (kind == "single" and result_count != 1):
                raise ValueError("WaveFEM HDF5 result_count is inconsistent with kind.")

            frequencies_dataset = _require_member(handle, "frequencies_hz")
            frequencies_raw = np.asarray(frequencies_dataset[...])
            if np.issubdtype(frequencies_raw.dtype, np.complexfloating):
                raise ValueError("HDF5 frequencies_hz must be real-valued.")
            try:
                frequencies = np.asarray(frequencies_raw, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("HDF5 frequencies_hz must be numeric.") from exc
            if frequencies.shape != (result_count,):
                raise ValueError(
                    "HDF5 frequencies_hz length does not match result_count."
                )
            if kind == "sweep":
                if not np.isfinite(frequencies).all() or np.any(frequencies <= 0.0):
                    raise ValueError("Sweep frequencies_hz must be finite and positive.")
                if np.any(np.diff(frequencies) <= 0.0):
                    raise ValueError("Sweep frequencies_hz must be strictly increasing.")
            elif not (np.isfinite(frequencies[0]) or np.isnan(frequencies[0])):
                raise ValueError("Single-result frequency must be positive or the NaN unknown sentinel.")
            elif np.isfinite(frequencies[0]) and frequencies[0] <= 0.0:
                raise ValueError("Single-result frequency must be positive.")

            result_group = _require_member(handle, "results")
            results = tuple(
                _load_result(
                    _require_member(result_group, f"{index:06d}"),
                    float(frequencies[index]),
                )
                for index in range(result_count)
            )
            if len(result_group) != result_count:
                raise ValueError("HDF5 results group count does not match result_count.")
    except OSError as exc:
        raise ValueError(f"Could not open HDF5 file {source}: {exc}") from exc

    loaded_frequencies = np.array(frequencies, copy=True)
    loaded_frequencies.setflags(write=False)
    return H5FileData(
        path=source,
        kind=kind,
        frequencies_hz=loaded_frequencies,
        results=results,
    )


__all__ = [
    "H5FileData",
    "H5ModeData",
    "H5ResultData",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "load_h5",
    "save_result_h5",
    "save_sweep_h5",
]

"""Independent reader for WaveFEM schema-v1 HDF5 result files.

This module intentionally does not import the WaveFEM solver package.  It
validates the on-disk interchange format and returns immutable lightweight
records from :mod:`wavefem_viewer.model`.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from operator import index as integer_index
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .model import FileData, ModeData, ResultData, SceneData, SceneLine


SCHEMA_NAME = "wavefem"
SCHEMA_VERSION = 1
_POWER_NAMES = (
    "reflected_power",
    "transmitted_power",
    "radiated_power",
    "absorbed_power",
    "incident_power",
)
_SCENE_LINE_KINDS = frozenset(("pec", "pmc", "wave_port", "pml"))


def _require_h5py() -> Any:
    try:
        import h5py
    except (ImportError, OSError) as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Reading WaveFEM results requires a working h5py installation."
        ) from exc
    return h5py


def _source_path(value: os.PathLike[str] | str) -> Path:
    try:
        source = Path(value).expanduser().resolve()
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError("HDF5 path must be a valid filesystem path.") from exc
    if not source.is_file():
        raise ValueError(f"HDF5 file does not exist: {source}")
    return source


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


def _read_dataset(dataset: Any, name: str) -> NDArray[Any]:
    try:
        return np.asarray(dataset[...])
    except Exception as exc:
        raise ValueError(f"Could not read HDF5 dataset {name!r}.") from exc


def _readonly(array: NDArray[Any], dtype: Any) -> NDArray[Any]:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _real_array(
    dataset: Any,
    name: str,
    *,
    allow_nan: bool = False,
) -> NDArray[np.float64]:
    raw = _read_dataset(dataset, name)
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError(f"HDF5 dataset {name!r} must be real-valued.")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"HDF5 dataset {name!r} must be numeric.") from exc
    invalid = np.isinf(result) if allow_nan else ~np.isfinite(result)
    if np.any(invalid):
        raise ValueError(f"HDF5 dataset {name!r} contains a non-finite value.")
    return _readonly(result, np.float64)


def _complex_array(
    dataset: Any,
    name: str,
    *,
    require_native_complex: bool = True,
) -> NDArray[np.complex128]:
    raw = _read_dataset(dataset, name)
    if require_native_complex and not np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError(f"HDF5 dataset {name!r} must use native complex storage.")
    if not (
        np.issubdtype(raw.dtype, np.complexfloating)
        or np.issubdtype(raw.dtype, np.number)
    ):
        raise ValueError(f"HDF5 dataset {name!r} must be numeric.")
    result = np.asarray(raw, dtype=np.complex128)
    if not np.isfinite(result).all():
        raise ValueError(f"HDF5 dataset {name!r} contains a non-finite value.")
    return _readonly(result, np.complex128)


def _index_array(dataset: Any, name: str) -> NDArray[np.int64]:
    raw = _read_dataset(dataset, name)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"HDF5 dataset {name!r} must contain integer indices.")
    return _readonly(raw, np.int64)


def _decode_text(value: object, name: str) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"HDF5 text value in {name!r} is not valid UTF-8.") from exc
    if isinstance(value, (str, np.str_)):
        return str(value)
    raise ValueError(f"HDF5 dataset {name!r} must contain text.")


def _text_array(dataset: Any, name: str) -> tuple[str, ...]:
    raw = _read_dataset(dataset, name)
    if raw.ndim != 1:
        raise ValueError(f"HDF5 dataset {name!r} must be one-dimensional.")
    return tuple(_decode_text(value, name) for value in raw)


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
                raise ValueError("HDF5 metadata has an unhashable mapping key.") from exc
        return restored
    if kind == "ndarray":
        try:
            data = _json_restore(value["data"])
            result = np.asarray(data, dtype=np.dtype(str(value["dtype"])))
            return result.reshape(tuple(int(item) for item in value["shape"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("HDF5 metadata has an invalid encoded array.") from exc
    if kind == "repr":
        return {
            "python_type": str(value.get("python_type", "unknown")),
            "repr": str(value.get("value", "")),
        }
    raise ValueError(f"HDF5 metadata uses unknown tagged type {kind!r}.")


def _metadata(group: Any) -> Mapping[str, Any]:
    if "metadata_json" not in group.attrs:
        raise ValueError(f"HDF5 group {group.name!r} is missing metadata_json.")
    encoded = group.attrs["metadata_json"]
    if isinstance(encoded, bytes):
        try:
            encoded = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Metadata for {group.name!r} is not valid UTF-8.") from exc
    if not isinstance(encoded, str):
        raise ValueError(f"Metadata for {group.name!r} must be a JSON string.")
    try:
        restored = _json_restore(json.loads(encoded))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Metadata for {group.name!r} is invalid WaveFEM JSON.") from exc
    if not isinstance(restored, Mapping):
        raise ValueError(f"Metadata for {group.name!r} must decode to a mapping.")
    return MappingProxyType(dict(restored))


def _load_mode(group: Any) -> ModeData:
    x = _real_array(_require_member(group, "x"), f"{group.name}/x")
    electric = _complex_array(_require_member(group, "E"), f"{group.name}/E")
    magnetic = _complex_array(_require_member(group, "H"), f"{group.name}/H")
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"HDF5 mode {group.name!r} x must be a nonempty 1D array.")
    if electric.shape != (3, x.size) or magnetic.shape != (3, x.size):
        raise ValueError(
            f"HDF5 mode {group.name!r} E/H arrays must have shape (3, {x.size})."
        )
    raw_group = _require_member(group, "raw_components")
    raw: dict[str, NDArray[Any]] = {}
    for name in raw_group:
        raw[name] = (
            _real_array(raw_group[name], f"{raw_group.name}/{name}")
            if name == "x_nodes"
            else _complex_array(raw_group[name], f"{raw_group.name}/{name}")
        )
        if raw[name].ndim != 1:
            raise ValueError(f"HDF5 raw mode component {name!r} must be 1D.")
    return ModeData(
        x=x,
        E=electric,
        H=magnetic,
        metadata=_metadata(group),
        raw_components=MappingProxyType(raw),
    )


def _load_s_parameters(group: Any) -> Mapping[tuple[str, int, int], complex]:
    sides = _text_array(_require_member(group, "side"), f"{group.name}/side")
    out_modes = _read_dataset(_require_member(group, "out_mode"), f"{group.name}/out_mode")
    in_modes = _read_dataset(_require_member(group, "in_mode"), f"{group.name}/in_mode")
    values = _complex_array(_require_member(group, "value"), f"{group.name}/value")
    lengths = {len(sides), out_modes.size, in_modes.size, values.size}
    if len(lengths) != 1 or any(
        array.ndim != 1 for array in (out_modes, in_modes, values)
    ):
        raise ValueError(f"HDF5 S-parameter records in {group.name!r} are inconsistent.")
    if not np.issubdtype(out_modes.dtype, np.integer) or not np.issubdtype(
        in_modes.dtype, np.integer
    ):
        raise ValueError("HDF5 S-parameter mode indices must be integers.")
    result: dict[tuple[str, int, int], complex] = {}
    for side, raw_out, raw_in, raw_value in zip(
        sides, out_modes, in_modes, values, strict=True
    ):
        normalized_side = side.lower()
        if normalized_side not in ("left", "right"):
            raise ValueError(f"HDF5 S-parameter side {normalized_side!r} is invalid.")
        out_mode = integer_index(raw_out)
        in_mode = integer_index(raw_in)
        if out_mode < 0 or in_mode < 0:
            raise ValueError("HDF5 S-parameter mode indices must be nonnegative.")
        key = (normalized_side, out_mode, in_mode)
        if key in result:
            raise ValueError(f"HDF5 file contains duplicate S-parameter key {key!r}.")
        result[key] = complex(raw_value)
    return MappingProxyType(result)


def _load_scene(group: Any) -> SceneData:
    if _attribute_text(group.attrs.get("format"), f"{group.name}.format") != "wavefem-scene":
        raise ValueError(f"HDF5 scene group {group.name!r} has an invalid format marker.")
    try:
        scene_version = integer_index(group.attrs.get("version"))
    except TypeError as exc:
        raise ValueError(f"HDF5 scene group {group.name!r} has no valid version.") from exc
    if scene_version != 1:
        raise ValueError(
            f"Unsupported HDF5 scene version {scene_version}; this viewer supports version 1."
        )
    if _attribute_text(
        group.attrs.get("coordinate_order"), f"{group.name}.coordinate_order"
    ) != "x,z":
        raise ValueError(
            f"HDF5 scene group {group.name!r} must use coordinate_order 'x,z'."
        )

    points = _real_array(_require_member(group, "points"), f"{group.name}/points")
    triangles = _index_array(
        _require_member(group, "triangles"), f"{group.name}/triangles"
    )
    eps_r = _complex_array(
        _require_member(group, "eps_r"),
        f"{group.name}/eps_r",
        require_native_complex=False,
    )
    x_span = _real_array(_require_member(group, "x_span"), f"{group.name}/x_span")
    z_span = _real_array(_require_member(group, "z_span"), f"{group.name}/z_span")
    if points.ndim != 2 or points.shape[0] != 2 or points.shape[1] < 3:
        raise ValueError(f"HDF5 scene {group.name!r} points must have shape (2, N), N >= 3.")
    if triangles.ndim != 2 or triangles.shape[0] != 3 or triangles.shape[1] == 0:
        raise ValueError(
            f"HDF5 scene {group.name!r} triangles must have shape (3, M), M > 0."
        )
    if np.any(triangles < 0) or np.any(triangles >= points.shape[1]):
        raise ValueError(f"HDF5 scene {group.name!r} has an out-of-range triangle index.")
    if eps_r.shape != (triangles.shape[1],):
        raise ValueError(
            f"HDF5 scene {group.name!r} eps_r must have one value per triangle."
        )
    for span, name in ((x_span, "x_span"), (z_span, "z_span")):
        if span.shape != (2,) or span[0] >= span[1]:
            raise ValueError(
                f"HDF5 scene {group.name!r} {name} must be an increasing length-2 array."
            )
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, *(abs(float(value)) for value in (*x_span, *z_span))
    )
    if (
        np.any(points[0] < x_span[0] - tolerance)
        or np.any(points[0] > x_span[1] + tolerance)
        or np.any(points[1] < z_span[0] - tolerance)
        or np.any(points[1] > z_span[1] + tolerance)
    ):
        raise ValueError(f"HDF5 scene {group.name!r} has points outside its spans.")
    if np.any(
        (triangles[0] == triangles[1])
        | (triangles[1] == triangles[2])
        | (triangles[2] == triangles[0])
    ):
        raise ValueError(f"HDF5 scene {group.name!r} has a repeated triangle vertex.")
    p0, p1, p2 = (points[:, triangles[row]] for row in range(3))
    twice_area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (
        p1[1] - p0[1]
    ) * (p2[0] - p0[0])
    area_scale = max(
        float((x_span[1] - x_span[0]) * (z_span[1] - z_span[0])),
        np.finfo(float).tiny,
    )
    area_tolerance = 64.0 * np.finfo(float).eps * area_scale
    if np.any(np.abs(twice_area) <= area_tolerance):
        raise ValueError(f"HDF5 scene {group.name!r} has a degenerate triangle.")

    lines_group = _require_member(group, "lines")
    kinds = _text_array(_require_member(lines_group, "kind"), f"{lines_group.name}/kind")
    labels = _text_array(
        _require_member(lines_group, "label"), f"{lines_group.name}/label"
    )
    if "endpoints" in lines_group:
        endpoints = _real_array(
            lines_group["endpoints"], f"{lines_group.name}/endpoints"
        )
    else:
        components = tuple(
            _real_array(_require_member(lines_group, name), f"{lines_group.name}/{name}")
            for name in ("x0", "z0", "x1", "z1")
        )
        if any(component.ndim != 1 for component in components):
            raise ValueError("HDF5 scene split line endpoints must be one-dimensional.")
        endpoints = _readonly(
            np.stack(
                (
                    np.stack((components[0], components[1]), axis=1),
                    np.stack((components[2], components[3]), axis=1),
                ),
                axis=1,
            ),
            np.float64,
        )
    if endpoints.shape != (len(kinds), 2, 2) or len(labels) != len(kinds):
        raise ValueError(f"HDF5 scene line records in {lines_group.name!r} are inconsistent.")
    if "count" in lines_group.attrs:
        try:
            count = integer_index(lines_group.attrs["count"])
        except TypeError as exc:
            raise ValueError("HDF5 scene line count must be an integer.") from exc
        if count != len(kinds):
            raise ValueError("HDF5 scene line count does not match its datasets.")
    lines: list[SceneLine] = []
    for index, (kind, label) in enumerate(zip(kinds, labels, strict=True)):
        normalized = kind.strip().lower()
        if normalized not in _SCENE_LINE_KINDS:
            raise ValueError(f"HDF5 scene line kind {kind!r} is invalid.")
        if np.allclose(endpoints[index, 0], endpoints[index, 1], rtol=0.0, atol=0.0):
            raise ValueError(f"HDF5 scene line {index} has identical endpoints.")
        if (
            np.any(endpoints[index, :, 0] < x_span[0] - tolerance)
            or np.any(endpoints[index, :, 0] > x_span[1] + tolerance)
            or np.any(endpoints[index, :, 1] < z_span[0] - tolerance)
            or np.any(endpoints[index, :, 1] > z_span[1] + tolerance)
        ):
            raise ValueError(f"HDF5 scene line {index} lies outside the scene spans.")
        endpoint_copy = _readonly(endpoints[index], np.float64)
        lines.append(SceneLine(normalized, endpoint_copy, label))  # type: ignore[arg-type]

    return SceneData(
        points=points,
        triangles=triangles,
        eps_r=eps_r,
        x_span=x_span,
        z_span=z_span,
        lines=tuple(lines),
    )


def _load_result(group: Any, root_frequency: float) -> ResultData:
    coordinates = _real_array(
        _require_member(group, "coordinates"), f"{group.name}/coordinates"
    )
    if coordinates.ndim != 2 or coordinates.shape[0] != 2 or coordinates.shape[1] == 0:
        raise ValueError(
            f"HDF5 result {group.name!r} coordinates must have shape (2, npoints)."
        )
    npoints = coordinates.shape[1]
    fields_group = _require_member(group, "fields")
    fields: dict[str, NDArray[np.complex128]] = {}
    for name in (
        "E_incident",
        "E_scattered",
        "E_total",
        "H_incident",
        "H_scattered",
        "H_total",
    ):
        field = _complex_array(
            _require_member(fields_group, name), f"{fields_group.name}/{name}"
        )
        if field.shape != (3, npoints):
            raise ValueError(
                f"HDF5 field {fields_group.name}/{name} must have shape (3, {npoints})."
            )
        fields[name] = field
    if not np.allclose(
        fields["E_total"], fields["E_incident"] + fields["E_scattered"],
        rtol=1e-13, atol=1e-15,
    ):
        raise ValueError("HDF5 E_total is inconsistent with incident + scattered fields.")
    if not np.allclose(
        fields["H_total"], fields["H_incident"] + fields["H_scattered"],
        rtol=1e-13, atol=1e-15,
    ):
        raise ValueError("HDF5 H_total is inconsistent with incident + scattered fields.")

    powers_group = _require_member(group, "powers")
    powers: dict[str, float] = {}
    for name in _POWER_NAMES:
        if name not in powers_group.attrs:
            raise ValueError(f"HDF5 powers group is missing {name!r}.")
        try:
            value = float(powers_group.attrs[name])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"HDF5 power {name!r} must be real.") from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"HDF5 power {name!r} must be finite and nonnegative.")
        powers[name] = value
    if powers["incident_power"] <= 0.0:
        raise ValueError("HDF5 incident_power must be positive.")

    modes_group = _require_member(group, "modes")
    try:
        mode_count = integer_index(modes_group.attrs["count"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"HDF5 mode group {modes_group.name!r} has no valid count.") from exc
    if mode_count < 0:
        raise ValueError("HDF5 mode count must be nonnegative.")
    modes = tuple(
        _load_mode(_require_member(modes_group, f"{index:06d}"))
        for index in range(mode_count)
    )
    if len(modes_group) != mode_count:
        raise ValueError(f"HDF5 mode count for {modes_group.name!r} is inconsistent.")

    frequency_hz = (
        float(group.attrs["frequency_hz"])
        if "frequency_hz" in group.attrs
        else (float(root_frequency) if np.isfinite(root_frequency) else None)
    )
    if frequency_hz is not None and (not np.isfinite(frequency_hz) or frequency_hz <= 0.0):
        raise ValueError("HDF5 result frequency_hz must be finite and positive.")
    if frequency_hz is not None and np.isfinite(root_frequency) and not np.isclose(
        frequency_hz, root_frequency, rtol=1e-12, atol=0.0
    ):
        raise ValueError("HDF5 result frequency_hz is inconsistent with frequencies_hz.")
    ky = float(group.attrs["ky"]) if "ky" in group.attrs else None
    if ky is not None and not np.isfinite(ky):
        raise ValueError("HDF5 result ky must be finite.")

    return ResultData(
        frequency_hz=frequency_hz,
        ky=ky,
        coordinates=coordinates,
        E_incident=fields["E_incident"],
        E_scattered=fields["E_scattered"],
        E_total=fields["E_total"],
        H_incident=fields["H_incident"],
        H_scattered=fields["H_scattered"],
        H_total=fields["H_total"],
        s_parameters=_load_s_parameters(_require_member(group, "s_parameters")),
        powers=MappingProxyType(powers),
        modes=modes,
        metadata=_metadata(group),
        scene=_load_scene(group["scene"]) if "scene" in group else None,
    )


def load_h5(path: os.PathLike[str] | str) -> FileData:
    """Load and validate one WaveFEM schema-v1 ``.h5``/``.hdf5`` file.

    Legacy schema-v1 results without a ``scene`` group remain valid; their
    :attr:`~wavefem_viewer.model.ResultData.scene` attribute is ``None``.
    """

    source = _source_path(path)
    h5py = _require_h5py()
    try:
        with h5py.File(source, "r") as handle:
            try:
                format_name = _attribute_text(handle.attrs["format"], "format")
            except KeyError as exc:
                raise ValueError("File is missing the WaveFEM format attribute.") from exc
            if format_name != SCHEMA_NAME:
                raise ValueError(f"HDF5 format {format_name!r} is not {SCHEMA_NAME!r}.")
            try:
                version = integer_index(handle.attrs["schema_version"])
            except (KeyError, TypeError) as exc:
                raise ValueError("WaveFEM HDF5 schema_version is missing or invalid.") from exc
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported WaveFEM HDF5 schema version {version}; "
                    f"this viewer supports version {SCHEMA_VERSION}."
                )
            try:
                kind_text = _attribute_text(handle.attrs["kind"], "kind")
            except KeyError as exc:
                raise ValueError("WaveFEM HDF5 kind attribute is missing.") from exc
            if kind_text not in ("single", "sweep"):
                raise ValueError(f"WaveFEM HDF5 kind {kind_text!r} is invalid.")
            kind: Literal["single", "sweep"] = kind_text
            try:
                result_count = integer_index(handle.attrs["result_count"])
            except (KeyError, TypeError) as exc:
                raise ValueError("WaveFEM HDF5 result_count is missing or invalid.") from exc
            if result_count <= 0 or (kind == "single" and result_count != 1):
                raise ValueError("WaveFEM HDF5 result_count is inconsistent with kind.")

            frequencies = _real_array(
                _require_member(handle, "frequencies_hz"),
                "/frequencies_hz",
                allow_nan=True,
            )
            if frequencies.shape != (result_count,):
                raise ValueError("HDF5 frequencies_hz length does not match result_count.")
            if kind == "sweep":
                if (
                    not np.isfinite(frequencies).all()
                    or np.any(frequencies <= 0.0)
                    or np.any(np.diff(frequencies) <= 0.0)
                ):
                    raise ValueError(
                        "Sweep frequencies_hz must be positive and strictly increasing."
                    )
            elif not (np.isfinite(frequencies[0]) or np.isnan(frequencies[0])):
                raise ValueError("Single-result frequency must be positive or unknown.")
            elif np.isfinite(frequencies[0]) and frequencies[0] <= 0.0:
                raise ValueError("Single-result frequency must be positive.")

            results_group = _require_member(handle, "results")
            results = tuple(
                _load_result(
                    _require_member(results_group, f"{index:06d}"),
                    float(frequencies[index]),
                )
                for index in range(result_count)
            )
            if len(results_group) != result_count:
                raise ValueError("HDF5 results group count does not match result_count.")
    except OSError as exc:
        raise ValueError(f"Could not open HDF5 file {source}: {exc}") from exc

    return FileData(path=source, kind=kind, frequencies_hz=frequencies, results=results)


__all__ = ["SCHEMA_NAME", "SCHEMA_VERSION", "load_h5"]

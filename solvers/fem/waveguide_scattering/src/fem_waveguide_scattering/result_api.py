"""Typed scattering archives, lazy sweep cases, and result presentation."""
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
import os
import tempfile
import threading

import h5py
import numpy as np

from cem_common import MeshSnapshot
from cem_common.errors import PersistenceError
from cem_common.persistence import validate_envelope, read_value
from .hdf5 import _load_result, save_result_h5, save_sweep_h5
from .results import ScatteringResult
from .modes import Mode


def _load_case(path, index, frequency):
    with h5py.File(path, "r") as handle:
        validate_envelope(handle, family="waveguide_scattering")
        group = handle["results"][f"{index:06d}"]
        data = _load_result(group, frequency)
        modes = []
        for record in data.modes:
            arguments = {**record.metadata, **record.raw_components}
            modes.append(Mode(**{item.name: arguments[item.name] for item in fields(Mode) if item.name in arguments}))
        names = ("ndofs", "solve_info", "mesh_info", "projection_condition_numbers", "reference_planes", "port_betas")
        result = ScatteringResult(coordinates=data.coordinates, E_incident=data.E_incident,
            E_scattered=data.E_scattered, H_incident=data.H_incident, H_scattered=data.H_scattered,
            s_parameters=data.s_parameters, **data.powers, **{key: data.metadata[key] for key in names},
            frequency_hz=data.frequency_hz, ky=data.ky, modes=tuple(modes), scene=data.scene)
        object.__setattr__(result, "_mesh_snapshot", read_value(group["mesh_snapshot"], {"MeshSnapshot": MeshSnapshot}))
        return result


class _LazyScatteringCases(Sequence):
    def __init__(self, path, frequencies):
        self.path, self.frequencies = Path(path).resolve(), frequencies

    def __len__(self): return len(self.frequencies)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool) or not 0 <= index < len(self):
            raise IndexError("case must be a zero-based result index.")
        return _load_case(self.path, index, self.frequencies[index])


def load_result(path):
    from .sweep import FrequencySweepResult
    try:
        with h5py.File(path, "r") as handle:
            validate_envelope(handle, family="waveguide_scattering")
            frequencies = np.asarray(handle["frequencies_hz"])
            if len(frequencies) != int(handle.attrs["result_count"]):
                raise PersistenceError("Archive result count does not match frequencies.")
            kind = handle.attrs["kind"]
        cases = _LazyScatteringCases(path, frequencies)
        if kind == "single" and len(cases) == 1: return cases[0]
        if kind == "sweep": return FrequencySweepResult(frequencies_hz=frequencies, results=cases)
        raise PersistenceError("Invalid scattering archive kind.")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise PersistenceError(f"Cannot load scattering result from {path}: {exc}") from exc


def save_result(result, path):
    from .sweep import FrequencySweepResult
    if isinstance(result, FrequencySweepResult):
        from .hdf5 import save_sweep_h5
        return save_sweep_h5(result.frequencies_hz, result.results, path,
            modes_per_result=[getattr(case, "modes", ()) for case in result.results])
    from .hdf5 import save_result_h5
    return save_result_h5(result, path)


def plot_result(result, *, component=None, quantity="real"):
    from matplotlib.figure import Figure
    figure = Figure()
    axis = figure.subplots()
    from .sweep import FrequencySweepResult
    if isinstance(result, FrequencySweepResult):
        ax = axis
        transforms = {"real": np.real, "imag": np.imag, "phase": np.angle, "abs": np.abs,
                      "magnitude": np.abs, "db": lambda values: 20*np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))}
        if quantity not in transforms:
            raise ValueError("quantity must be real, imag, phase, magnitude, abs, or db.")
        components = ("S11", "S21") if component is None else (component,)
        for name in components:
            if name not in ("S11", "S21"):
                raise ValueError("A sweep component must be S11 or S21.")
            ax.plot(result.frequencies_hz, transforms[quantity](getattr(result, name)), label=name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(quantity)
        ax.legend()
        return figure
    result._plot_field(component or "Ey", quantity="abs" if quantity == "magnitude" else quantity, ax=axis)
    return figure


def show_result(result, *, block=True):
    from .viewer import launch_viewer
    descriptor, name = tempfile.mkstemp(prefix="cem-scattering-view-", suffix=".h5")
    os.close(descriptor)
    path = Path(name)
    try:
        save_result(result, path)
        process = launch_viewer(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    def cleanup():
        try: process.wait()
        finally: path.unlink(missing_ok=True)
    if block: cleanup()
    else: threading.Thread(target=cleanup, daemon=True).start()
    return process

"""Presentation and persistence of completed waveguide mode results."""
from pathlib import Path
import h5py

from fem_common import MeshSnapshot
from fem_common.errors import PersistenceError
from fem_common.persistence import atomic_h5, write_envelope, validate_envelope, write_value, read_value
from .results import ModeSet, Mode, SampledFields
from .meshing import MeshInfo


def _sampled_fields(*, layout, sample_shape, **values):
    result = SampledFields(**values)
    if result.layout != layout or result.sample_shape != sample_shape:
        raise PersistenceError("Sampled-field layout does not match its coordinates.")
    return result


def save_result(result, path):
    with atomic_h5(path) as handle:
        write_envelope(handle, family="waveguide_modes", kind="modes", dimension=result.dimension,
                       representation="sampled-fields; exp(-i*beta*z)")
        write_value(handle, "result", result)
        write_value(handle, "mesh", result.mesh_data)
    return Path(path)


def load_result(path):
    try:
        with h5py.File(path, "r") as handle:
            validate_envelope(handle, family="waveguide_modes")
            registry = {c.__name__: c for c in (ModeSet, Mode, MeshSnapshot, MeshInfo)}
            registry['SampledFields'] = _sampled_fields
            result = read_value(handle["result"], registry)
            if not isinstance(result, ModeSet):
                raise PersistenceError("Archive does not contain waveguide modes.")
            object.__setattr__(result, "_mesh_snapshot", read_value(handle["mesh"], registry))
            return result
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise PersistenceError(f"Cannot load waveguide mode result from {path}: {exc}") from exc


def plot_result(result, *, component=None, quantity="real", mode=0):
    from .visualization import visualize
    from matplotlib.figure import Figure
    selected = result.mode(mode)
    figure = Figure()
    visualize(selected, component=component or "Ey", quantity=quantity, show=False, axes=figure.subplots())
    return figure


def show_result(result, *, block=True):
    from .visualization import visualize_with_gui
    return visualize_with_gui(result, show=True, block=block)

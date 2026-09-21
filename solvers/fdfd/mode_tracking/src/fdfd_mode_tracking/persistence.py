"""Safe, versioned data-only tracking archives."""
from pathlib import Path
import h5py
from cem_common.persistence import atomic_h5, write_value, read_value
from cem_common.errors import PersistenceError
from cem_common.grid import GridData
from fdfd_waveguide_modes import ModeSet
from .contracts import (PortSpec, TrackingConfig, CandidateSet, TrackingSample, TrackedSweep, PortMode)

REGISTRY = {cls.__name__: cls for cls in (GridData, ModeSet, PortSpec, TrackingConfig,
            CandidateSet, TrackingSample, TrackedSweep, PortMode)}


def save_sweep(sweep, path):
    with atomic_h5(path) as handle:
        handle.attrs.update(format='cem-fdfd-tracking', schema='1.0',
                            time_convention='exp(+i*omega*t)', units='SI')
        write_value(handle, 'sweep', sweep)
    return Path(path)


def load_sweep(path):
    try:
        with h5py.File(path, 'r') as handle:
            for key, expected in {'format': 'cem-fdfd-tracking', 'schema': '1.0',
                                  'time_convention': 'exp(+i*omega*t)', 'units': 'SI'}.items():
                if handle.attrs.get(key) != expected: raise ValueError(f'Invalid {key}.')
            sweep = read_value(handle['sweep'], REGISTRY)
            if not isinstance(sweep, TrackedSweep): raise ValueError('Expected TrackedSweep.')
            for sample in sweep.samples:
                metadata = sample.candidates.result.metadata
                if 'pml' not in metadata or metadata['pml']:
                    raise ValueError('Tracking archive lacks no-PML provenance.')
            return sweep
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PersistenceError(f'Cannot load tracking archive: {exc}') from exc

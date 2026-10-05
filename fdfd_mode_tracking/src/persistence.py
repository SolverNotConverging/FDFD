"""Safe, versioned data-only tracking archives."""
from pathlib import Path
import h5py
from fdfd_common.persistence import atomic_h5, write_value, read_value
from fdfd_common.errors import PersistenceError
from fdfd_common.grid import GridData
from fdfd_waveguide_modes import ModeSet
from .contracts import (PortSpec, TrackingConfig, CandidateSet, TrackingSample, TrackedSweep, PortMode)

REGISTRY = {cls.__name__: cls for cls in (GridData, ModeSet, PortSpec, TrackingConfig,
            CandidateSet, TrackingSample, TrackedSweep, PortMode)}


def _legacy_config(**values):
    """Read pre-removal configuration without retaining the retired control."""
    values.pop('neural_weight', None)
    return TrackingConfig(**values)


def save_sweep(sweep, path):
    with atomic_h5(path) as handle:
        handle.attrs.update(format='cem-fdfd-tracking', schema='1.1',
                            time_convention='exp(+i*omega*t)', units='SI')
        write_value(handle, 'sweep', sweep)
    return Path(path)


def load_sweep(path):
    try:
        with h5py.File(path, 'r') as handle:
            for key, expected in {'format': 'cem-fdfd-tracking',
                                  'time_convention': 'exp(+i*omega*t)', 'units': 'SI'}.items():
                if handle.attrs.get(key) != expected: raise ValueError(f'Invalid {key}.')
            schema = handle.attrs.get('schema')
            if schema not in ('1.0', '1.1'): raise ValueError('Invalid schema.')
            registry = REGISTRY if schema == '1.1' else {**REGISTRY, 'TrackingConfig': _legacy_config}
            sweep = read_value(handle['sweep'], registry)
            if not isinstance(sweep, TrackedSweep): raise ValueError('Expected TrackedSweep.')
            for sample in sweep.samples:
                metadata = sample.candidates.result.metadata
                if 'pml' not in metadata or metadata['pml']:
                    raise ValueError('Tracking archive lacks no-PML provenance.')
            return sweep
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PersistenceError(f'Cannot load tracking archive: {exc}') from exc

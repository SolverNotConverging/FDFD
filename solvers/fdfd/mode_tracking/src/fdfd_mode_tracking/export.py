"""Certified frequency-domain port fields; no time-domain engine dependency."""
import numpy as np
from .contracts import PortMode


def _port_mode(sample, index, phase, port, amplitude, track):
    candidate = sample.candidates
    if not candidate.eligible[index]:
        raise ValueError(f'Mode at {sample.frequency:g} Hz is not injection eligible: {candidate.evidence[index]}')
    fields = {n: np.array(v[..., index]*phase*amplitude, copy=True) for n, v in candidate.fields.items()}
    # z reflection: E_t and H_z even; E_z and H_t odd. Never flip beta alone.
    if port.direction == -1:
        for name in ('Ez', 'Hx', 'Hy'): fields[name] *= -1
    result = candidate.result
    return PortMode(sample.frequency, complex(port.direction*result.beta[index]), fields,
                    result.field_coordinates, complex(port.direction*candidate.complex_power[index]*abs(amplitude)**2),
                    {'track': track, 'candidate': int(index), 'reference_plane_m': port.reference_plane,
                     'direction': port.direction, 'decay_direction': '+z' if port.direction == 1 else '-z',
                     'propagation': candidate.propagation[index], 'confinement': candidate.confinement[index],
                     'normalization': 'combined_E_H_RMS_1_V_per_m', 'spectral_amplitude': complex(amplitude),
                     'phase': complex(phase), 'power_units': 'W' if len(result.mesh_data.axes) == 2 else 'W/m',
                     'field_units': {'E': 'V/m', 'H': 'A/m'}, 'eta0': result.metadata['eta0'],
                     'time_convention': 'exp(+i*omega*t)', 'spatial_convention': 'exp(-i*beta*z)',
                     'pml': (), 'boundaries': result.metadata['boundaries'],
                     'evidence': candidate.evidence[index], 'context': result.metadata.get('context')})


def export_track(sweep, track=0, *, frequencies=None, amplitude=1.0):
    """Export sampled profiles only; never interpolate cutoff or missing modes.

    An explicit frequency subset declares exclusion of unsampled/unresolved
    spectral intervals. The result is discrete data, not a continuous-band source.
    """
    if isinstance(track, bool) or int(track) != track or track < 0:
        raise ValueError('track must be a nonnegative integer.')
    if not np.isfinite(amplitude): raise ValueError('amplitude must be finite.')
    frequencies = sweep.requested_frequencies if frequencies is None else tuple(frequencies)
    if not frequencies: raise ValueError('At least one export frequency is required.')
    samples = {s.frequency: s for s in sweep.samples}
    output = []
    for frequency in frequencies:
        if frequency not in samples: raise ValueError(f'No validated sample at {frequency:g} Hz.')
        sample = samples[frequency]
        if track >= len(sample.candidate_indices): raise ValueError('track index out of range.')
        index = sample.candidate_indices[track]
        if index < 0: raise ValueError(f'Track is unmatched at {frequency:g} Hz.')
        if any(lo < frequency < hi for lo, hi in sweep.unresolved_intervals):
            raise ValueError('Sample lies inside an unresolved spectral interval.')
        if any(e['type'] == 'reverse_audit_unresolved' and e['interval'][0] <= frequency <= e['interval'][1]
               for e in sweep.events):
            raise ValueError('Forward/reverse identity audit is unresolved at this sample.')
        if any(track in cluster['tracks'] for cluster in sample.clusters):
            raise ValueError('Cluster identity is a subspace; use export_subspace with explicit coefficients.')
        value = _port_mode(sample, index, sample.phases[track], sweep.port, amplitude, track)
        value.metadata['unresolved_intervals_hz'] = sweep.unresolved_intervals
        value.metadata['continuous_band_certified'] = not any(
            min(frequencies) < hi and max(frequencies) > lo for lo, hi in sweep.unresolved_intervals)
        output.append(value)
    return tuple(output)


def export_subspace(sweep, frequency, cluster_index, coefficients):
    """Export a smooth-basis excitation as explicit raw-eigenmode terms.

    Each term retains its own beta; no near-degenerate mixture is called a
    single eigenmode. Coefficients refer to the stored orthonormal smooth basis.
    """
    sample = next((s for s in sweep.samples if s.frequency == frequency), None)
    if sample is None: raise ValueError('Frequency is not sampled.')
    if any(lo < frequency < hi for lo, hi in sweep.unresolved_intervals):
        raise ValueError('Sample lies inside an unresolved spectral interval.')
    if any(e['type'] == 'reverse_audit_unresolved' and e['interval'][0] <= frequency <= e['interval'][1]
           for e in sweep.events):
        raise ValueError('Forward/reverse identity audit is unresolved at this sample.')
    if isinstance(cluster_index, bool) or int(cluster_index) != cluster_index or not 0 <= cluster_index < len(sample.clusters):
        raise ValueError('cluster index out of range.')
    cluster = sample.clusters[cluster_index]
    coefficients = np.asarray(coefficients, complex)
    if coefficients.shape != (len(cluster['tracks']),) or not np.isfinite(coefficients).all():
        raise ValueError('Provide one finite complex coefficient per smooth basis vector.')
    amplitudes = cluster['transform'] @ coefficients
    output = []
    for index, amplitude in zip(cluster['candidates'], amplitudes):
        mode = _port_mode(sample, index, 1., sweep.port, amplitude, None)
        mode.metadata.update(cluster_index=cluster_index, smooth_basis_coefficients=coefficients,
                             basis_transform=cluster['transform'], representation='eigenmode_superposition_term')
        output.append(mode)
    return tuple(output)

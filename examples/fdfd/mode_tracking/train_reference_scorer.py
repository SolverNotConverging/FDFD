"""Generate an analytical plate dataset and demonstrate optional neural scoring.

This narrow corpus demonstrates the workflow; it does not qualify a model for
new guide families, degeneracies, radiation rejection, or production use.
"""
import json
from pathlib import Path
import numpy as np
from cem_common.persistence import atomic_h5, write_value
from fdfd_mode_tracking import PortSpec, TrackingConfig, track_modes
from fdfd_mode_tracking.neural import PairDataset, reference_pairs, train_scorer
from parallel_plate_cutoff import plate_factory, C

OUTPUT = Path(__file__).resolve().parents[3] / 'outputs/examples/fdfd/mode_tracking/reference_scorer'


def analytical_labels(candidate, width):
    expected = (np.arange(1, 3)*C/(2*width*candidate.frequency))**2-1
    labels = []
    for i, value in enumerate(candidate.eigenvalues):
        closest = int(np.argmin(abs(expected-value)))
        labels.append(f'TE{closest+1}' if candidate.eligible[i] and abs(expected[closest]-value) < .003 else None)
    if len([x for x in labels if x is not None]) != len(set(x for x in labels if x is not None)):
        raise ValueError('Analytical labels are not one-to-one.')
    return labels


def main():
    parts, manifest = [], []
    for width in np.linspace(.020, .030, 6):
        cutoff = C/(2*width)
        frequencies = cutoff*np.array([.75, .85, 1.15, 1.25])
        sweep = track_modes(plate_factory(width), frequencies, port=PortSpec(boundary='enclosed'),
                            config=TrackingConfig(num_candidates=2, max_candidates=3, polarization='TE',
                                neff_guess=-.8j, max_depth=3), seed_modes=(0, 1))
        geometry = f'plates_{width:.8f}m'
        wanted = [s.candidates for s in sweep.samples if s.frequency in frequencies]
        for left, right in zip(wanted, wanted[1:]):
            parts.append(reference_pairs(left, right, analytical_labels(left, width), analytical_labels(right, width),
                geometry_id=geometry, evidence='PEC plate TE_m: lambda=(m*c/(2*w*f))^2-1; independent 2x mesh verification'))
        manifest.append({'geometry_id': geometry, 'width_m': float(width), 'cells': 128,
                         'frequencies_hz': frequencies.tolist(), 'pml': False})
    dataset = PairDataset(np.concatenate([p.features for p in parts]), np.concatenate([p.labels for p in parts]),
                          sum((p.geometry_ids for p in parts), ()), sum((p.evidence for p in parts), ()))
    model, report = train_scorer(dataset, seed=17)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with atomic_h5(OUTPUT/'dataset.h5') as handle:
        handle.attrs.update(format='cem-mode-pair-dataset', schema='pair-invariants-v1')
        write_value(handle, 'dataset', dataset)
        write_value(handle, 'manifest', manifest)
    model.save(OUTPUT/'scorer.h5')
    (OUTPUT/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()

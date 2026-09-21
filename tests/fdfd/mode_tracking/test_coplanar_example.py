"""Geometry and smoke coverage for the open CPW example."""
from pathlib import Path
import runpy

import numpy as np
import pytest

from fdfd_mode_tracking import TrackingConfig, VerificationSpec, load_sweep


def example_module():
    root = Path(__file__).resolve().parents[3]
    return runpy.run_path(str(
        root / 'examples/fdfd/mode_tracking/tracked_coplanar_waveguide_2d.py'))


def test_coplanar_default_air_clearance_and_mesh():
    tracker = example_module()['build_tracker'](frequencies=[10e9])
    assert tracker.port.boundary == 'open'
    np.testing.assert_allclose(tracker.x_range, (-.014, .014))
    np.testing.assert_allclose(tracker.y_range, (-.008, .0094))
    assert tracker.mesh_data.resolution == (280, 174)
    assert len(tracker._objects) == 4  # substrate, strip, two grounds; no walls
    for spec in (VerificationSpec(mesh_factor=2), VerificationSpec(padding_fraction=.5)):
        solver = tracker._factory(10e9, spec)
        assert len(solver._objects) == 4
        for original, variant in zip(tracker._objects.values(), solver._objects.values()):
            assert original[0].shape == variant[0].shape
    for kwargs in ({'air_padding': 0}, {'cell_size': -1}, {'cell_size': np.nan}):
        with pytest.raises(ValueError):
            example_module()['build_tracker'](**kwargs)


def test_coplanar_example_tracks_and_saves(tmp_path):
    # Smaller smoke mesh, not the demonstration's fine default grid.
    tracker = example_module()['build_tracker'](
        frequencies=[10e9, 11e9], air_padding=4e-3, cell_size=.2e-3)
    sweep = tracker.solve(
        num_modes=3,
        tracking_config=TrackingConfig(
            max_depth=0, verification_overlap=.995,
            verification_beta_tolerance=.01,
        ),
    )
    assert len(sweep.samples) == 2
    for sample in sweep.samples:
        candidates = sample.candidates
        assert len(candidates.result) == 3
        assert candidates.result.metadata['pml'] == ()
        assert not candidates.result.metadata['boundaries']['closed_conductor_boundary']
        assert {check['kind'] for check in candidates.evidence[0]['verification']} == {
            'mesh', 'padding_1', 'padding_2'}
        assert candidates.eligible[0] == (candidates.numerical_valid[0]
                                         and candidates.confinement[0] == 'bound')
        assert candidates.propagation[0] == 'propagating'
        assert set(sample.candidate_indices) == {0, 1, 2}
        assert np.isfinite(candidates.vectors).all()
    assert sweep.samples[0].candidate_indices[0] == sweep.samples[1].candidate_indices[0]
    archive = tmp_path / 'tracked_modes.h5'
    sweep.save(archive)
    restored = load_sweep(archive)
    assert restored.port.name == 'open coplanar-waveguide port'
    np.testing.assert_allclose(restored.samples[-1].candidates.result.neff,
                               sweep.samples[-1].candidates.result.neff)

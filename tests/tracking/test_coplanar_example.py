"""Geometry and smoke coverage for the open CPW example."""

import ast
from pathlib import Path

import numpy as np

from fdfd import load as load_sweep
from fdfd.tracking import TrackingConfig, VerificationSpec


def example_tracker(*, frequencies=None, air_padding=8e-3, cell_size=0.1e-3):
    """Run the example geometry before its demonstration sweep, with a smoke mesh."""
    root = Path(__file__).resolve().parents[2]
    path = root / "examples/tracking/2d_tracked_coplanar_waveguide.py"
    tree = ast.parse(path.read_text())
    settings = {"frequencies", "air_padding", "cell_size"}
    prefix = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if "sweep" in names:
                break
            if names & settings:
                continue
        prefix.append(node)
    scope = dict(__file__=str(path), frequencies=frequencies, air_padding=air_padding, cell_size=cell_size)
    exec(compile(ast.Module(body=prefix, type_ignores=[]), str(path), "exec"), scope)
    return scope["tracker"]


def test_coplanar_air_clearance_and_mesh():
    tracker = example_tracker(frequencies=[10e9])
    assert tracker.port.boundary == "open"
    np.testing.assert_allclose(tracker.x_range, (-0.014, 0.014))
    np.testing.assert_allclose(tracker.y_range, (-0.008, 0.0094))
    assert tracker.grid.resolution == (280, 174)
    assert len(tracker._objects) == 4  # substrate, strip, two grounds; no walls
    for spec in (VerificationSpec(mesh_factor=2), VerificationSpec(padding_fraction=0.5)):
        solver = tracker._factory(10e9, spec)
        assert len(solver._objects) == 4
        for original, variant in zip(tracker._objects.values(), solver._objects.values(), strict=True):
            assert original.shape == variant.shape


def test_coplanar_example_tracks_and_saves(tmp_path):
    # Smaller smoke mesh, not the demonstration's fine default grid.
    tracker = example_tracker(frequencies=[10e9, 11e9], air_padding=4e-3, cell_size=0.2e-3)
    sweep = tracker.solve(
        num_modes=3,
        config=TrackingConfig(
            max_depth=0,
            verification_overlap=0.995,
            verification_beta_tolerance=0.01,
        ),
    )
    assert len(sweep.samples) == 2
    for sample in sweep.samples:
        candidates = sample.candidates
        assert len(candidates.result) == 3
        assert candidates.result.metadata["pml"] == ()
        assert not candidates.result.metadata["boundaries"]["closed_conductor_boundary"]
        assert {check["kind"] for check in candidates.evidence[0]["verification"]} == {
            "mesh",
            "padding_1",
            "padding_2",
        }
        assert candidates.eligible[0] == (
            candidates.numerical_valid[0] and candidates.confinement[0] == "bound"
        )
        assert candidates.propagation[0] == "propagating"
        assert set(sample.candidate_indices) == {0, 1, 2}
        assert np.isfinite(candidates.vectors).all()
    assert sweep.samples[0].candidate_indices[0] == sweep.samples[1].candidate_indices[0]
    archive = tmp_path / "tracked_modes.h5"
    sweep.save(archive)
    restored = load_sweep(archive)
    assert restored.port.name == "open coplanar-waveguide port"
    np.testing.assert_allclose(
        restored.samples[-1].candidates.result.neff, sweep.samples[-1].candidates.result.neff
    )

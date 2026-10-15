"""Persistent spans, resolved crossings and physical sector regressions."""

import numpy as np
import pytest

from fdfd.tracking import TrackingConfig
from fdfd.tracking.assignment import assign_modes, match_clusters, mode_groups
from fdfd.tracking.metrics import compatible_character


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], complex)


def test_defaults_and_complete_link():
    config = TrackingConfig()
    assert config.overlap_min == 0.9
    assert mode_groups(np.array([0, 0.75e-8, 1.5e-8]), config) == ((0, 1), (2,))
    with pytest.raises(ValueError):
        TrackingConfig(degeneracy_tolerance=0)


def test_close_but_resolved_modes_cannot_use_subspace_exemption():
    values = np.array([1, 1 + 1e-6])
    vectors = np.eye(2)
    new = rotation(np.pi / 4)
    config = TrackingConfig()
    assert not match_clusters(vectors, new, values, values, config)
    ids, _, _ = assign_modes(vectors, new, values, values, config)
    assert np.all(ids == -1)


def test_prediction_separates_accidental_crossing():
    assert not match_clusters(
        np.eye(2), rotation(0.8), np.ones(2), np.ones(2), TrackingConfig(), prediction=np.array([0.9, 1.1])
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_two_rotating_pairs_survive_fourfold_crossing(reverse):
    old = np.eye(4, dtype=complex)
    new = np.zeros((4, 4), complex)
    new[:2, :2] = rotation(0.7)
    new[2:, 2:] = rotation(1.1) * 1j
    a, b = np.array([1, 1, 2, 2]), np.ones(4) * 1.5
    if reverse:
        old, new, a, b = new, old, b, a
    records = match_clusters(old, new, a, b, TrackingConfig())
    assert len(records) == 2
    for record in records:
        assert len(record["old_members"]) == 2
        assert record["overlap"] == pytest.approx(1)
        np.testing.assert_allclose(record["aligned"], old[:, record["old_members"]], atol=1e-14)


def test_fully_mixed_crossing_abstains_and_rank_loss_rejected():
    mixed = np.fft.fft(np.eye(4)) / 2
    config = TrackingConfig()
    assert not match_clusters(np.eye(4), mixed, np.array([1, 1, 2, 2]), np.ones(4), config)
    assert not match_clusters(np.eye(2), np.ones((2, 2)), np.ones(2), np.ones(2), config)


def test_te_tm_groups_never_rotate_together():
    config = TrackingConfig()
    labels = ["TE", "TM"]
    assert mode_groups(np.ones(2), config, labels=labels) == ((0,), (1,))
    assert not match_clusters(
        np.eye(2), rotation(0.7), np.ones(2), np.ones(2), config, old_labels=labels, new_labels=labels
    )


def test_definite_symmetry_and_eh_sectors():
    te = {"family": "TE", "mirror_parity": (1, -1)}
    tm = {"family": "TM", "mirror_parity": (1, -1)}
    odd = {"family": "TE", "mirror_parity": (-1, -1)}
    unknown = {"family": "hybrid", "mirror_parity": (0, 0)}
    assert compatible_character(te, te)
    assert not compatible_character(te, tm)
    assert not compatible_character(te, odd)
    assert compatible_character(te, unknown)

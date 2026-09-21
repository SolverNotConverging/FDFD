import numpy as np
import pytest
from fdfd_mode_tracking.neural import PairDataset, train_scorer, load_scorer


def test_geometry_split_calibration_and_model_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(240, 7))
    # Synthetic algorithm test, not physical training data or a model qualification.
    y = np.tile([0., 1.], 120)
    x[:, 0] = .1+.8*y+rng.normal(0, .01, len(y))
    groups = tuple(f'geometry_{i//40}' for i in range(len(y)))
    dataset = PairDataset(x, y, groups, ('synthetic_known_correspondence',)*len(y))
    model, report = train_scorer(dataset, max_iterations=100)
    train, cal, test = (set(report[key]) for key in ('train_geometries', 'calibration_geometries', 'test_geometries'))
    assert not (train & cal or train & test or cal & test)
    assert report['test_accuracy'] > .95
    assert not report['enabled_by_default']
    model.save(tmp_path/'model.h5')
    loaded = load_scorer(tmp_path/'model.h5')
    np.testing.assert_allclose(loaded.predict(x), model.predict(x))
    with pytest.raises(ValueError, match='out-of-distribution'):
        loaded.predict(x+1000)


def test_no_frequency_only_train_test_split():
    x = np.zeros((10, 7))
    dataset = PairDataset(x, np.tile([0., 1.], 5), ('same_geometry',)*10, ('reference',)*10)
    with pytest.raises(ValueError, match='three independent'):
        train_scorer(dataset)

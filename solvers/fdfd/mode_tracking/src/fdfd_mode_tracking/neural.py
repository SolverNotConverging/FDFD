"""Optional small MLP, explicit reference labels and geometry-held-out calibration.

Imported only by callers that train/load a scorer. Uses NumPy/SciPy; no deep
learning runtime or pretrained model is required by the conventional tracker.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from .assignment import pair_features

FEATURE_SCHEMA = 'pair-invariants-v1'


@dataclass
class PairDataset:
    features: np.ndarray
    labels: np.ndarray
    geometry_ids: tuple
    evidence: tuple


def reference_pairs(left, right, left_labels, right_labels, *, geometry_id, evidence):
    """Build pair data from explicit independently verified branch labels.

    None labels are unresolved and excluded. Eigenvalue rank and tracker-assigned
    indices must not be used as ground truth. Keep all mesh/padding variants of
    a geometry under the same geometry_id.
    """
    if not geometry_id or not evidence:
        raise ValueError('Reference labels require a geometry ID and verification evidence.')
    if len(left_labels) != len(left.result) or len(right_labels) != len(right.result):
        raise ValueError('Labels must follow candidate order.')
    overlap = abs(left.vectors.conj().T @ right.vectors)**2
    features = pair_features(overlap, left.eigenvalues, right.eigenvalues, left.eigenvalues,
                             abs(right.frequency-left.frequency)/left.frequency)
    keep, targets = [], []
    for i, a in enumerate(left_labels):
        for j, b in enumerate(right_labels):
            if a is None or b is None: continue
            if a == b and not (left.eligible[i] and right.eligible[j]):
                raise ValueError('Positive reference pairs require verified bound candidates.')
            keep.append(features[i, j])
            targets.append(float(a == b))
    return PairDataset(np.asarray(keep).reshape(-1, 7), np.asarray(targets),
                       (str(geometry_id),)*len(keep), (str(evidence),)*len(keep))


@dataclass
class PairScorer:
    mean: np.ndarray
    scale: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float
    temperature: float
    schema: str = FEATURE_SCHEMA

    def predict(self, features):
        features = np.asarray(features, float)
        if self.schema != FEATURE_SCHEMA or features.shape[-1] != len(self.mean):
            raise ValueError('unsupported feature schema')
        z = (features-self.mean)/self.scale
        if not np.isfinite(z).all() or np.any(abs(z) > 8.):
            raise ValueError('out-of-distribution pair features')
        return expit((np.tanh(z @ self.w1+self.b1) @ self.w2+self.b2)/self.temperature)

    def save(self, path):
        from cem_common.persistence import atomic_h5, write_value
        with atomic_h5(path) as handle:
            handle.attrs.update(format='cem-mode-pair-scorer', schema=FEATURE_SCHEMA)
            write_value(handle, 'model', self)


def load_scorer(path):
    import h5py
    from cem_common.persistence import read_value
    with h5py.File(path, 'r') as handle:
        if handle.attrs.get('format') != 'cem-mode-pair-scorer' or handle.attrs.get('schema') != FEATURE_SCHEMA:
            raise ValueError('Unsupported scorer archive.')
        model = read_value(handle['model'], {'PairScorer': PairScorer})
    if not isinstance(model, PairScorer) or not np.isfinite(model.temperature) or model.temperature <= 0:
        raise ValueError('Invalid scorer calibration.')
    return model


def train_scorer(dataset, *, seed=0, hidden=12, max_iterations=300):
    """Fit train geometries; temperature-calibrate on separate geometries.

    Returns model and untouched test-set metrics, including split IDs. At least
    three independent geometry groups are needed. No production-accuracy claim
    is implied by successful optimization.
    """
    x, y = np.asarray(dataset.features, float), np.asarray(dataset.labels, float)
    groups = np.asarray(dataset.geometry_ids)
    if x.ndim != 2 or x.shape != (len(y), 7) or len(groups) != len(y) or len(dataset.evidence) != len(y):
        raise ValueError('Inconsistent pair dataset.')
    if not np.isfinite(x).all() or not np.isin(y, [0., 1.]).all() or not all(dataset.evidence):
        raise ValueError('Features/labels/evidence must be finite, binary and verified.')
    unique = np.unique(groups)
    if len(unique) < 3: raise ValueError('At least three independent geometry groups are required.')
    rng = np.random.default_rng(seed)
    unique = rng.permutation(unique)
    n_hold = max(1, len(unique)//5)
    test_ids, calibration_ids, train_ids = unique[:n_hold], unique[n_hold:2*n_hold], unique[2*n_hold:]
    train, calibration, test = (np.isin(groups, ids) for ids in (train_ids, calibration_ids, test_ids))
    if any(len(np.unique(y[mask])) != 2 for mask in (train, calibration, test)):
        raise ValueError('Every geometry split must contain positive and negative pairs.')
    mean, scale = x[train].mean(axis=0), np.maximum(x[train].std(axis=0), 1e-6)
    z = (x-mean)/scale
    d = x.shape[1]
    def unpack(theta):
        a = d*hidden
        return theta[:a].reshape(d, hidden), theta[a:a+hidden], theta[a+hidden:a+2*hidden], theta[-1]
    theta = np.r_[rng.normal(0, .1, d*hidden), np.zeros(hidden), rng.normal(0, .1, hidden), 0.]
    def objective(theta):
        w1, b1, w2, b2 = unpack(theta)
        h = np.tanh(z[train] @ w1+b1)
        logits = h @ w2+b2
        error = (expit(logits)-y[train])/train.sum()
        dh = error[:, None]*w2*(1-h*h)
        loss = np.mean(np.logaddexp(0., logits)-y[train]*logits) + 1e-4*np.dot(theta, theta)
        grad = np.r_[(z[train].T @ dh).ravel(), dh.sum(axis=0), h.T @ error, error.sum()]+2e-4*theta
        return loss, grad
    fit = minimize(objective, theta, jac=True, method='L-BFGS-B', options={'maxiter': max_iterations})
    w1, b1, w2, b2 = unpack(fit.x)
    logits = np.tanh(z @ w1+b1) @ w2+b2
    calibration_fit = minimize_scalar(lambda log_t: np.mean(
        np.logaddexp(0, logits[calibration]/np.exp(log_t))-y[calibration]*logits[calibration]/np.exp(log_t)),
        bounds=(-3., 3.), method='bounded')
    temperature = float(np.exp(calibration_fit.x))
    p = expit(logits[test]/temperature)
    model = PairScorer(mean, scale, w1, b1, w2, float(b2), temperature)
    report = {'seed': seed, 'train_geometries': tuple(train_ids), 'calibration_geometries': tuple(calibration_ids),
              'test_geometries': tuple(test_ids), 'optimization_converged': bool(fit.success),
              'test_brier_score': float(np.mean((p-y[test])**2)),
              'test_accuracy': float(np.mean((p >= .5) == y[test])),
              'test_false_confident_matches': int(np.sum((p >= .95) & (y[test] == 0))),
              'feature_schema': FEATURE_SCHEMA, 'enabled_by_default': False}
    return model, report

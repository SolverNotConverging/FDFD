"""Joint branch assignment with explicit abstention and subspace transport."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .metrics import align_subspaces


def eigen_clusters(values, gap):
    """Connected components of the local eigenvalue-gap graph."""
    remaining = set(range(len(values)))
    groups = []
    while remaining:
        group = {min(remaining)}
        while True:
            extra = {j for j in remaining-group if any(
                abs(values[j]-values[i]) <= gap*max(1., abs(values[j]), abs(values[i])) for i in group)}
            if not extra: break
            group.update(extra)
        remaining -= group
        groups.append(tuple(sorted(group)))
    return groups


def assign_modes(old_vectors, new_vectors, old_values, new_values, config,
                 *, prediction=None, allowed=None):
    overlap = np.clip(abs(old_vectors.conj().T @ new_vectors)**2, 0., 1.)
    prediction = old_values if prediction is None else prediction
    error = abs(new_values[None, :]-prediction[:, None])/np.maximum(1., abs(old_values))[:, None]
    cost = 1-overlap + .1*np.minimum(error**2, 4.)
    allowed = np.ones(cost.shape, bool) if allowed is None else np.asarray(allowed, bool)
    cost = np.where(allowed & (overlap >= config.overlap_min), cost, 1e6)
    count = len(old_values)
    augmented = np.column_stack((cost, np.full((count, count), config.unmatched_cost)))
    rows, columns = linear_sum_assignment(augmented)
    best = float(augmented[rows, columns].sum())
    matches = np.full(count, -1, int)
    margins = np.zeros(count)
    for row, col in zip(rows, columns):
        if col >= len(new_values) or cost[row, col] >= config.unmatched_cost: continue
        alternative = augmented.copy()
        alternative[row, col] = 1e6
        ar, ac = linear_sum_assignment(alternative)
        margins[row] = float(alternative[ar, ac].sum()-best)
        if margins[row] >= config.assignment_margin:
            matches[row] = col
    return matches, overlap, margins


def match_clusters(old_vectors, new_vectors, old_values, new_values, config):
    """Match equal-rank eigenvalue clusters jointly; reserve all their members."""
    old = [g for g in eigen_clusters(old_values, config.cluster_gap) if len(g) > 1]
    new = [g for g in eigen_clusters(new_values, config.cluster_gap) if len(g) > 1]
    if not old or not new: return ()
    cost = np.full((len(old), len(new)+len(old)), config.unmatched_cost)
    details = {}
    for i, a in enumerate(old):
        for j, b in enumerate(new):
            cost[i, j] = 1e6
            if len(a) != len(b): continue
            record = align_subspaces(old_vectors[:, a], new_vectors[:, b])
            if record['old_rank'] != len(a) or record['new_rank'] != len(b): continue
            score = float(np.min(record['singular_values'])**2)
            if score < config.overlap_min: continue
            drift = abs(np.mean(old_values[list(a)])-np.mean(new_values[list(b)]))/max(1., abs(np.mean(old_values[list(a)])))
            cost[i, j] = 1-score + .1*min(drift**2, 4.)
            details[i, j] = {**record, 'old_members': a, 'new_members': b, 'overlap': score}
    rows, columns = linear_sum_assignment(cost)
    best = cost[rows, columns].sum()
    result = []
    for i, j in zip(rows, columns):
        if (i, j) not in details or cost[i, j] >= config.unmatched_cost: continue
        alt = cost.copy()
        alt[i, j] = 1e6
        ar, ac = linear_sum_assignment(alt)
        if alt[ar, ac].sum()-best >= config.assignment_margin:
            result.append(details[i, j])
    return tuple(result)

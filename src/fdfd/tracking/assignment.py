"""Joint branch assignment with explicit abstention and subspace transport."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metrics import align_subspaces, orthonormal_basis


def assign_modes(old_vectors, new_vectors, old_values, new_values, config, *, prediction=None, allowed=None):
    overlap = np.clip(abs(old_vectors.conj().T @ new_vectors) ** 2, 0.0, 1.0)
    prediction = old_values if prediction is None else prediction
    error = abs(new_values[None, :] - prediction[:, None]) / np.maximum(1.0, abs(old_values))[:, None]
    cost = 1 - overlap + 0.1 * np.minimum(error**2, 4.0)
    allowed = np.ones(cost.shape, bool) if allowed is None else np.asarray(allowed, bool)
    cost = np.where(allowed & (overlap >= config.overlap_min), cost, 1e6)
    count = len(old_values)
    augmented = np.column_stack((cost, np.full((count, count), config.unmatched_cost)))
    rows, columns = linear_sum_assignment(augmented)
    best = float(augmented[rows, columns].sum())
    matches = np.full(count, -1, int)
    margins = np.zeros(count)
    for row, col in zip(rows, columns, strict=True):
        if col >= len(new_values) or cost[row, col] >= config.unmatched_cost:
            continue
        alternative = augmented.copy()
        alternative[row, col] = 1e6
        ar, ac = linear_sum_assignment(alternative)
        margins[row] = float(alternative[ar, ac].sum() - best)
        if margins[row] >= config.assignment_margin:
            matches[row] = col
    return matches, overlap, margins


def mode_groups(values, config, *, prediction=None, labels=None):
    """Complete-link numerical degeneracies, separated by known solver sectors.

    A close eigenvalue neighbourhood alone does not authorize basis rotation.
    Every pair must remain degenerate in the prediction as well as the anchor.
    """
    records = np.atleast_2d(values) if prediction is None else np.stack((values, prediction))
    tolerance = min(config.cluster_gap, config.degeneracy_tolerance)
    groups = []
    for i in range(records.shape[1]):
        for group in groups:
            if labels is not None and any(labels[i] != labels[j] for j in group):
                continue
            a, b = records[:, i, None], records[:, group]
            scale = np.maximum(1.0, np.maximum(abs(a), abs(b)))
            if np.all(np.isfinite(a)) and np.all(abs(a - b) <= tolerance * scale):
                group.append(i)
                break
        else:
            groups.append([i])
    return tuple(tuple(g) for g in groups)


def _subset(reference, vectors, members, margin):
    rank = reference.shape[1]
    q, _ = orthonormal_basis(reference)
    if q.shape[1] != rank or len(members) < rank:
        return None
    columns = vectors[:, members]
    norms = np.sum(abs(columns) ** 2, axis=0)
    if np.any(norms <= 0) or not np.all(np.isfinite(columns)):
        return None
    scores = np.sum(abs(q.conj().T @ columns) ** 2, axis=0) / norms
    order = np.argsort(-scores, kind="stable")
    if len(members) > rank and scores[order[rank - 1]] - scores[order[rank]] <= margin:
        return None
    return tuple(sorted(members[k] for k in order[:rank]))


def match_clusters(
    old_vectors,
    new_vectors,
    old_values,
    new_values,
    config,
    *,
    prediction=None,
    old_labels=None,
    new_labels=None,
):
    """Transport persistent spans, including identifiable subsets at crossings.

    Proposals competing for any member must have a clear cost margin. A fully
    mixed larger span has no identifiable raw-column subset: abstain there.
    """
    prediction = old_values if prediction is None else prediction
    old = [g for g in mode_groups(old_values, config, prediction=prediction, labels=old_labels) if len(g) > 1]
    new = [g for g in mode_groups(new_values, config, labels=new_labels) if len(g) > 1]
    proposals = []
    for a in old:
        for b in new:
            if old_labels is not None and new_labels is not None and old_labels[a[0]] != new_labels[b[0]]:
                continue
            left, right = a, b
            if len(a) <= len(b):
                right = _subset(old_vectors[:, a], new_vectors, b, config.subspace_margin)
            else:
                left = _subset(new_vectors[:, b], old_vectors, a, config.subspace_margin)
            if left is None or right is None:
                continue
            record = align_subspaces(old_vectors[:, left], new_vectors[:, right])
            if record["old_rank"] != len(left) or record["new_rank"] != len(right):
                continue
            score = float(np.clip(np.min(record["singular_values"]) ** 2, 0, 1))
            drift = abs(np.mean(prediction[list(left)]) - np.mean(new_values[list(right)]))
            drift /= max(1.0, abs(np.mean(old_values[list(left)])))
            cost = 1 - score + 0.1 * min(drift**2, 4.0)
            if score >= config.overlap_min and cost < config.unmatched_cost:
                proposals.append(
                    (float(cost), {**record, "old_members": left, "new_members": right, "overlap": score})
                )
    result = []
    used_old, used_new = set(), set()
    for cost, record in sorted(proposals, key=lambda item: item[0]):
        a, b = set(record["old_members"]), set(record["new_members"])
        if a & used_old or b & used_new:
            continue
        competitors = [
            c
            for c, r in proposals
            if r is not record and (a.intersection(r["old_members"]) or b.intersection(r["new_members"]))
        ]
        if min([config.unmatched_cost, *competitors]) - cost < config.assignment_margin:
            continue
        result.append(record)
        used_old.update(a)
        used_new.update(b)
    return tuple(sorted(result, key=lambda record: record["old_members"]))

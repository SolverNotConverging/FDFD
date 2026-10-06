"""Locations and rectangular operators for cell and node Yee samples."""
import numpy as np
from scipy.sparse import diags


def node_average(values, axis, *, periodic=False):
    """Interpolate cell values to the bounding nodes along one axis."""
    values = np.asarray(values)
    if periodic:
        return .5*(values + np.roll(values, 1, axis=axis))
    shape = list(values.shape)
    shape[axis] += 1
    out = np.zeros(shape, dtype=np.result_type(values.dtype, np.float64))
    counts = np.zeros(shape, dtype=float)
    lower = [slice(None)]*values.ndim
    upper = lower.copy()
    lower[axis], upper[axis] = slice(None, -1), slice(1, None)
    out[tuple(lower)] += values
    out[tuple(upper)] += values
    counts[tuple(lower)] += 1
    counts[tuple(upper)] += 1
    return out/counts


def occupied_nodes(mask, axis, *, periodic=False):
    """Include every node touching an occupied cell (including the seam)."""
    mask = np.asarray(mask, dtype=bool)
    if periodic:
        return mask | np.roll(mask, 1, axis=axis)
    shape = list(mask.shape)
    shape[axis] += 1
    out = np.zeros(shape, dtype=bool)
    lower = [slice(None)]*mask.ndim
    upper = lower.copy()
    lower[axis], upper[axis] = slice(None, -1), slice(1, None)
    out[tuple(lower)] |= mask
    out[tuple(upper)] |= mask
    return out


def interior_nodes(mask, axis, *, periodic=False):
    """Nodes strictly inside occupied cells; exclude material interfaces."""
    mask = np.asarray(mask, dtype=bool)
    if periodic:
        return mask & np.roll(mask, 1, axis=axis)
    shape = list(mask.shape)
    shape[axis] += 1
    out = np.zeros(shape, dtype=bool)
    middle = [slice(None)]*mask.ndim
    lower, upper = middle.copy(), middle.copy()
    middle[axis], lower[axis], upper[axis] = slice(1, -1), slice(None, -1), slice(1, None)
    out[tuple(middle)] = mask[tuple(lower)] & mask[tuple(upper)]
    return out


def node_to_cell_difference(cells, step):
    """Derivative at cell centres from all cells+1 bounding node values."""
    return diags((-np.ones(cells), np.ones(cells)), (0, 1),
                 shape=(cells, cells+1), format='csr') / step

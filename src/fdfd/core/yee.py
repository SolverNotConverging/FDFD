"""Yee-lattice building blocks shared by every kernel.

Field arrays are indexed in physical-axis order and flattened in Fortran
order (first axis fastest), so a sparse operator acting on axis ``k`` is a
Kronecker product with identities on the other axes. "Cells" are the
``N`` samples at cell centres along an axis and "nodes" the ``N + 1`` samples
on cell faces; on a periodic axis the duplicate end node is dropped and node
``k`` lies on the lower face of cell ``k``.
"""

import numpy as np
from scipy import sparse


def diagonal3(name, value):
    """Return a scalar or ``(xx, yy, zz)`` value as a complex length-3 array."""
    if np.isscalar(value):
        return np.full(3, value, dtype=complex)
    array = np.asarray(value, dtype=complex)
    if array.shape == (3,):
        return array
    raise ValueError(f"{name} must be a scalar or a length-3 sequence (xx, yy, zz).")


def diag(values):
    """Sparse diagonal matrix of ``values`` flattened in Fortran order."""
    return sparse.diags(np.asarray(values).ravel(order="F"), format="csr")


def inverse_diag(values, constrained):
    """Sparse inverse diagonal that is exactly zero on ``constrained`` samples."""
    flat = np.asarray(values).ravel(order="F")
    keep = ~np.asarray(constrained, dtype=bool).ravel(order="F")
    inverse = np.zeros(flat.shape, dtype=complex)
    inverse[keep] = 1.0 / flat[keep]
    return sparse.diags(inverse, format="csr")


def _along(axis, operator, in_shape, out_shape):
    """Embed a 1D operator on ``axis`` in an N-D Fortran-ordered lattice."""
    factors = []
    for k in reversed(range(len(in_shape))):
        if k == axis:
            factors.append(operator)
        else:
            factors.append(sparse.eye(out_shape[k], in_shape[k], format="csr"))
    result = factors[0]
    for factor in factors[1:]:
        result = sparse.kron(result, factor, format="csr")
    return result.tocsr()


def difference(in_shape, out_shape, axis, step):
    """Forward difference ``(f[i+1] - f[i]) / step`` between two lattices.

    Entries that would read outside ``in_shape`` are dropped, which gives the
    rectangular node-to-cell (``N+1 -> N``) operator and, as its negative
    adjoint, the cell-to-node operator with natural end rows.
    """
    n_out, n_in = out_shape[axis], in_shape[axis]
    rows = np.arange(n_out)
    ones = np.ones(n_out)
    plus = rows + 1 < n_in
    one_d = (
        sparse.coo_matrix(
            (
                np.r_[ones[plus], -ones[rows < n_in]],
                (np.r_[rows[plus], rows[rows < n_in]], np.r_[rows[plus] + 1, rows[rows < n_in]]),
            ),
            shape=(n_out, n_in),
        ).tocsr()
        / step
    )
    return _along(axis, one_d, in_shape, out_shape)


def periodic_difference(shape, axis, step):
    """Forward difference with wrap-around on a periodic axis."""
    n = shape[axis]
    rows = np.arange(n)
    one_d = (
        sparse.coo_matrix(
            (np.r_[np.ones(n), -np.ones(n)], (np.r_[rows, rows], np.r_[(rows + 1) % n, rows])),
            shape=(n, n),
        ).tocsr()
        / step
    )
    return _along(axis, one_d, shape, shape)


def bloch_difference(shape, axis, step, wavenumber):
    """Forward difference on a Bloch-periodic axis.

    The wrapped entry carries ``exp(-i*k*L)`` (``L`` the period), i.e. fields
    vary as ``exp(-i*k*r)`` times a periodic envelope under ``exp(+i*omega*t)``.
    A single-cell axis reduces to multiplication by ``-i*k``.
    """
    n = shape[axis]
    if n == 1:
        return _along(axis, sparse.csr_matrix([[-1j * wavenumber]]), shape, shape)
    rows = np.arange(n)
    wrap = np.exp(-1j * wavenumber * n * step)
    data = np.r_[-np.ones(n), np.ones(n - 1), [wrap]].astype(complex)
    cols = np.r_[rows, rows[:-1] + 1, [0]]
    one_d = sparse.coo_matrix((data, (np.r_[rows, rows[:-1], [n - 1]], cols)), shape=(n, n)).tocsr() / step
    return _along(axis, one_d, shape, shape)


def periodic_average(shape, axis):
    """Forward average ``(f[i] + f[i+1]) / 2`` with wrap-around."""
    n = shape[axis]
    rows = np.arange(n)
    one_d = sparse.coo_matrix(
        (np.full(2 * n, 0.5), (np.r_[rows, rows], np.r_[rows, (rows + 1) % n])), shape=(n, n)
    ).tocsr()
    return _along(axis, one_d, shape, shape)


def node_average(values, axis, *, periodic=False):
    """Average cell values onto the bounding nodes along one axis."""
    values = np.asarray(values)
    if periodic:
        return 0.5 * (values + np.roll(values, 1, axis=axis))
    shape = list(values.shape)
    shape[axis] += 1
    out = np.zeros(shape, dtype=np.result_type(values.dtype, np.float64))
    counts = np.zeros(shape, dtype=float)
    lower = [slice(None)] * values.ndim
    upper = lower.copy()
    lower[axis], upper[axis] = slice(None, -1), slice(1, None)
    out[tuple(lower)] += values
    out[tuple(upper)] += values
    counts[tuple(lower)] += 1
    counts[tuple(upper)] += 1
    return out / counts


def occupied_nodes(mask, axis, *, periodic=False):
    """Nodes touching an occupied cell along ``axis`` (boundary faces included)."""
    mask = np.asarray(mask, dtype=bool)
    if periodic:
        return mask | np.roll(mask, 1, axis=axis)
    shape = list(mask.shape)
    shape[axis] += 1
    out = np.zeros(shape, dtype=bool)
    lower = [slice(None)] * mask.ndim
    upper = lower.copy()
    lower[axis], upper[axis] = slice(None, -1), slice(1, None)
    out[tuple(lower)] |= mask
    out[tuple(upper)] |= mask
    return out


def interior_nodes(mask, axis, *, periodic=False):
    """Nodes strictly inside occupied cells along ``axis`` (boundary faces excluded)."""
    mask = np.asarray(mask, dtype=bool)
    if periodic:
        return mask & np.roll(mask, 1, axis=axis)
    shape = list(mask.shape)
    shape[axis] += 1
    out = np.zeros(shape, dtype=bool)
    middle = [slice(None)] * mask.ndim
    lower, upper = middle.copy(), middle.copy()
    middle[axis], lower[axis], upper[axis] = slice(1, -1), slice(None, -1), slice(1, None)
    out[tuple(middle)] = mask[tuple(lower)] & mask[tuple(upper)]
    return out


def _on_node(axis, component, field):
    # Along axis a, E_c sits on nodes for a != c and H_c for a == c.
    return (axis != component) if field == "E" else (axis == component)


def staggered_shape(cells, component, axes, field, *, periodic=()):
    """Sample count of field ``"E"``/``"H"`` component ``"x"|"y"|"z"`` along each axis."""
    return tuple(
        n + 1 if _on_node(axis, component, field) and axis not in periodic else n
        for n, axis in zip(cells, axes, strict=True)
    )


def staggered_average(values, component, axes, field, *, periodic=()):
    """Average cell-centred material values onto a field component's samples."""
    values = np.asarray(values, dtype=complex)
    for index, axis in enumerate(axes):
        if _on_node(axis, component, field):
            values = node_average(values, index, periodic=axis in periodic)
    return np.asarray(values, dtype=complex)


def footprint(mask, component, axes, field, *, periodic=(), interior=False):
    """Samples of a field component touching (or, with ``interior``, strictly inside) a cell mask."""
    nodes = interior_nodes if interior else occupied_nodes
    mask = np.asarray(mask, dtype=bool)
    for index, axis in enumerate(axes):
        if _on_node(axis, component, field):
            mask = nodes(mask, index, periodic=axis in periodic)
    return mask


def unflatten(flat, shape):
    """Reshape Fortran-flattened mode columns to ``(*shape, modes)`` as a copy."""
    flat = np.asarray(flat, dtype=complex)
    return flat.reshape((*shape, flat.shape[1]), order="F").copy()


def most_real_phase(*fields):
    """Unit phase that makes ``sum(field**2)`` real, i.e. the fields most nearly real."""
    values = np.concatenate([np.asarray(field).ravel() for field in fields])
    values = values[np.isfinite(values)]
    if values.size == 0 or np.max(np.abs(values)) == 0:
        return 1.0 + 0j
    return np.exp(-0.5j * np.angle(np.sum(values**2)))


def passive_root(neff_squared):
    """Effective index from ``neff**2``, choosing the passive forward branch.

    Under ``exp(+i*omega*t - i*beta*z)`` a forward mode has ``Re(neff) >= 0``
    and a purely evanescent mode decays with ``Im(neff) < 0``.
    """
    root = np.sqrt(neff_squared)
    tolerance = 1e-12 * np.maximum(1.0, np.abs(root))
    flip = (np.real(root) < -tolerance) | ((np.abs(np.real(root)) <= tolerance) & (np.imag(root) > tolerance))
    neff = np.where(flip, -root, root)
    real = np.where(np.abs(neff.real) <= tolerance, 0.0, neff.real)
    imag = np.where(np.abs(neff.imag) <= tolerance, 0.0, neff.imag)
    return real + 1j * imag

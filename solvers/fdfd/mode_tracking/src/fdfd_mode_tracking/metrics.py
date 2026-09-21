"""Native Yee quadrature, physical field normalization and subspace metrics."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


def weights(result, component):
    shape = result.fields[component].shape[:-1]
    out = np.ones(shape)
    for axis, (n, cells, (lo, hi)) in enumerate(zip(shape, result.mesh_data.resolution,
                                                  result.mesh_data.bounds)):
        w = np.full(n, (hi-lo)/cells)
        if n == cells+1:
            w[[0, -1]] *= 0.5
        elif n != cells:
            raise ValueError('Unsupported staggered field shape.')
        reshape = [1]*len(shape)
        reshape[axis] = n
        out *= w.reshape(reshape)
    return out


def measure(result, fields):
    """Area/length averaged E/H norm, with magnetic fields in physical units."""
    eta = result.metadata['eta0']
    area = np.prod([hi-lo for lo, hi in result.mesh_data.bounds])
    blocks = []
    for name in COMPONENTS:
        factor = eta if name.startswith('H') else 1.0
        blocks.append((fields[name]*factor*np.sqrt(weights(result, name)/area)[..., None])
                      .reshape(-1, len(result)))
    vectors = np.concatenate(blocks)
    return vectors, np.sqrt(np.sum(abs(vectors)**2, axis=0))


def cell_fields(result, fields):
    values = {}
    for name, raw in fields.items():
        value = raw
        for axis, cells in enumerate(result.mesh_data.resolution):
            if value.shape[axis] == cells+1:
                value = (np.take(value, range(cells), axis=axis) +
                         np.take(value, range(1, cells+1), axis=axis))*0.5
        values[name] = value
    return values


def complex_power(result, fields):
    f = cell_fields(result, fields)
    integrand = (f['Ex']*f['Hy'].conj() - f['Ey']*f['Hx'].conj())*0.5
    cell_volume = np.prod([(hi-lo)/n for (lo, hi), n in
                           zip(result.mesh_data.bounds, result.mesh_data.resolution)])
    return np.sum(integrand, axis=tuple(range(integrand.ndim-1)))*cell_volume


def normalize_fields(result):
    eta = result.metadata['eta0']
    fields = {n: np.array(result.fields[n], dtype=complex, copy=True) *
              (1j/eta if n.startswith('H') else 1.0) for n in COMPONENTS}
    vectors, norms = measure(result, fields)
    good = np.isfinite(norms) & (norms > np.finfo(float).tiny)
    scales = np.divide(1., norms, out=np.zeros_like(norms), where=good)
    for name in fields:
        fields[name] = np.where(good, fields[name]*scales, 0.)
    vectors = np.where(good, vectors*scales, 0.)
    return fields, vectors, scales, complex_power(result, fields), good


def edge_fractions(result, fields):
    f = cell_fields(result, fields)
    energy = sum(abs(v)**2*(result.metadata['eta0']**2 if n.startswith('H') else 1.)
                 for n, v in f.items())
    edge = np.zeros(energy.shape[:-1], dtype=bool)
    for axis, n in enumerate(edge.shape):
        width = max(1, int(np.ceil(n*0.1)))
        index = [slice(None)]*edge.ndim
        index[axis] = slice(0, width)
        edge[tuple(index)] = True
        index[axis] = slice(n-width, n)
        edge[tuple(index)] = True
    total = energy.sum(axis=tuple(range(edge.ndim)))
    return np.divide(energy[edge].sum(axis=0), total, out=np.ones_like(total), where=total > 0)


def resampled_vectors(reference, other, fields):
    """Compare on the reference interior; clamp only half-cell edge offsets."""
    sampled = {}
    for name in COMPONENTS:
        axes = other.field_coordinates[name]
        target = reference.field_coordinates[name]
        # Node/cell lattices on refined meshes may differ by half a cell at edges.
        clipped = []
        for grid, coordinates in zip(axes, target):
            spacing = np.max(np.diff(grid))
            if coordinates[0] < grid[0]-spacing or coordinates[-1] > grid[-1]+spacing:
                raise ValueError('Verification domain does not contain the comparison region.')
            clipped.append(np.clip(coordinates, grid[0], grid[-1]))
        points = np.stack(np.meshgrid(*clipped, indexing='ij'), axis=-1)
        sampled[name] = RegularGridInterpolator(axes, fields[name], bounds_error=True)(points)
    # The candidate count may differ from the reference.
    from dataclasses import replace
    proxy = replace(reference, neff=other.neff)
    vectors, norms = measure(proxy, sampled)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def orthonormal_basis(vectors, tolerance=1e-10):
    u, s, vh = np.linalg.svd(vectors, full_matrices=False)
    rank = int(np.sum(s > (s[0]*tolerance if len(s) else tolerance)))
    transform = vh[:rank].conj().T / s[:rank]
    if rank == vectors.shape[1]:
        # Polar orthonormalization preserves an already orthonormal gauge.
        return u @ vh, transform @ vh
    return u[:, :rank], transform


def align_subspaces(old, new):
    """Return principal overlaps and the full raw-new -> aligned transformation."""
    qo, _ = orthonormal_basis(old)
    qn, tn = orthonormal_basis(new)
    u, s, vh = np.linalg.svd(qo.conj().T @ qn, full_matrices=False)
    rotation = vh.conj().T @ u.conj().T
    return {'singular_values': s, 'old_rank': qo.shape[1], 'new_rank': qn.shape[1],
            'transform': tn @ rotation, 'aligned': qn @ rotation}

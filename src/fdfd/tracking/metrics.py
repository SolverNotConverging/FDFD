"""Native Yee quadrature, physical field normalization and subspace metrics."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def weights(result, component):
    shape = result.fields[component].shape[:-1]
    out = np.ones(shape)
    for axis, (n, cells, (lo, hi)) in enumerate(
        zip(shape, result.grid.resolution, result.grid.bounds, strict=True)
    ):
        w = np.full(n, (hi - lo) / cells)
        if n == cells + 1:
            w[[0, -1]] *= 0.5
        elif n != cells:
            raise ValueError("Unsupported staggered field shape.")
        reshape = [1] * len(shape)
        reshape[axis] = n
        out *= w.reshape(reshape)
    return out


def measure(result, fields):
    """Area/length averaged E/H norm, with magnetic fields in physical units."""
    eta = result.metadata["eta0"]
    area = np.prod([hi - lo for lo, hi in result.grid.bounds])
    blocks = []
    for name in COMPONENTS:
        factor = eta if name.startswith("H") else 1.0
        blocks.append(
            (fields[name] * factor * np.sqrt(weights(result, name) / area)[..., None]).reshape(
                -1, len(result)
            )
        )
    vectors = np.concatenate(blocks)
    return vectors, np.sqrt(np.sum(abs(vectors) ** 2, axis=0))


def cell_fields(result, fields):
    values = {}
    for name, raw in fields.items():
        value = raw
        for axis, cells in enumerate(result.grid.resolution):
            if value.shape[axis] == cells + 1:
                value = (
                    np.take(value, range(cells), axis=axis) + np.take(value, range(1, cells + 1), axis=axis)
                ) * 0.5
        values[name] = value
    return values


def complex_power(result, fields):
    f = cell_fields(result, fields)
    integrand = (f["Ex"] * f["Hy"].conj() - f["Ey"] * f["Hx"].conj()) * 0.5
    cell_volume = np.prod(
        [(hi - lo) / n for (lo, hi), n in zip(result.grid.bounds, result.grid.resolution, strict=True)]
    )
    return np.sum(integrand, axis=tuple(range(integrand.ndim - 1))) * cell_volume


def normalize_fields(result):
    eta = result.metadata["eta0"]
    fields = {
        n: np.array(result.fields[n], dtype=complex, copy=True) * (1j / eta if n.startswith("H") else 1.0)
        for n in COMPONENTS
    }
    vectors, norms = measure(result, fields)
    good = np.isfinite(norms) & (norms > np.finfo(float).tiny)
    scales = np.divide(1.0, norms, out=np.zeros_like(norms), where=good)
    for name in fields:
        fields[name] = np.where(good, fields[name] * scales, 0.0)
    vectors = np.where(good, vectors * scales, 0.0)
    return fields, vectors, scales, complex_power(result, fields), good


def edge_fractions(result, fields):
    f = cell_fields(result, fields)
    energy = sum(
        abs(v) ** 2 * (result.metadata["eta0"] ** 2 if n.startswith("H") else 1.0) for n, v in f.items()
    )
    edge = np.zeros(energy.shape[:-1], dtype=bool)
    for axis, n in enumerate(edge.shape):
        width = max(1, int(np.ceil(n * 0.1)))
        index = [slice(None)] * edge.ndim
        index[axis] = slice(0, width)
        edge[tuple(index)] = True
        index[axis] = slice(n - width, n)
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
        for grid, coordinates in zip(axes, target, strict=True):
            spacing = np.max(np.diff(grid))
            if coordinates[0] < grid[0] - spacing or coordinates[-1] > grid[-1] + spacing:
                raise ValueError("Verification domain does not contain the comparison region.")
            clipped.append(np.clip(coordinates, grid[0], grid[-1]))
        points = np.stack(np.meshgrid(*clipped, indexing="ij"), axis=-1)
        sampled[name] = RegularGridInterpolator(axes, fields[name], bounds_error=True)(points)
    # The candidate count may differ from the reference.
    from dataclasses import replace

    proxy = replace(reference, neff=other.neff)
    vectors, norms = measure(proxy, sampled)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def orthonormal_basis(vectors, tolerance=1e-10):
    u, s, vh = np.linalg.svd(vectors, full_matrices=False)
    rank = int(np.sum(s > (s[0] * tolerance if len(s) else tolerance)))
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
    return {
        "singular_values": s,
        "old_rank": qo.shape[1],
        "new_rank": qn.shape[1],
        "transform": tn @ rotation,
        "aligned": qn @ rotation,
    }


def field_character(result, fields, tolerance=1e-8):
    """Measured longitudinal character and full-vector mirror parity.

    E is a polar vector and H an axial vector. A mixed eigensolver basis can
    have indefinite parity even when its span is symmetry invariant.
    """
    vectors, norms = measure(result, fields)
    energy = {}
    for name in COMPONENTS:
        factor = result.metadata["eta0"] ** 2 if name.startswith("H") else 1.0
        axes = tuple(range(fields[name].ndim - 1))
        energy[name] = factor * np.sum(abs(fields[name]) ** 2 * weights(result, name)[..., None], axis=axes)
    electric = sum(energy[n] for n in COMPONENTS[:3])
    magnetic = sum(energy[n] for n in COMPONENTS[3:])
    ez = np.divide(energy["Ez"], electric, out=np.zeros_like(electric), where=electric > 0)
    hz = np.divide(energy["Hz"], magnetic, out=np.zeros_like(magnetic), where=magnetic > 0)
    parities = []
    for axis in range(len(result.grid.resolution)):
        reflected = {}
        for name in COMPONENTS:
            normal = name[1] == "xyz"[axis]
            sign = (-1 if normal else 1) * (-1 if name.startswith("H") else 1)
            reflected[name] = sign * np.flip(fields[name], axis=axis)
        mirror, _ = measure(result, reflected)
        parity = np.divide(
            np.sum(vectors.conj() * mirror, axis=0).real, norms**2, out=np.zeros_like(norms), where=norms > 0
        )
        parities.append(parity)
    records = []
    for i in range(len(result)):
        family = (
            "TEM"
            if ez[i] <= tolerance and hz[i] <= tolerance
            else "TE"
            if ez[i] <= tolerance
            else "TM"
            if hz[i] <= tolerance
            else "hybrid"
        )
        if not np.isfinite(norms[i]) or norms[i] <= 0:
            family = "unresolved"
        records.append(
            {
                "family": family,
                "electric_longitudinal_fraction": float(ez[i]),
                "magnetic_longitudinal_fraction": float(hz[i]),
                "mirror_parity": tuple(
                    1 if abs(p[i] - 1) <= tolerance else -1 if abs(p[i] + 1) <= tolerance else 0
                    for p in parities
                ),
            }
        )
    return tuple(records)


def compatible_character(left, right):
    """Forbid contradictory definite sectors; hybrid/indefinite labels abstain."""
    pure = {"TE", "TM", "TEM"}
    if left["family"] in pure and right["family"] in pure and left["family"] != right["family"]:
        return False
    return all(
        a == b or a == 0 or b == 0 for a, b in zip(left["mirror_parity"], right["mirror_parity"], strict=True)
    )


def canonical_subspace(result, fields, indices, tolerance=1e-8):
    """Resolve longitudinal nullspaces, then invariant mirror sectors of a seed.

    All operations stay inside an already certified numerical degeneracy.
    Mirrors are used only when they preserve each previously resolved sector.
    No geometric symmetry is inferred from eigenvalue proximity.
    """
    from dataclasses import replace

    vectors, _ = measure(result, fields)
    q, transform = orthonormal_basis(vectors[:, indices])
    rank = len(indices)
    if q.shape[1] != rank:
        return transform, ()
    groups = [np.arange(rank)]
    offsets = np.cumsum([0, *(int(np.prod(fields[n].shape[:-1])) for n in COMPONENTS)])
    # The nullspace of Ez is TE; the nullspace of Hz is TM. Their common
    # nullspace is TEM. Resolve null/non-null only, not arbitrary energy levels.
    for name in ("Ez", "Hz"):
        block = COMPONENTS.index(name)
        updated = []
        for group in groups:
            part = q[offsets[block] : offsets[block + 1], group]
            values, rotation = np.linalg.eigh(part.conj().T @ part)
            q[:, group] = q[:, group] @ rotation
            transform[:, group] = transform[:, group] @ rotation
            for mask in (values <= tolerance, values > tolerance):
                if np.any(mask):
                    updated.append(group[mask])
        groups = updated
    for axis in range(len(result.grid.resolution)):
        reflected = np.empty_like(q)
        for block, name in enumerate(COMPONENTS):
            shape = fields[name].shape[:-1]
            section = slice(offsets[block], offsets[block + 1])
            normal = name[1] == "xyz"[axis]
            sign = (-1 if normal else 1) * (-1 if name.startswith("H") else 1)
            reflected[section] = sign * np.flip(q[section].reshape(*shape, rank), axis=axis).reshape(-1, rank)
        updated = []
        for group in groups:
            basis = q[:, group]
            compressed = basis.conj().T @ reflected[:, group]
            error = np.linalg.norm(reflected[:, group] - basis @ compressed)
            if error > tolerance * np.sqrt(len(group)):
                updated.append(group)
                continue
            values, rotation = np.linalg.eigh((compressed + compressed.conj().T) / 2)
            q[:, group] = basis @ rotation
            transform[:, group] = transform[:, group] @ rotation
            for mask in (values < 0, values >= 0):
                if np.any(mask):
                    updated.append(group[mask])
        groups = updated
    aligned = {name: fields[name][..., indices] @ transform for name in COMPONENTS}
    proxy = replace(result, neff=result.neff[list(indices)])
    return transform, field_character(proxy, aligned, tolerance)

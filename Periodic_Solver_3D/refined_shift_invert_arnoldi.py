import numpy as np
from scipy.sparse.linalg import splu


def _default_arnoldi_ncv(n, num_modes, ncv):
    if n <= num_modes + 1:
        raise ValueError(f"Not enough DOFs to solve {num_modes} modes.")
    if ncv is None:
        ncv = max(20, 4 * int(num_modes) + 8)
    ncv = int(ncv)
    ncv = max(num_modes + 2, ncv)
    return min(n - 1, ncv)


def _normalised_vector(v):
    nrm = np.linalg.norm(v)
    if nrm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / nrm


def _arnoldi_factorization(apply_op, n, ncv, v0):
    V = np.zeros((n, ncv + 1), dtype=complex)
    H = np.zeros((ncv + 1, ncv), dtype=complex)
    V[:, 0] = _normalised_vector(v0.astype(complex, copy=False))

    basis_count = 1
    for j in range(ncv):
        w = apply_op(V[:, j])
        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)
            w -= H[i, j] * V[:, i]

        # A second pass is cheap for these subspace sizes and helps with
        # nonsymmetric, ill-conditioned shift-invert operators.
        for i in range(j + 1):
            correction = np.vdot(V[:, i], w)
            H[i, j] += correction
            w -= correction * V[:, i]

        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] <= 1e-14:
            basis_count = j + 1
            break

        if j + 1 < ncv:
            V[:, j + 1] = w / H[j + 1, j]
            basis_count = j + 2

    return V[:, :basis_count], H[:basis_count, :basis_count]


def _relative_residual_norm(Ax, Bx, eigenvalue):
    residual = Ax - eigenvalue * Bx
    scale = np.linalg.norm(Ax) + abs(eigenvalue) * np.linalg.norm(Bx)
    if scale == 0:
        scale = 1.0
    return np.linalg.norm(residual) / scale


def _refined_candidates(A, B, V, H, sigma, num_modes):
    if H.shape[0] < num_modes:
        raise RuntimeError("Arnoldi subspace is smaller than the requested number of modes.")

    mu_values, _ = np.linalg.eig(H)
    finite = np.isfinite(mu_values) & (np.abs(mu_values) > 1e-14)
    mu_values = mu_values[finite]
    if mu_values.size < num_modes:
        raise RuntimeError("Not enough finite Ritz values were generated.")

    eigenvalues = sigma + 1.0 / mu_values
    order = np.argsort(np.abs(eigenvalues - sigma))

    AV = A @ V
    BV = B @ V
    candidates = []
    seen = []
    for idx in order:
        eigenvalue = eigenvalues[idx]
        if any(abs(eigenvalue - old) <= 1e-8 * max(1.0, abs(eigenvalue)) for old in seen):
            continue

        residual_basis = AV - eigenvalue * BV
        try:
            _, _, vh = np.linalg.svd(residual_basis, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        y = vh.conj().T[:, -1]
        vector = V @ y
        vector_norm = np.linalg.norm(vector)
        if vector_norm == 0:
            continue
        vector /= vector_norm

        Ax = A @ vector
        Bx = B @ vector
        residual = _relative_residual_norm(Ax, Bx, eigenvalue)
        candidates.append((residual, eigenvalue, vector))
        seen.append(eigenvalue)

    if len(candidates) < num_modes:
        raise RuntimeError("Refined extraction produced too few candidate modes.")

    selected = sorted(candidates, key=lambda item: abs(item[1] - sigma))[:num_modes]
    residuals = np.array([item[0] for item in selected], dtype=float)
    eigenvalues = np.array([item[1] for item in selected], dtype=complex)
    eigenvectors = np.column_stack([item[2] for item in selected])
    return eigenvalues, eigenvectors, residuals


def _restart_vector(eigenvectors, rng, n):
    weights = np.ones(eigenvectors.shape[1], dtype=complex)
    start = eigenvectors @ weights
    start += 1e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return _normalised_vector(start)


def refined_shift_invert_arnoldi(
    A,
    B,
    sigma,
    num_modes,
    tol,
    ncv=None,
    max_restarts=12,
    random_seed=0,
):
    """Solve a generalized eigenproblem with refined shift-invert Arnoldi.

    The target problem is ``A x = lambda B x``. Arnoldi is applied to
    ``(A - sigma B)^-1 B`` and the reported eigenvalues are mapped back to
    the original pencil.
    """
    n = A.shape[0]
    ncv = _default_arnoldi_ncv(n, num_modes, ncv)
    tol = 0.0 if tol is None else float(tol)
    convergence_tol = max(tol, 1e-12)

    shifted = (A - sigma * B).tocsc()
    lu = splu(shifted)

    def apply_shift_invert(v):
        return lu.solve(B @ v)

    rng = np.random.default_rng(random_seed)
    v0 = np.ones(n, dtype=complex)
    v0 += 1e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    best = None
    for restart in range(int(max_restarts) + 1):
        V, H = _arnoldi_factorization(apply_shift_invert, n, ncv, v0)
        eigenvalues, eigenvectors, residuals = _refined_candidates(A, B, V, H, sigma, num_modes)

        if best is None or np.max(residuals) < np.max(best[2]):
            best = (eigenvalues, eigenvectors, residuals, restart)

        if np.all(residuals <= convergence_tol):
            return eigenvalues, eigenvectors, residuals, restart

        v0 = _restart_vector(eigenvectors, rng, n)

    eigenvalues, eigenvectors, residuals, restart = best
    return eigenvalues, eigenvectors, residuals, restart

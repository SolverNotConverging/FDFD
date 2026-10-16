"""Refined projected extraction restored from commit 3402f85.

Repeated Ritz roots retain independent right singular vectors. Only this small
projected problem runs on CPU; the Arnoldi basis stays on the CUDA device.
"""

import numpy as np
from scipy import linalg


def refined_coefficients(Hbar, sigma, num_modes):
    """Return finite roots, refined basis coordinates and projected residuals."""
    arnoldi_size = Hbar.shape[1]
    if arnoldi_size < num_modes or Hbar.shape != (arnoldi_size + 1, arnoldi_size):
        raise ValueError("Invalid Arnoldi subspace dimensions.")
    projected = Hbar[:arnoldi_size, :]
    mu_values = linalg.eigvals(projected, check_finite=False)
    finite = np.isfinite(mu_values) & (np.abs(mu_values) > 1e-14)
    mu_values = mu_values[finite]
    if mu_values.size < num_modes:
        raise RuntimeError("Not enough finite Ritz values were generated.")

    mapped_values = complex(sigma) + 1.0 / mu_values
    order = np.argsort(np.abs(mapped_values - sigma))
    eigenvalues: list[complex] = []
    refined_coefficients: list[np.ndarray] = []
    projected_residuals: list[float] = []

    diagonal = np.arange(arnoldi_size)
    clusters: list[list[int]] = []
    repeated_root_rtol = 64.0 * np.finfo(np.float64).eps
    for raw_index in order:
        index = int(raw_index)
        eigenvalue = mapped_values[index]
        for cluster in clusters:
            reference = mapped_values[cluster[0]]
            scale = max(1.0, abs(eigenvalue), abs(reference))
            if abs(eigenvalue - reference) <= repeated_root_rtol * scale:
                cluster.append(index)
                break
        else:
            clusters.append([index])

    def residual_matrix(mu: complex) -> np.ndarray:
        small_residual = np.array(Hbar, dtype=np.complex128, order="F", copy=True)
        small_residual[diagonal, diagonal] -= mu
        return small_residual

    for cluster in clusters:
        if len(eigenvalues) == num_modes:
            break

        if len(cluster) == 1:
            index = cluster[0]
            try:
                _u, singular_values, vh = linalg.svd(
                    residual_matrix(mu_values[index]),
                    full_matrices=False,
                    check_finite=False,
                    lapack_driver="gesvd",
                )
            except linalg.LinAlgError:
                continue
            coefficient = np.asarray(vh[-1, :].conj(), dtype=np.complex128)
            projected_residual = float(singular_values[-1])
            if not (np.all(np.isfinite(coefficient)) and np.isfinite(projected_residual)):
                continue
            eigenvalues.append(complex(mapped_values[index]))
            refined_coefficients.append(coefficient)
            projected_residuals.append(projected_residual)
            continue

        # A repeated root makes the smallest singular vector non-unique. Build
        # the whole small right-singular subspace at the cluster centre, then
        # minimize each root's residual successively in its unused complement.
        representative_mu = complex(np.mean(mu_values[cluster]))
        try:
            _u, _singular_values, representative_vh = linalg.svd(
                residual_matrix(representative_mu),
                full_matrices=False,
                check_finite=False,
                lapack_driver="gesvd",
            )
        except linalg.LinAlgError:
            continue
        cluster_size = len(cluster)
        available = np.asarray(
            representative_vh[-cluster_size:, :].conj().T[:, ::-1],
            dtype=np.complex128,
            order="F",
        )

        for index in cluster:
            if len(eigenvalues) == num_modes or available.shape[1] == 0:
                break
            try:
                _u, restricted_singular_values, restricted_vh = linalg.svd(
                    residual_matrix(mu_values[index]) @ available,
                    full_matrices=False,
                    check_finite=False,
                    lapack_driver="gesvd",
                )
            except linalg.LinAlgError:
                continue
            restricted_coefficient = np.asarray(restricted_vh[-1, :].conj(), dtype=np.complex128)
            coefficient = np.asarray(available @ restricted_coefficient, dtype=np.complex128)
            coefficient /= np.linalg.norm(coefficient)
            projected_residual = float(restricted_singular_values[-1])

            if available.shape[1] > 1:
                complement = linalg.null_space(restricted_coefficient.conj()[np.newaxis, :])
                available = np.asfortranarray(available @ complement)
            else:
                available = np.empty((arnoldi_size, 0), dtype=np.complex128, order="F")

            if not (np.all(np.isfinite(coefficient)) and np.isfinite(projected_residual)):
                continue
            eigenvalues.append(complex(mapped_values[index]))
            refined_coefficients.append(coefficient)
            projected_residuals.append(projected_residual)

    if len(eigenvalues) < num_modes:
        raise RuntimeError("Refined extraction produced too few candidate modes.")

    coefficient_matrix = np.asfortranarray(np.column_stack(refined_coefficients), dtype=np.complex128)
    return np.asarray(eigenvalues), coefficient_matrix, np.asarray(projected_residuals)

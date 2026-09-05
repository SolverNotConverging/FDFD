"""Portable reference kernels for refined Arnoldi."""

from __future__ import annotations

import numpy as np


def arnoldi_step(
    basis: np.ndarray,
    hessenberg: np.ndarray,
    work: np.ndarray,
    column: int,
    breakdown_tolerance: float,
) -> float:
    """Apply two-pass modified Gram--Schmidt to one Arnoldi column."""
    for index in range(column + 1):
        coefficient = np.vdot(basis[:, index], work)
        hessenberg[index, column] = coefficient
        work -= coefficient * basis[:, index]

    for index in range(column + 1):
        correction = np.vdot(basis[:, index], work)
        hessenberg[index, column] += correction
        work -= correction * basis[:, index]

    beta = float(np.linalg.norm(work))
    hessenberg[column + 1, column] = beta
    if beta > breakdown_tolerance:
        basis[:, column + 1] = work / beta
    return beta


def relative_residuals(
    ax: np.ndarray,
    bx: np.ndarray,
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Return scale-invariant residuals for a batch of pencil eigenpairs."""
    residuals = np.empty(eigenvalues.size, dtype=np.float64)
    for index, eigenvalue in enumerate(eigenvalues):
        ax_column = ax[:, index]
        bx_column = bx[:, index]
        scale = np.linalg.norm(ax_column) + abs(eigenvalue) * np.linalg.norm(
            bx_column
        )
        if scale == 0.0:
            scale = 1.0
        residuals[index] = np.linalg.norm(
            ax_column - eigenvalue * bx_column
        ) / scale
    return residuals

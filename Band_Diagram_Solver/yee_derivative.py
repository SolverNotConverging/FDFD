import numpy as np
from scipy.sparse import spdiags, eye, csr_matrix, diags


def yeeder2d(NS, RES, BC, kinc=None):
    """
    YEEDER2D Derivative Matrices on a 2D Yee Grid

    Parameters
    ----------
    NS : list of int
        [Nx, Ny] Grid Size
    RES : list of float
        [dx, dy] Grid Resolution
    BC : list of int
        [xbc, ybc] Boundary Conditions
        0: Dirichlet boundary conditions
        1: Periodic boundary conditions
    kinc : list of float, optional
        [kx, ky] Incident Wave Vector (only needed for PBCs)

    Returns
    -------
    DEX : scipy.sparse.csc_matrix
        Derivative Matrix wrt x for Electric Fields
    DEY : scipy.sparse.csc_matrix
        Derivative Matrix wrt y for Electric Fields
    DHX : scipy.sparse.csc_matrix
        Derivative Matrix wrt x for Magnetic Fields
    DHY : scipy.sparse.csc_matrix
        Derivative Matrix wrt y for Magnetic Fields
    """

    # Extract grid parameters
    Nx, Ny = NS
    dx, dy = RES

    # Default kinc if not provided
    if kinc is None:
        kinc = np.array([0, 0])

    # Determine matrix size
    M = Nx * Ny

    # Build DEX
    if Nx == 1:
        DEX = -1j * kinc[0] * eye(M, format='csr')
    else:
        d0 = -np.ones(M)
        d1 = np.ones(M)
        d1[Nx - 1::Nx] = 0
        DEX = diags([d0, d1], [0, 1], shape=(M, M), format='csr') / dx
        if BC[0] == 1:
            d1 = np.zeros(M, dtype='complex')
            d1[::Nx] = np.exp(-1j * kinc[0] * Nx * dx) / dx
            DEX += diags(d1, 1 - Nx, shape=(M, M), format='csr')

    # Build DEY
    if Ny == 1:
        DEY = -1j * kinc[1] * eye(M, format='csr')
    else:
        d0 = -np.ones(M)
        d1 = np.ones(M)
        DEY = diags([d0, d1], [0, Nx], shape=(M, M), format='csr') / dy
        if BC[1] == 1:
            d1 = np.exp(-1j * kinc[1] * Ny * dy) / dy * np.ones(M, dtype='complex')
            DEY += diags(d1, Nx - M, shape=(M, M), format='csr')

    # Build DHX and DHY
    DHX = -DEX.conj().T
    DHY = -DEY.conj().T

    return DEX, DEY, DHX, DHY


def yeeder3d(NS, RES, BC, kinc=[0, 0, 0]):
    # Extract grid parameters
    Nx, Ny, Nz = NS
    dx, dy, dz = RES

    # Determine matrix size
    M = Nx * Ny * Nz

    # Zero matrix
    Z = csr_matrix((M, M), dtype=np.complex128)

    # Build DEX
    if Nx == 1:
        DEX = -1j * kinc[0] * eye(M, M, dtype=np.complex128)
    else:
        d0 = -np.ones(M)
        d1 = np.ones(M)
        d1[Nx::Nx] = 0

        DEX = spdiags([d0, d1] / dx, [0, 1], M, M)

        if BC[0] == 1:
            d1 = np.zeros(M)
            d1[0::Nx] = np.exp(-1j * kinc[0] * Nx * dx) / dx
            DEX += spdiags(d1, 1 - Nx, M, M)

    # Build DEY
    if Ny == 1:
        DEY = -1j * kinc[1] * eye(M, M, dtype=np.complex128)
    else:
        d0 = -np.ones(M)
        d1 = np.concatenate([np.ones((Ny - 1) * Nx), np.zeros(Nx)])
        d1 = np.tile(d1, Nz)
        d1 = np.concatenate([np.zeros(Nx), d1, np.ones((Ny - 1) * Nx)])

        DEY = spdiags([d0, d1] / dy, [0, Nx], M, M)

        if BC[1] == 1:
            ph = np.exp(-1j * kinc[1] * Ny * dy) / dy
            d1 = np.concatenate([np.ones(Nx), np.zeros((Ny - 1) * Nx)])
            d1 = np.tile(d1, Nz)
            DEY += spdiags(ph * d1, -Nx * (Ny - 1), M, M)

    # Build DEZ
    if Nz == 1:
        DEZ = -1j * kinc[2] * eye(M, M, dtype=np.complex128)
    else:
        d0 = np.ones(M)

        DEZ = spdiags([-d0, d0] / dz, [0, Nx * Ny], M, M)

        if BC[3] == 1:
            d0 = (np.exp(-1j * kinc[2] * Nz * dz) / dz) * np.ones(M)
            DEZ += spdiags(d0, -Nx * Ny * (Nz - 1), M, M)

    # Build DHX, DHY and DHZ
    DHX = -DEX.conj().transpose()
    DHY = -DEY.conj().transpose()
    DHZ = -DEZ.conj().transpose()

    return DEX, DEY, DEZ, DHX, DHY, DHZ

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

from yee_derivative import yeeder2d

# CRYSTAL PARAMETERS
a = 1.0

# FDFD PARAMETERS
Nx = 40
Ny = Nx
NBETA = 100
NBANDS = 5
wnmax = 0.6

# CALCULATE OPTIMIZED GRID
dx = a / Nx
dy = a / Ny

# 2X GRID
Nx2 = 2 * Nx
dx2 = dx / 2
Ny2 = 2 * Ny
dy2 = dy / 2

# CALCULATE 2X MESHGRID
xa2 = np.arange(1, Nx2 + 1) * dx2
xa2 = xa2 - np.mean(xa2)
ya2 = np.arange(1, Ny2 + 1) * dy2
ya2 = ya2 - np.mean(ya2)
X2, Y2 = np.meshgrid(xa2, ya2)

# BUILD UNIT CELL

r = 0.4 * a
erhole = 1.0
erfill = 10.2
ER2 = (X2 ** 2 + Y2 ** 2) <= r ** 2
ER2 = erfill + (erhole - erfill) * ER2
UR2 = np.ones((Nx2, Ny2))

# EXTRACT YEE GRID MATERIAL ARRAYS
ERxx = ER2[1::2, ::2]
ERyy = ER2[::2, 1::2]
ERzz = ER2[::2, 1::2]
URxx = UR2[1::2, ::2]
URyy = UR2[::2, 1::2]
URzz = UR2[::2, 1::2]
# Constants
a = 1  # Define 'a' as required for your specific problem
NBETA = 100  # Define 'NBETA' as required

# RECIPROCAL LATTICE VECTORS
T1 = (2 * np.pi / a) * np.array([[1], [0]])
T2 = (2 * np.pi / a) * np.array([[0], [1]])

# KEY POINTS OF SYMMETRY
G = np.array([[0], [0]])
X = 0.5 * T1
M = 0.5 * T1 + 0.5 * T2

# CHOOSE PATH AROUND IBZ
KP = np.hstack((G, X, M, G))
KL = ['Γ', 'X', 'M', 'Γ']

# DETERMINE LENGTH OF IBZ PERIMETER
NKP = KP.shape[1]
LIBZ = 0

for m in range(NKP - 1):
    LIBZ += np.linalg.norm(KP[:, m + 1] - KP[:, m])

# GENERATE LIST OF POINTS AROUND IBZ
dibz = LIBZ / NBETA
BETA = KP[:, [0]].copy()
KT = [1]
NBETA = 1

for m in range(NKP - 1):
    dK = KP[:, m + 1] - KP[:, m]
    N = int(np.ceil(np.linalg.norm(dK) / dibz))
    points = KP[:, m].reshape(-1, 1) + np.outer(dK, np.arange(1, N + 1)) / N
    BETA = np.hstack((BETA, points))
    NBETA += N
    KT.append(NBETA)

# PERFORM FDFD ANALYSIS
ERxx_diag = diags(ERxx.flatten(order='F'))
ERyy_diag = diags(ERyy.flatten(order='F'))
ERzz_diag = diags(ERzz.flatten(order='F'))
URxx_diag = diags(URxx.flatten(order='F'))
URyy_diag = diags(URyy.flatten(order='F'))
URzz_diag = diags(URzz.flatten(order='F'))

# INITIALIZE BAND DATA
WNTE = np.zeros((NBANDS, NBETA), dtype='complex')
WNTM = np.zeros((NBANDS, NBETA), dtype='complex')

# MAIN LOOP -- ITERATE OVER IBZ
for nbeta in range(NBETA):
    # Get Next Bloch Wave Vector
    beta = BETA[:, nbeta]

    # Build Derivative Matrices
    NS = [Nx, Ny]
    RES = [dx, dy]
    BC = [1, 1]
    DEX, DEY, DHX, DHY = yeeder2d(NS, RES, BC, beta)

    # TM Mode Analysis
    A = -DHX @ URyy_diag.power(-1) @ DEX - DHY @ URxx_diag.power(-1) @ DEY
    B = ERzz_diag
    D = eigs(A, M=B, k=NBANDS, sigma=0)[0]
    D = np.sort(D)
    WNTM[:, nbeta] = D[:NBANDS]

    # TE Mode Analysis
    A = -DEX @ ERyy_diag.power(-1) @ DHX - DEY @ ERxx_diag.power(-1) @ DHY
    B = URzz_diag
    D = eigs(A, M=B, k=NBANDS, sigma=0)[0]
    D = np.sort(D)
    WNTE[:, nbeta] = D[:NBANDS]

# NORMALIZE THE FREQUENCIES
WNTE = a / (2 * np.pi) * np.real(np.sqrt(WNTE))
WNTM = a / (2 * np.pi) * np.real(np.sqrt(WNTM))

# PLOT
fig = plt.figure(constrained_layout=True)
gs = gridspec.GridSpec(6, 8, figure=fig)
ax1 = fig.add_subplot(gs[0:3, 0:3])
ax2 = fig.add_subplot(gs[3:6, 0:3])
ax3 = fig.add_subplot(gs[:, 3:])

im = ax1.imshow(ER2.T, extent=(xa2.min(), xa2.max(), ya2.min(), ya2.max()), origin='lower', cmap='viridis')
ax1.set_title('Unit Cell')
cbar = fig.colorbar(im, ax=ax1)  # Add a colorbar to ax1
cbar.set_label('$\\epsilon_r$')

image = mpimg.imread('2D_Band_Diagram_Illustration.png')  # Replace with your image file path
ax2.imshow(image)
ax2.axis('off')

ax3.plot(range(1, NBETA + 1), WNTM.T, '.b', label='TM')
ax3.plot(range(1, NBETA + 1), WNTE.T, '.r', label='TE')
handles, labels = ax3.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax3.legend(unique_labels.values(), unique_labels.keys())
ax3.set_xlim([1, NBETA])
ax3.set_ylim([0, wnmax])
ax3.set_xticks(KT)
ax3.set_xticklabels(KL)
ax3.set_xlabel('Bloch Wave Vector $\\vec{\\beta}$')
ax3.set_ylabel('Frequency $\\omega_{n} = a/\\lambda_0$')
ax3.set_title('Photonic Band Diagram')
plt.show()

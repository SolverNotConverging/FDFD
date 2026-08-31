"""SI constants and conventions used by the periodic FEM solvers.

Fields use ``exp(+1j*omega*t)``.  A periodic envelope is multiplied by
``exp(-gamma*z) = exp(-1j*k0*neff*z)``; passive forward modes consequently
have a non-positive imaginary effective index.
"""

from scipy.constants import c as C_0
from scipy.constants import epsilon_0 as EPSILON_0
from scipy.constants import mu_0 as MU_0

ETA_0 = (MU_0 / EPSILON_0) ** 0.5

__all__ = ["C_0", "EPSILON_0", "ETA_0", "MU_0"]

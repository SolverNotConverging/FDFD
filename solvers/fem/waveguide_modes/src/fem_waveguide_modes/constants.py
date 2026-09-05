"""SI constants and the public time-harmonic convention.

The package uses ``exp(+1j*omega*t - 1j*beta*z)``.  Passive bulk materials
therefore have non-positive imaginary relative permittivity/permeability.
"""

from scipy.constants import c as C_0
from scipy.constants import epsilon_0 as EPSILON_0
from scipy.constants import mu_0 as MU_0

ETA_0 = (MU_0 / EPSILON_0) ** 0.5

__all__ = ["C_0", "EPSILON_0", "ETA_0", "MU_0"]

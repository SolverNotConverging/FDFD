"""Physical constants and the phasor conventions used by every solver.

Fields are complex phasors with the time dependence ``exp(+i*omega*t)``. A
mode travelling in +z varies as ``exp(-i*beta*z)``, so passive loss appears as
a negative imaginary part of ``neff`` and of the relative permittivity.
"""

from math import sqrt

from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPSILON_0
from scipy.constants import mu_0 as MU_0

ETA_0 = sqrt(MU_0 / EPSILON_0)
"""Free-space wave impedance in ohms."""

TIME_CONVENTION = "exp(+i*omega*t)"
SPATIAL_CONVENTION = "exp(-i*beta*z)"

__all__ = ["C0", "EPSILON_0", "ETA_0", "MU_0", "SPATIAL_CONVENTION", "TIME_CONVENTION"]

"""Grid-size guidance."""

import numpy as np

from .constants import C0
from .errors import ConfigurationError
from .materials import Material, _positive, max_index


def suggest_resolution(*, frequency, ranges, materials=(), points_per_wavelength=20):
    """Cells per axis that resolve the shortest wavelength in the problem.

    Parameters
    ----------
    frequency : float
        Highest frequency of interest in hertz.
    ranges : sequence of (float, float)
        Domain extent along each axis in metres, as passed to the solver.
    materials : sequence of Material
        Media present; the largest ``|sqrt(epsilon*mu)|`` sets the wavelength.
        Vacuum is always included.
    points_per_wavelength : int
        Cells per shortest wavelength. 20 is a reasonable start for
        effective indices; resolve thin layers and conductor edges separately.

    Returns
    -------
    tuple of int
        Suitable for ``solver.mesh(resolution=...)``.
    """
    frequency = _positive(frequency, "frequency")
    points = _positive(points_per_wavelength, "points_per_wavelength")
    if any(not isinstance(material, Material) for material in materials):
        raise ConfigurationError("materials must be Material objects.")
    index = max(1.0, max_index(materials))
    step = C0 / (frequency * index) / points
    spans = [float(hi) - float(lo) for lo, hi in ranges]
    if not spans or any(not np.isfinite(span) or span <= 0 for span in spans):
        raise ConfigurationError("ranges must be increasing (min, max) pairs in metres.")
    return tuple(max(2, int(np.ceil(span / step - 1e-9))) for span in spans)

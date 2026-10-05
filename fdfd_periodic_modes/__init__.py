"""Supported material-first FDFD user API."""

from pathlib import Path as _Path
__path__.append(str(_Path(__file__).parent / "src"))

from fdfd_common import Material, GoodConductor, SurfaceImpedance, materials, shapes

from .api import PeriodicModeSolver2D, PeriodicModeSolver3D, PeriodicModeSet, load_result
from fdfd_common.dispersion import plot_dispersion
__version__ = "1.1.0"
__all__ = ['PeriodicModeSolver2D', 'PeriodicModeSolver3D', 'PeriodicModeSet', 'load_result', 'plot_dispersion']

__all__ += ["Material", "GoodConductor", "SurfaceImpedance", "materials", "shapes"]

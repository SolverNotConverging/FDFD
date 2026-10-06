"""FDFD waveguide modes: supported material-first user API."""

from pathlib import Path as _Path
__path__.append(str(_Path(__file__).parent / "src"))

from fdfd_common import Material, GoodConductor, SurfaceImpedance, materials, shapes

from .api import ModeSolver1D, ModeSolver2D, ModeSet, load_result
from fdfd_common.dispersion import plot_dispersion
__version__ = "1.1.2"
__all__ = ['ModeSolver1D', 'ModeSolver2D', 'ModeSet', 'load_result', 'plot_dispersion']

__all__ += ["Material", "GoodConductor", "SurfaceImpedance", "materials", "shapes"]

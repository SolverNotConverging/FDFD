"""FDFD waveguide modes: supported material-first user API."""

from pathlib import Path as _Path
__path__.append(str(_Path(__file__).parent / "src"))

from fdfd_common import Material, GoodConductor, SurfaceImpedance, materials, shapes

from .api import ModeSolver1D, ModeSolver2D, ModeSet, load_result
__version__ = "1.1.0"
__all__ = ['ModeSolver1D', 'ModeSolver2D', 'ModeSet', 'load_result']

__all__ += ["Material", "GoodConductor", "SurfaceImpedance", "materials", "shapes"]

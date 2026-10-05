"""Supported material-first FDFD user API."""

from pathlib import Path as _Path
__path__.append(str(_Path(__file__).parent / "src"))

from .api import BandStructureSolver2D, BandStructureResult, load_result
__version__ = "1.1.0"
__all__ = ['BandStructureSolver2D', 'BandStructureResult', 'load_result']

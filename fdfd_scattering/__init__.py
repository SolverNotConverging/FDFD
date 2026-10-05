"""Supported material-first FDFD user API."""

from pathlib import Path as _Path
__path__.append(str(_Path(__file__).parent / "src"))

from .api import ScatteringSolver2D, ScatteringResult, load_result
__version__ = "1.1.0"
__all__ = ['ScatteringSolver2D', 'ScatteringResult', 'load_result']

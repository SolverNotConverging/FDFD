"""Standalone reader and plotting tools for WaveFEM HDF5 files."""

from .app import H5ViewerApp
from .model import FileData, ModeData, ResultData, SceneData, SceneLine
from .plotting import (
    SParameterRow,
    plot_modal_field,
    plot_s_parameter_sweep,
    plot_s_parameters,
    plot_scene,
    plot_vector_field,
    plot_vector_field_2d,
    s_parameter_label,
    s_parameter_rows,
)
from .reader import load_h5

__version__ = "0.1.0"

__all__ = [
    "FileData",
    "H5ViewerApp",
    "ModeData",
    "ResultData",
    "SParameterRow",
    "SceneData",
    "SceneLine",
    "load_h5",
    "plot_modal_field",
    "plot_s_parameter_sweep",
    "plot_s_parameters",
    "plot_scene",
    "plot_vector_field",
    "plot_vector_field_2d",
    "s_parameter_label",
    "s_parameter_rows",
]

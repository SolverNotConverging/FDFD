"""Build the optional periodic eigensolver kernel into this checkout."""
from pathlib import Path
import os
from setuptools import Extension, setup
from Cython.Build import cythonize

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
setup(
    name="periodic-eigensolver-kernel",
    packages=[],
    ext_modules=cythonize([Extension("periodic_eigensolver._cython_kernels",
        ["periodic_eigensolver/src/_cython_kernels.pyx"])],
        language_level=3, build_dir="build/cython"),
)

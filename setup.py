"""Package the Python solvers and compile the periodic eigensolver Cython kernel."""
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).resolve().parent
source_roots = [Path("src")]
source_roots += sorted(Path("libraries").glob("*/src"))
source_roots += sorted(Path("solvers/fdfd").glob("*/src"))
packages = []
package_dir = {}
for source in source_roots:
    for name in find_packages(where=str(ROOT / source)):
        packages.append(name)
        package_dir[name] = (source / name.replace(".", "/")).as_posix()

extension = Extension(
    "periodic_eigensolver._cython_kernels",
    ["libraries/periodic_eigensolver/src/periodic_eigensolver/_cython_kernels.pyx"],
)
setup(
    packages=packages,
    package_dir=package_dir,
    ext_modules=cythonize([extension], language_level=3, build_dir="build/cython"),
)

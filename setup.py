"""Install Python solvers, optionally including the Cython kernel."""
import os
from setuptools import Extension, setup
from pathlib import Path

ROOT = Path(__file__).resolve().parent
packages = [p.name for p in ROOT.iterdir()
            if (p / "__init__.py").is_file() and (p.name in ("fdfd", "periodic_eigensolver") or (p / "src").is_dir())]
build_kernel = os.environ.get("FDFD_BUILD_CYTHON") == "1"
extensions = []
if build_kernel:
    from Cython.Build import cythonize
    extensions = cythonize([Extension("periodic_eigensolver._cython_kernels",
          ["periodic_eigensolver/_cython_kernels.pyx"])],
          language_level=3, build_dir="build/cython")
setup(packages=packages, package_data={name: ["src/*.py"] for name in packages},
      options={"build": {"build_base": "build/fdfd-cython" if build_kernel else "build/fdfd-python"}},
      ext_modules=extensions)

"""Optional distribution build; checkout imports need no installation."""
from setuptools import Extension, setup
from Cython.Build import cythonize
from pathlib import Path

ROOT = Path(__file__).resolve().parent
packages = [p.name for p in ROOT.iterdir()
            if (p / "__init__.py").is_file() and (p.name == "fdfd" or (p / "src").is_dir())]
setup(packages=packages, package_data={name: ["src/*.py"] for name in packages},
      options={"build": {"build_base": "build/distribution"}},
      ext_modules=cythonize([Extension("periodic_eigensolver._cython_kernels",
          ["periodic_eigensolver/src/_cython_kernels.pyx"])],
          language_level=3, build_dir="build/cython"))

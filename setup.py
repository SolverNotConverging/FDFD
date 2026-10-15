"""Optional Cython kernel for the refined Arnoldi eigensolver.

All metadata lives in pyproject.toml. Set ``FDFD_BUILD_CYTHON=1`` (with Cython
and NumPy installed) to compile ``fdfd.core._arnoldi_cython``; otherwise the
pure-Python kernel is used and the wheel stays platform independent.
"""

import os

from setuptools import Extension, setup

extensions = []
if os.environ.get("FDFD_BUILD_CYTHON") == "1":
    from Cython.Build import cythonize

    extensions = cythonize(
        [Extension("fdfd.core._arnoldi_cython", ["src/fdfd/core/_arnoldi_cython.pyx"])],
        language_level=3,
    )

setup(ext_modules=extensions)

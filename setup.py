"""Build one distribution from the maintained solver/library source directories."""
import os
from pathlib import Path
import shutil

from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).parent
source_roots = [Path("src"), *sorted(Path("libraries").glob("*/src")),
                *sorted(Path("solvers").glob("*/*/src"))]
packages = []
package_dirs = {}
for source in source_roots:
    for package in find_packages(where=str(source)):
        packages.append(package)
        package_dirs[package] = str(source / package.replace(".", "/"))


class BuildPythonAndNative(build_py):
    def run(self):
        super().run()
        native = os.environ.get("FDFD_NATIVE_BUNDLE")
        if native:
            source = Path(native).resolve()
            for name in ("build-manifest.json", "SOURCE_INDEX.md", "bin/qt.conf",
                         "bin/transmission-line-calculator.exe",
                         "bin/fem-periodic-mode-viewer.exe",
                         "bin/fem-waveguide-scattering-viewer.exe"):
                if not (source / name).is_file():
                    raise RuntimeError(f"Incomplete qualified native bundle: {source / name}")
            shutil.copytree(source, Path(self.build_lib) / "fdfd/native", dirs_exist_ok=True)


setup(
    packages=packages,
    package_dir=package_dirs,
    include_package_data=False,
    ext_modules=[Extension(
        "periodic_eigensolver._cython_kernels",
        sources=["libraries/periodic_eigensolver/src/periodic_eigensolver/_cython_kernels.pyx"],
        optional=not bool(os.environ.get("FDFD_NATIVE_BUNDLE")),
    )],
    cmdclass={"build_py": BuildPythonAndNative},
)

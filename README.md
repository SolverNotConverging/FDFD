# Computational Electromagnetics

Version 1.0.0 organizes independently installable Python solver families and separately built native applications.

| Method | Problem | Package | Usage guide | API reference |
|---|---|---|---|---|
| FDFD | waveguide modes | `fdfd_waveguide_modes` | [Guide](doc/solvers/fdfd/waveguide_modes/guide.rst) | [API](doc/solvers/fdfd/waveguide_modes/API_REFERENCE.rst) |
| FDFD | periodic modes | `fdfd_periodic_modes` | [Guide](doc/solvers/fdfd/periodic_modes/guide.rst) | [API](doc/solvers/fdfd/periodic_modes/API_REFERENCE.rst) |
| FDFD | band structure | `fdfd_band_structure` | [Guide](doc/solvers/fdfd/band_structure/guide.rst) | [API](doc/solvers/fdfd/band_structure/API_REFERENCE.rst) |
| FDFD | scattering | `fdfd_scattering` | [Guide](doc/solvers/fdfd/scattering/guide.rst) | [API](doc/solvers/fdfd/scattering/API_REFERENCE.rst) |
| FEM | waveguide modes | `fem_waveguide_modes` | [Guide](doc/solvers/fem/waveguide_modes/guide.rst) | [API](doc/solvers/fem/waveguide_modes/API_REFERENCE.rst) |
| FEM | periodic modes | `fem_periodic_modes` | [Guide](doc/solvers/fem/periodic_modes/guide.rst) | [API](doc/solvers/fem/periodic_modes/API_REFERENCE.rst) |
| FEM | waveguide scattering | `fem_waveguide_scattering` | [Guide](doc/solvers/fem/waveguide_scattering/guide.rst) | [API](doc/solvers/fem/waveguide_scattering/API_REFERENCE.rst) |
| FEM | electrostatics | `fem_electrostatics` | [Guide](doc/solvers/fem/electrostatics/guide.rst) | [API](doc/solvers/fem/electrostatics/API_REFERENCE.rst) |

Waveguide scattering is **2.5D full-vector** physics on a 2D mesh. Electromagnetic solvers use `exp(+i*omega*t)`; passive material loss has a nonpositive imaginary part.

```text
solvers/      Independently installable Python FDFD and FEM families
examples/     Runnable tutorials organized by method and solver family
apps/         Native viewers and transmission-line calculator
libraries/    Shared CEM API, FEM adaptivity, and periodic eigensolver
doc/          Central user guides, API references, and contributor documentation
tests/        Solver, library, regression, and integration checks
benchmarks/   Numerical performance and convergence studies
tools/        User grid and PML calculators
scripts/      Development, build, and release tooling
outputs/      Ignored generated files
```

## Installation

Create the project environment from the repository root:

```sh
conda env create -f environment.yml
conda activate cem
python scripts/install_python.py --editable
python -m pytest
```

Installation and run scripts use the active Python interpreter. An existing environment with the required dependencies also works.

If you already have a development environment, activate it and run the installation
step from the repository root. Repeat this after moving or reorganizing the checkout;
old editable installations still point to their previous directories:

```sh
python scripts/install_python.py --editable --no-build-isolation
```

`--no-build-isolation` uses the build dependencies already present in the environment
(including setuptools, Cython, and NumPy, as listed in `environment.yml`). Omit it to
let pip provision isolated build dependencies.

## Quick start

From the repository root, run:

```sh
python examples/fem/electrostatics/parallel_plate_capacitor_1d.py
```

The example prints capacitor energy and opens an interactive potential and field
viewer. The [solver guides](doc/README.rst) explain how to configure geometry,
materials, meshing, and solving for each family.

The `src/` packages must be installed before running examples. Once installed in
the same Python environment, examples work from their own directories too:

```sh
cd examples/fem/electrostatics
python embedded_electrode_2d_anisotropic.py
```

If an example reports `ModuleNotFoundError`, check `python -m pip list --editable`
and rerun the installation command with that same interpreter.

The FEM workflow is `solver.mesh(...)`, `result = solver.solve(...)`, and `solver.show()`. Meshing can be automatic; geometry changes invalidate the current mesh and result. Solves do not open windows or save files. Use `result.plot(...)`, `result.save(path)`, and the family’s `load_result(path)` for completed results. See the [example index](examples/README.rst) for runnable tutorials and their recommended order.

Example names follow `<physical_problem>_<dimension>[_<feature>].py`. Scripts that
save results write under `outputs/examples/<method>/<family>/<example>/`, regardless
of the working directory. Postprocessing scripts read those results by default and
also accept an explicit input path. Python packages remain independently installable;
the tutorial collection lives in the checkout.

All FEM archives use `cem-fem-results` schema `1.0`, with physical units, field representation, and convention metadata. Loaded results support inspection rather than solver restart. Old imports, scattering phasors, and archive formats have no compatibility layer.

## Native applications

Build the native applications after installing Qt 6, HDF5, Eigen, Gmsh, and FTXUI; the periodic viewer also supports VTK:

```sh
cmake -S . -B outputs/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build outputs/build --parallel
ctest --test-dir outputs/build --output-on-failure
```

- [FEM Waveguide Scattering Viewer](apps/fem_waveguide_scattering_viewer/README.rst)
- [FEM Periodic Mode Viewer](apps/fem_periodic_mode_viewer/README.rst)
- [Transmission Line Calculator](apps/transmission_line_calculator/README.rst)

## Testing and release checks

Python tests live under `tests/fdfd/`, `tests/fem/`, and `tests/libraries/`;
cross-family tests remain directly under `tests/`. Run `python -m pytest` for all
Python tests, or select a family such as `python -m pytest tests/fem/electrostatics`.
Native C++ tests remain with their CMake applications.

Build and qualify Python release artifacts with:

```sh
python scripts/build_wheels.py
python scripts/qualify_wheels.py
python scripts/check_documentation.py
python scripts/qualify_examples.py
python scripts/qualify_native.py
```

Wheel qualification installs every package outside the checkout and checks the compiled eigensolver. Example qualification runs every solver example with viewer launches suppressed. Native qualification checks Python-written archives in the inspectors and offscreen viewers. Publishing is a separate operation. Release qualification is tracked in [the implementation record](doc/development/overhaul_status.md).

The [documentation index](doc/README.rst) contains all Python solver and library
guides and API references. Package READMEs are short navigation pages retained
for independent wheel metadata.

Shared library guides: [materials, shapes, and errors](doc/libraries/cem_common/guide.rst),
[FEM adaptivity](doc/libraries/fem_adaptivity/guide.rst), and
[periodic eigensolver](doc/libraries/periodic_eigensolver/guide.rst).


## Python native eigensolver

The optional Cython extension speeds up the shared periodic eigensolver. A source
installation requires a compiler compatible with the active Python ABI; official
Windows CPython normally uses MSVC. Cython, NumPy, SciPy, and setuptools build
dependencies are included in `environment.yml`. Without a compiler, a source
install can use the NumPy/SciPy fallback. A release binary wheel must contain the
extension; `scripts/build_wheels.py` enforces that requirement.

Check the active environment with:

```sh
python -c "from periodic_eigensolver import native_backend_available; print(native_backend_available())"
```

The extension uses SciPy's BLAS implementation. If a source install fell back to
Python and native acceleration is wanted, configure a compatible compiler and
rerun `python scripts/install_python.py --editable --no-build-isolation`.

## Analytical benchmarks

The [benchmark index](benchmarks/README.md) provides executable comparisons of
solver results with analytical waveguide and electrostatic solutions. These save
numerical error tables and plots under `outputs/benchmarks/analytical/`.

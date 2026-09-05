# Computational Electromagnetics

Version 1.0.0 organizes independently installable Python solver families and separately built native applications.

| Method | Problem | Package | User reference |
|---|---|---|---|
| FDFD | waveguide modes | `fdfd_waveguide_modes` | [API](solvers/fdfd/waveguide_modes/API_REFERENCE.rst) |
| FDFD | periodic modes | `fdfd_periodic_modes` | [API](solvers/fdfd/periodic_modes/API_REFERENCE.rst) |
| FDFD | band structure | `fdfd_band_structure` | [API](solvers/fdfd/band_structure/API_REFERENCE.rst) |
| FDFD | scattering | `fdfd_scattering` | [API](solvers/fdfd/scattering/API_REFERENCE.rst) |
| FEM | waveguide modes | `fem_waveguide_modes` | [API](solvers/fem/waveguide_modes/API_REFERENCE.rst) |
| FEM | periodic modes | `fem_periodic_modes` | [API](solvers/fem/periodic_modes/API_REFERENCE.rst) |
| FEM | waveguide scattering | `fem_waveguide_scattering` | [API](solvers/fem/waveguide_scattering/API_REFERENCE.rst) |
| FEM | electrostatics | `fem_electrostatics` | [API](solvers/fem/electrostatics/API_REFERENCE.rst) |

Waveguide scattering is **2.5D full-vector** physics on a 2D mesh. Electromagnetic solvers use `exp(+i*omega*t)`; passive material loss has a nonpositive imaginary part.

```text
solvers/      Python FDFD and FEM families, each with src/, examples/, and tests/
apps/         Native viewers and transmission-line calculator
libraries/    Shared FEM contracts, adaptivity, and periodic eigensolver
docs/         User and contributor documentation
tests/        Cross-package regression and integration checks
benchmarks/   Numerical performance and convergence studies
tools/        User grid and PML calculators
scripts/      Development, build, and release tooling
outputs/      Ignored generated files
```

Create the project environment from the repository root:

```sh
conda env create -f environment.yml
conda activate cem
python scripts/install_python.py --editable
python -m pytest
```

Installation and run scripts use the active Python interpreter. An existing environment with the required dependencies also works.

The FEM workflow is `solver.mesh(...)`, `result = solver.solve(...)`, and `solver.show()`. Meshing can be automatic; geometry changes invalidate the current mesh and result. Solves do not open windows or save files. Use `result.plot(...)`, `result.save(path)`, and the family’s `load_result(path)` for completed results. See each package README for runnable examples.

All FEM archives use `cem-fem-results` schema `1.0`, with physical units, field representation, and convention metadata. Loaded results support inspection rather than solver restart. Old imports, scattering phasors, and archive formats have no compatibility layer.

Build the native applications after installing Qt 6, HDF5, Eigen, Gmsh, and FTXUI; the periodic viewer also supports VTK:

```sh
cmake -S . -B outputs/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build outputs/build --parallel
ctest --test-dir outputs/build --output-on-failure
```

- [FEM Waveguide Scattering Viewer](apps/fem_waveguide_scattering_viewer/README.rst)
- [FEM Periodic Mode Viewer](apps/fem_periodic_mode_viewer/README.rst)
- [Transmission Line Calculator](apps/transmission_line_calculator/README.rst)

Build and qualify Python release artifacts with:

```sh
python scripts/build_wheels.py
python scripts/qualify_wheels.py
python scripts/check_documentation.py
python scripts/qualify_examples.py
python scripts/qualify_native.py
```

Wheel qualification installs every package outside the checkout and checks the compiled eigensolver. Example qualification runs the FEM examples with viewer launches suppressed. Native qualification checks Python-written archives in the inspectors and offscreen viewers. Publishing is a separate operation. Release qualification is tracked in [the implementation record](docs/development/overhaul_status.md).

# FDFD

Finite-Difference Frequency-Domain solvers for computational electromagnetics.

The repository is organised by problem type. Each solver folder contains the solver implementation, example scripts, and a solver-specific ``README.rst`` with API and workflow notes.

## Solver Map

### FEM solvers and viewers

| Folder | Solver | Use case | Documentation |
|---|---|---|---|
| `FEM_Mode_Solver/` | `ModeSolver1D`, `ModeSolver2D` | Standalone conforming-FEM modes with adaptive meshing and 2D SIBC conductors | [`README.rst`](FEM_Mode_Solver/README.rst) |
| `TransmissionLineCalculator/` | Native Qt/FTXUI quasi-TEM calculator | Fast Gmsh/P1-FEM coaxial, microstrip, stripline, and CPW extraction | [`README.md`](TransmissionLineCalculator/README.md) |
| `WaveFEM/` | Full-wave finite-element solver | 2D electromagnetic scattering, ports, modes, sweeps, and HDF5 results | [`README.md`](WaveFEM/README.md) |
| `WaveFEMViewer/` | Native Qt HDF5 viewer | Interactive inspection of WaveFEM schema-v1 fields, modes, and S-parameters | [`README.md`](WaveFEMViewer/README.md) |
| `FEM_Periodic_Solver/` | `PeriodicModeSolver2D`, `PeriodicModeSolver3D` | Self-contained P1/Nedelec fixed-frequency periodic FEM | [`README.rst`](FEM_Periodic_Solver/README.rst) |
| `FEMPeriodicViewer/` | Native Qt/HDF5 viewer and inspector | Lazy 2D/optional-VTK 3D FEM periodic result viewing | [`README.md`](FEMPeriodicViewer/README.md) |

### FDFD solvers

| Folder | Solver | Use case | Documentation |
|---|---|---|---|
| `Mode_Solver_1D/` | `ModeSolver1D` | TE/TM modes of 1D slab waveguides | [`README.rst`](Mode_Solver_1D/README.rst) |
| `Mode_Solver_2D/` | `ModeSolver2D` | Full-vector modes of 2D waveguide cross-sections | [`README.rst`](Mode_Solver_2D/README.rst) |
| `Periodic_Solver_2D/` | `PeriodicModeSolver2D` | 2D Bloch-periodic TE/TM unit-cell modes | [`README.rst`](Periodic_Solver_2D/README.rst) |
| `Periodic_Solver_3D/` | `PeriodicModeSolver3D` | 3D Bloch-periodic full-vector modes | [`README.rst`](Periodic_Solver_3D/README.rst) |
| `Band_Diagram_Solver/` | `BandDiagramSolver2D` | 2D photonic-crystal band diagrams | [`README.rst`](Band_Diagram_Solver/README.rst) |
| `Scattering/` | `FDFD2DScatteringSolver` | 2D TEz/TMz scattering problems | [`README.rst`](Scattering/README.rst) |
| `Electrostatic_Solver/` | `ElectrostaticSolver` | 1D/2D electrostatic potential problems | [`README.rst`](Electrostatic_Solver/README.rst) |

The shared [`periodic_eigensolver/`](periodic_eigensolver/README.md) package
provides the Cython/BLAS refined shift-and-invert Arnoldi backend used by both
the periodic FDFD and FEM solvers.

Utility scripts at the repository root include `Mesh_points_calculation.py` and `PML_sigma_calculation.py`.

## Requirements

The solvers are plain Python modules. A typical environment needs:

```bash
pip install numpy scipy matplotlib
```

Some visualizers use Tk through Matplotlib. If GUI windows do not open, install the Tk package for your Python distribution.

The standalone FEM packages additionally need scikit-fem and Gmsh; their
installation and environments are documented in
[`FEM_Mode_Solver/README.rst`](FEM_Mode_Solver/README.rst) and
[`FEM_Periodic_Solver/README.rst`](FEM_Periodic_Solver/README.rst).

The native C++20 applications support MinGW-w64 or MSVC on Windows, AppleClang on
macOS, and GCC or Clang on Linux. The native viewers use Qt 6.2 or newer.
The transmission line calculator additionally needs FTXUI, Eigen 3.4 or newer, and Gmsh; the
WaveFEM and FEM periodic viewers need HDF5 1.10 or newer. The FEM periodic
viewer can optionally use VTK 9.2 or newer. Platform-specific dependency and
build commands are documented in the
[`TransmissionLineCalculator`](TransmissionLineCalculator/README.md) and
[`WaveFEMViewer`](WaveFEMViewer/README.md), and
[`FEMPeriodicViewer`](FEMPeriodicViewer/README.md) READMEs.

### Build native applications

When the dependencies are in CMake's normal search path, configure all applications from
the repository root with any single-configuration generator:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For Visual Studio and other multi-configuration generators, select the
configuration at build and test time:

```powershell
cmake -S . -B build-msvc -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build-msvc --config Release --parallel
ctest --test-dir build-msvc -C Release --output-on-failure
```

The root build can be limited to one application with
`-DFDFD_BUILD_TRANSMISSION_LINE_CALCULATOR=OFF` or
`-DFDFD_BUILD_WAVEFEM_VIEWER=OFF`, and
`-DFDFD_BUILD_FEM_PERIODIC_VIEWER=OFF`. The periodic viewer's optional VTK
mode is selected with `-DFEM_PERIODIC_VIEWER_WITH_VTK=AUTO|ON|OFF`.
`CMAKE_PREFIX_PATH` may be supplied when
Qt or the other libraries are installed outside the platform's standard
locations.

## Basic Workflow

1. Pick the solver folder that matches the physics.
2. Read that folder's `README.rst` for the API and expected outputs.
3. Run or modify the example script in the same folder.
4. Keep generated data in the folder's `example_outputs/` directory.

## Examples

```bash
cd Mode_Solver_1D
python example_grounded_isotropic_slab.py
```

```bash
cd Mode_Solver_2D
python example_ridge_dielectric_waveguide.py
```

```bash
python -m FEM_Mode_Solver.examples.slab_1d
python -m FEM_Mode_Solver.examples.ridge_2d
python -m FEM_Mode_Solver.examples.microstrip_sibc
```

```bash
python -m FEM_Periodic_Solver.examples.leaky_wave_antenna_2d
python -m FEM_Periodic_Solver.examples.iris_loaded_waveguide_filter_3d
```

```bash
cd Band_Diagram_Solver
python example_square_lattice.py
```

## Notes

- Length units are SI metres unless an example explicitly normalises geometry.
- Material values are relative tensors unless stated otherwise.
- Large grids produce large sparse systems. Start with coarse examples before increasing resolution.
- Several solvers use shift-invert eigensolves. If convergence is poor, adjust the mode count, grid size, or eigenvalue guess.

## Reference

R. Rumpf, *Electromagnetic and Photonic Simulation for the Beginner: Finite-Difference Frequency-Domain in MATLAB*. Artech House, 2022.

# FDFD v1.0.0

FDFD v1.0.0 intentionally uses simplified Python syntax to make
electromagnetic calculations easier to set up and explore. Define a solver, define
reusable materials, add geometry, mesh, solve, and inspect the results in a GUI.
You can change field components, modes, and display options without writing a new
plotting script for each view.

FDFD remains the project's name, including its FEM solvers. Version 1.0.0 installs
all eight solver families, shared libraries, and three native applications together
from one Windows wheel. See the [release history](doc/development/release_history.md)
for changes since the earlier FDFD releases.

[Installation](#installation) · [Quick start and GUI example](#quick-start) ·
[Native apps](#native-applications) · [Examples](examples/README.rst) ·
[Benchmarks](benchmarks/README.md)

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
solvers/      Python FDFD and FEM solver implementations
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

**Windows x64 with Python 3.12:** install the complete release wheel. It includes
all solvers, the compiled periodic eigensolver, both native FEM viewers, and the
Transmission Line Calculator, with their runtime DLLs and Qt plugins.
**Linux and macOS users must [build from source](#build-from-source)**; this release
provides no wheels for those platforms or for other Python versions.

### 1. Create the FDFD Python environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create
the project-local environment. FDFD pins Python 3.12 in `.python-version`, and uv
downloads that interpreter when it is not already installed:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
```

Activation is optional when commands are prefixed with `uv run`.

### 2. Install everything with one command

Install the release into the FDFD environment:

```sh
uv pip install "https://github.com/SolverNotConverging/FDFD/releases/download/v1.0.0/fdfd-1.0.0-cp312-cp312-win_amd64.whl"
```

uv installs FDFD and its numerical Python dependencies. No repository clone,
compiler, Qt installer, vcpkg, or separate native-app installation is needed.
The wheel is larger than the solver code because it includes the native runtimes.
It is distributed through GitHub Releases; use the complete URL above.

You can also download the single `.whl` from the
[FDFD v1.0.0 release](https://github.com/SolverNotConverging/FDFD/releases/tag/v1.0.0)
and install the local file with `uv pip install path/to/fdfd-1.0.0-cp312-cp312-win_amd64.whl`.
Use a fresh environment if you previously installed the separate internal packages,
so multiple distributions do not own the same Python files.

### 3. Check the installation and open an app

```sh
uv pip check
python -m fdfd info
python -m fdfd calculator
```

`info` should report the compiled eigensolver as `True` and display all three
bundled executable paths. The calculator opens a GUI and runs without a Python
solver script. Open the native viewers with:

```sh
python -m fdfd periodic-viewer
python -m fdfd scattering-viewer
```

`solver.show()` and `result.show()` automatically find the bundled viewers.
No viewer environment variables need to be set for the wheel installation.
The executables can also be run directly from the paths printed by `info`.
In a terminal, `python -m fdfd calculator-cli` opens the calculator's terminal UI.
A desktop session is needed for GUI windows; the periodic 3D view uses OpenGL.

To run the tutorial files, optionally install [Git](https://git-scm.com/downloads)
and clone the repository into your chosen working folder:

```sh
git clone https://github.com/SolverNotConverging/FDFD.git
cd FDFD
python examples/fem/waveguide_modes/microstrip_2d_surface_impedance.py
```

The wheel already supplies the solver packages. You can also copy the short
Python example below into your own script without cloning anything.

| Installation problem | Fix |
|---|---|
| Wheel is not supported on this platform | Check `python --version` is 3.12 and `python -c "import struct; print(struct.calcsize('P') * 8)"` prints 64. Use Windows x64, or build from source on Linux/macOS. |
| `ModuleNotFoundError` after installing | Check `python -c "import sys; print(sys.executable)"` and select that environment in your terminal/IDE. |
| An older viewer opens | Remove an old `FEM_PERIODIC_MODE_VIEWER_EXECUTABLE` or `FEM_WAVEGUIDE_SCATTERING_VIEWER_EXECUTABLE` override, then restart your terminal/IDE. Explicit overrides take priority over the bundle. |

## Quick start

The general workflow is:

```text
Define solver → Define materials → Add geometry → Mesh → Solve → Show results
```

A small 10 GHz copper microstrip example follows. All lengths are in metres;
`epsilon` is relative permittivity. This model uses a finite PEC enclosure, an
air background, a lossy substrate, and copper surface-impedance boundaries.

```python
from cem_common import Material, materials
from fem_waveguide_modes import ModeSolver2D

# 1. Define the solver and its physical domain.
x_range = (-6e-3, 6e-3)
solver = ModeSolver2D(
    frequency=10e9, x_range=x_range, y_range=(-35e-6, 6e-3),
    background_material=materials.air, boundary=materials.PEC,
)

# 2. Define materials; copper is a built-in good-conductor SIBC preset.
substrate = Material(name="microwave laminate", epsilon=3.55 * (1 - 0.0027j))
copper = materials.copper

# 3. Assign materials to the substrate, ground plane, and strip.
solver.add_rectangle(x_range=x_range, y_range=(0, 1.524e-3), material=substrate)
solver.add_rectangle(x_range=x_range, y_range=(-35e-6, 0), material=copper)
solver.add_rectangle(
    x_range=(-1.5e-3, 1.5e-3), y_range=(1.524e-3, 1.559e-3), material=copper,
)

# 4. Mesh, solve for one mode, and open the interactive results viewer.
solver.mesh(max_element_size=0.6e-3, wavelength_elements=10, material_aware=True)
result = solver.solve(num_modes=1, neff_guess=1.65, max_refinements=0)
solver.show()
```

In the GUI, select **E**, **magnitude**, **mesh**, and **normalize** to obtain
the view below. The field is strongest near the strip edges, where the mesh is
also finer. The example gives approximately `neff = 1.7001 - 0.00276j`;
the negative imaginary part represents passive forward attenuation.

![Microstrip electric-field magnitude in the interactive FEM viewer, with mesh overlay and normalized color scale](doc/assets/microstrip_mode_e_mesh_gui.png)

*Actual viewer screenshot from the [copper microstrip example](examples/fem/waveguide_modes/microstrip_2d_surface_impedance.py).
Copper interiors are excluded from the mesh. Field amplitudes are normalized
for display; this is an eigenmode calculation, not an applied-voltage simulation.*

Run the complete example after installation:

```sh
python examples/fem/waveguide_modes/microstrip_2d_surface_impedance.py
```

`solver.show()` and `result.show()` open interactive GUIs. Waveguide modes and
electrostatics use Matplotlib; FEM periodic modes and waveguide scattering use
the native viewers included in the complete wheel. For a static figure, use
`figure = result.plot(component="E", quantity="magnitude", mode=0)` and
`figure.savefig("microstrip.png")`. The [solver guides](doc/README.rst) explain
the available controls and physics-specific operations.

The `src/` packages must be installed before running examples. Once installed in
the same Python environment, examples work from their own directories too:

```sh
cd examples/fem/electrostatics
python embedded_electrode_2d_anisotropic.py
```

If an example reports `ModuleNotFoundError`, check `python -c "import sys; print(sys.executable)"`
and `uv pip show fdfd`.
Activate the environment where you installed the packages, or repeat your chosen
installation method with the interpreter used to run the example.

Meshing can be automatic; geometry changes invalidate the current mesh and result.
Solves do not open windows or save files. Use `result.save(path)` and the family’s
`load_result(path)` to inspect a completed result later without solving again.
The example selects a fixed mesh with `max_refinements=0`; FEM solves otherwise
default to up to two adaptive refinements. See the [example index](examples/README.rst)
for runnable tutorials and their recommended order.

Example names follow `<physical_problem>_<dimension>[_<feature>].py`. Scripts that
save results write under `outputs/examples/<method>/<family>/<example>/`, regardless
of the working directory. Postprocessing scripts read those results by default and
also accept an explicit input path. All solver imports come from the single FDFD
distribution; the tutorial collection lives in the checkout.

All FEM archives use `cem-fem-results` schema `1.0`, with physical units, field representation, and convention metadata. Loaded results support inspection rather than solver restart. Old imports, scattering phasors, and archive formats have no compatibility layer.

## Build from source

Install [Git](https://git-scm.com/downloads), [uv](https://docs.astral.sh/uv/),
a C/C++ toolchain, and the native development libraries below. Then run:

```sh
git clone https://github.com/SolverNotConverging/FDFD.git
cd FDFD
uv sync
uv run python -m fdfd info
```

`uv sync` creates the checkout's `.venv`, installs the locked Python dependencies,
and automatically builds and installs the Cython extension and all three native
applications through scikit-build-core. No separate native build or install step
is needed. The checkout defaults to Python 3.12; source builds support 3.11–3.13.

### Native prerequisites

Use a C++20 compiler and standard library supporting `std::format`: MinGW or MSVC
on Windows, Apple Clang on macOS, or GCC/Clang on Linux. Install CMake 3.24+ and
a suitable build tool (Ninja, Make, or Visual Studio), plus the libraries and
development headers below. All native dependencies must match the compiler and
architecture; MSVC and MinGW libraries cannot be mixed.

| Library | Required components |
|---|---|
| Qt 6.2+ | Widgets and Concurrent; OpenGL for 3D |
| HDF5 1.10+ | C library |
| Eigen 3.4+ | Headers |
| Gmsh 4 | C++ library and headers, with OpenCASCADE support |
| FTXUI | Component, DOM, and screen libraries |
| VTK 9.2+ | Optional periodic 3D viewer: Qt 6 integration and OpenGL rendering |

Install these through your platform's package manager or SDK installer. The
Python `gmsh` and `h5py` packages do not replace the native development libraries.
Without a compatible VTK installation, the periodic viewer builds with 2D support.

For MSVC, use an **x64 Visual Studio Developer PowerShell** and a
[vcpkg installation](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started).
Install the dependencies once, then point the build at that installation:

```powershell
$vcpkgRoot = "C:\dev\vcpkg"
& "$vcpkgRoot/vcpkg.exe" install "qtbase[concurrent,widgets,opengl]" hdf5 eigen3 "gmsh[occ]" ftxui "vtk[qt,opengl]" --triplet x64-windows
$env:CMAKE_ARGS = "-DCMAKE_TOOLCHAIN_FILE=$vcpkgRoot/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows"
uv sync
```

For MinGW, macOS, or Linux, make the compiler and dependency installation
discoverable in your build shell. Set `CMAKE_PREFIX_PATH` if the libraries are
outside the usual search paths. Platform-specific prerequisite details are in
the [periodic viewer](apps/fem_periodic_mode_viewer/README.rst),
[scattering viewer](apps/fem_waveguide_scattering_viewer/README.rst), and
[calculator](apps/transmission_line_calculator/README.rst) guides.

Keep the native libraries installed: a local source build uses their runtime
files. If changing compilers in the same checkout, use a fresh build directory,
for example `uv sync --reinstall-package fdfd --config-setting build-dir=build/new-toolchain`.
To rebuild after native source changes, use `uv sync --reinstall-package fdfd`
with the same toolchain settings.

## Native applications

After installing the release wheel or running `uv sync`, open an app with:

```sh
uv run python -m fdfd calculator
uv run python -m fdfd periodic-viewer
uv run python -m fdfd scattering-viewer
```

The viewers inspect saved HDF5 results; the calculator includes its own solver.
`solver.show()` and `result.show()` discover the installed viewers automatically.
Run an example from the checkout with:

```sh
uv run python examples/fem/waveguide_modes/microstrip_2d_surface_impedance.py
```

## Testing and release checks

Python tests live under `tests/fdfd/`, `tests/fem/`, and `tests/libraries/`;
cross-family tests remain directly under `tests/`. Run `python -m pytest` for all
Python tests, or select a family such as `python -m pytest tests/fem/electrostatics`.
Native C++ tests remain with their CMake applications.

On the Windows release build machine, build/test C++ first, then prepare the
runtimes and build the single complete wheel:

```sh
python scripts/qualify_native.py
python scripts/package_native_windows.py --phase stage
python scripts/package_native_windows.py --phase finish
uv run python scripts/build_wheels.py
python scripts/qualify_wheels.py --fresh
python scripts/check_documentation.py
python scripts/qualify_examples.py
python scripts/qualify_native.py
```

Wheel qualification installs the one wheel outside the checkout and checks every solver family, the compiled eigensolver, native applications, and launch commands. The `--fresh` check downloads Python dependencies into a clean environment. Example qualification runs every solver example with viewer launches suppressed. Native qualification checks Python-written archives in the inspectors and offscreen viewers. Published changes are recorded in the [release history](doc/development/release_history.md).

The [documentation index](doc/README.rst) contains all Python solver and library
guides and API references. Solver and library READMEs are short navigation pages.
Native dependency licenses, source archives, and build recipes are recorded in
the [source index](doc/development/native_dependency_sources.md) and inside the wheel.

Shared library guides: [materials, shapes, and errors](doc/libraries/cem_common/guide.rst),
[FEM adaptivity](doc/libraries/fem_adaptivity/guide.rst), and
[periodic eigensolver](doc/libraries/periodic_eigensolver/guide.rst).


## Python native eigensolver

The Cython extension speeds up the shared periodic eigensolver. A source
installation requires a compiler compatible with the active Python ABI; official
Windows CPython normally uses MSVC. Cython, cython-cmake, NumPy, and SciPy are
isolated build dependencies in `pyproject.toml`. CMake always builds the extension,
and `scripts/build_wheels.py` verifies it in release wheels.

Check the active environment with:

```sh
python -c "from periodic_eigensolver import native_backend_available; print(native_backend_available())"
```

The extension uses SciPy's BLAS implementation and is built automatically by
`uv sync`. If compilation fails, check the compiler setup and rerun `uv sync`.

## Analytical benchmarks

The [benchmark index](benchmarks/README.md) provides executable comparisons of
solver results with analytical waveguide and electrostatic solutions. These save
numerical error tables and plots under `outputs/benchmarks/analytical/`.

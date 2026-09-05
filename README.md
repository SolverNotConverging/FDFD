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

### 1. Create or activate a Python 3.12 environment

Use an existing **64-bit CPython 3.12** environment, or choose one of these options.
You only need one environment manager. Commands below use Windows PowerShell.

**Python venv:** install [Python 3.12 for Windows](https://www.python.org/downloads/windows/)
with its Python launcher and Tcl/Tk components, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of
`python` in subsequent commands; activation is optional. See Python's
[venv documentation](https://docs.python.org/3.12/library/venv.html).

**Conda:** install [Miniforge](https://github.com/conda-forge/miniforge#install),
Miniconda, or Anaconda, open its prompt, and run:

```sh
conda create --name fdfd --channel conda-forge python=3.12 pip tk
conda activate fdfd
```

**uv:** [install uv](https://docs.astral.sh/uv/getting-started/installation/), then
create an environment with pip included (`--seed`):

```powershell
uv venv --python 3.12 --seed .venv
.\.venv\Scripts\Activate.ps1
```

uv can download Python 3.12 if necessary. See its
[environment guide](https://docs.astral.sh/uv/pip/environments/).

### 2. Install everything with one command

Run this in your chosen environment:

```sh
python -m pip install "https://github.com/SolverNotConverging/FDFD/releases/download/v1.0.0/fdfd-1.0.0-cp312-cp312-win_amd64.whl"
```

Pip installs FDFD and its numerical Python dependencies. No repository clone,
compiler, Qt installer, vcpkg, or separate native-app installation is needed.
The wheel is larger than the solver code because it includes the native runtimes.
It is distributed through GitHub Releases; use the complete URL above.

You can also download the single `.whl` from the
[FDFD v1.0.0 release](https://github.com/SolverNotConverging/FDFD/releases/tag/v1.0.0)
and install the local file with `python -m pip install path/to/fdfd-1.0.0-cp312-cp312-win_amd64.whl`.
Use a fresh environment if you previously installed the separate internal packages,
so multiple distributions do not own the same Python files.

### 3. Check the installation and open an app

```sh
python -m pip check
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
and `python -m pip show fdfd`.
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

This is the installation route for **Linux and macOS**, and for developers or
Windows users who want to compile their own build. The Windows release wheel
above is the default for end users. Source builds support Python 3.11–3.13.

Install [Git](https://git-scm.com/downloads) and
[Miniforge](https://github.com/conda-forge/miniforge#install) (an existing conda
installation also works). Open its prompt on Windows, or a terminal on Linux/macOS:

```sh
git clone https://github.com/SolverNotConverging/FDFD.git
cd FDFD
conda env create --name fdfd -f environment.yml
conda activate fdfd
python -m pip install --no-build-isolation .
python -m pip check
python examples/fem/waveguide_modes/microstrip_2d_surface_impedance.py
```

Run installation commands from the root directory containing `pyproject.toml`.
Conda installs numerical/build dependencies; pip builds and installs the one FDFD
source distribution. Internet access is needed to download dependencies. For an
editable development installation use `python -m pip install --editable --no-build-isolation .`.
Repeat installation after changing metadata or compiled extension code, or moving
an editable checkout. `python scripts/install_python.py` is a convenience wrapper
around the same root-package installation using the active interpreter.

The optional Cython eigensolver needs a C compiler: MSVC on Windows, GCC on Linux,
or Apple's command-line tools on macOS. Without one, source installation can use
the NumPy/SciPy fallback. `python -m fdfd info` reports whether acceleration is active.

Source installs build Python solvers first. Build the C++ applications separately
using the sections below, then point the two viewer environment variables at your
executables. The microstrip example uses Matplotlib and works before building the
native apps. On Linux/macOS, follow the app-specific prerequisites and commands:

- [FEM Waveguide Scattering Viewer](apps/fem_waveguide_scattering_viewer/README.rst)
- [FEM Periodic Mode Viewer](apps/fem_periodic_mode_viewer/README.rst)
- [Transmission Line Calculator](apps/transmission_line_calculator/README.rst)

## Native applications

The following steps are for building applications **from source**. Windows wheel
users already have all three native apps installed. The two FEM viewers inspect
saved HDF5 results; the Transmission Line Calculator includes its own solver.

### Windows: MSVC and vcpkg, step by step

These instructions build all three apps for **64-bit Windows**. MSVC compiles C++,
CMake configures the build, and vcpkg builds and supplies the external libraries.
Use one compiler/architecture throughout. The Python conda environment supplies
Python dependencies; native dependencies come from vcpkg.

#### 1. Install the Windows build tools

Download [Visual Studio 2022 Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe)
and run the installer. The full Visual Studio 2022 IDE also works. Select the
**Desktop development with C++** workload and include these components:

- **MSVC v143 – VS 2022 C++ x64/x86 build tools** (the current VS 2022 update).
- A **Windows 10 or Windows 11 SDK**.
- **C++ CMake tools for Windows**; this project needs CMake 3.24 or newer.

Click **Install**, allow the installation to finish, then open **Developer
PowerShell for VS 2022** from the Start menu. Use this terminal for all native
commands below. Keep it separate from your conda/MinGW terminal to avoid loading
their Qt DLLs. Install Git if you have not already done so, then check:

```powershell
git --version
cmake --version
cl
```

`cl` should print the Microsoft compiler banner (with no source file it may also
report that a filename is missing). If it is not recognized, reopen the Developer
PowerShell shortcut after installing the workload. Microsoft's
[C++ installation walkthrough](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)
shows the installer screens.

#### 2. Download vcpkg and install the dependencies

Use a short writable directory for vcpkg. The example uses `C:\dev\vcpkg`; you can
choose another location by changing `$vcpkgRoot`. Run this once:

```powershell
New-Item -ItemType Directory -Force C:\dev | Out-Null
$vcpkgRoot = "C:\dev\vcpkg"
git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
& "$vcpkgRoot/bootstrap-vcpkg.bat"
& "$vcpkgRoot/vcpkg.exe" install "qtbase[concurrent,widgets,opengl,windeployqt]" hdf5 eigen3 "gmsh[occ]" ftxui "vtk[qt,opengl]" --triplet x64-windows
```

If you already have a current vcpkg checkout, set `$vcpkgRoot` to it and run the
install command. Wait for each command to succeed before continuing. The first
dependency build, particularly Qt/VTK/OpenCASCADE, can take a long time and use
many gigabytes of disk space. vcpkg downloads and builds transitive dependencies
automatically; a separate Qt installer is unnecessary. See Microsoft's
[vcpkg setup guide](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started).

| Library / vcpkg port | Used by | Required feature |
|---|---|---|
| Qt 6.2+ / `qtbase` | All GUIs | Widgets, Concurrent; OpenGL for 3D; `windeployqt` for Windows deployment |
| HDF5 1.10+ / `hdf5` | Both viewers | C library for result archives |
| Eigen 3.4+ / `eigen3` | Calculator | Linear algebra headers |
| Gmsh 4 / `gmsh[occ]` | Calculator | OpenCASCADE geometry operations |
| `ftxui` | Calculator CLI | Terminal interface components |
| VTK 9.2+ / `vtk[qt,opengl]` | Periodic viewer's 3D display | Qt 6 integration and OpenGL rendering |

The [`gmsh` port's `occ` feature](https://vcpkg.io/en/package/gmsh.html) is necessary
for the calculator's geometry, and [`qtbase`'s `windeployqt` feature](https://vcpkg.io/en/package/qtbase.html)
installs the deployment tool. The [`vtk` Qt feature](https://vcpkg.io/en/package/vtk.html)
provides the Qt integration required by the periodic viewer.

For a smaller **2D-only periodic viewer** build, omit `"vtk[qt,opengl]"` from the
install command and use `-DFEM_PERIODIC_MODE_VIEWER_WITH_VTK=OFF` below. All three
apps still build, but the periodic viewer has no interactive 3D viewport.

#### 3. Configure and compile FDFD's native apps

In Developer PowerShell, change into the root of your FDFD checkout. If you only
installed wheels and have no checkout yet:

```powershell
cd C:\dev
git clone https://github.com/SolverNotConverging/FDFD.git
cd FDFD
```

If you already cloned FDFD elsewhere, use `cd` with that folder's absolute path
instead. From the directory containing the root `CMakeLists.txt`, run:

```powershell
$vcpkgRoot = "C:\dev\vcpkg"
$vcpkgPrefix = Join-Path $vcpkgRoot "installed/x64-windows"
cmake -S . -B outputs/build-msvc -G "Visual Studio 17 2022" -A x64 `
  "-DCMAKE_TOOLCHAIN_FILE=$vcpkgRoot/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DFEM_PERIODIC_MODE_VIEWER_WITH_VTK=ON
cmake --build outputs/build-msvc --config Release --parallel 2
```

The trailing backtick continues a PowerShell command onto the next line; do not
put spaces after it. Configure locates the dependencies and writes build files;
build produces the executables. Two parallel compile jobs limit peak memory use;
increase this number if your machine has capacity. Executables are placed under
`outputs/build-msvc/apps/<app_directory>/Release/`. If you change compiler or
architecture, choose a new build directory instead of reusing another toolchain's
CMake cache.

Make the vcpkg runtime libraries and Qt plugins visible in this terminal, then test:

```powershell
$env:Path = "$vcpkgPrefix/bin;$env:Path"
$env:QT_PLUGIN_PATH = & "$vcpkgPrefix/tools/Qt6/bin/qtpaths.exe" --query QT_INSTALL_PLUGINS
ctest --test-dir outputs/build-msvc -C Release --output-on-failure
```

CTest runs numerical, archive-reader, and GUI startup checks. A failing test prints
its diagnostic output; resolve failures before installing.

#### 4. Install the executables and their runtime files

Install to a directory owned by your Windows user, then copy the release DLLs and
deploy the Qt plugins. Continue in the same Developer PowerShell terminal:

```powershell
$installRoot = Join-Path $env:LOCALAPPDATA "FDFD"
$installBin = Join-Path $installRoot "bin"
cmake --install outputs/build-msvc --config Release --prefix "$installRoot"
Copy-Item "$vcpkgPrefix/bin/*.dll" -Destination $installBin -Force
$deployQt = Join-Path $vcpkgPrefix "tools/Qt6/bin/windeployqt.exe"
$guiApps = @(
  "fem-waveguide-scattering-viewer.exe",
  "fem-periodic-mode-viewer.exe",
  "transmission-line-calculator.exe"
)
foreach ($app in $guiApps) {
  & $deployQt --release --compiler-runtime --dir "$installBin" (Join-Path $installBin $app)
  if ($LASTEXITCODE -ne 0) { throw "Qt deployment failed for $app" }
}
```

CMake installs the applications; the DLL copy supplies the vcpkg release runtimes
(including HDF5, Gmsh, and VTK). Qt's
[`windeployqt`](https://doc.qt.io/qt-6/windows-deployment.html) supplies the Qt plugins
and compiler runtime deployment. Keep the resulting `bin` directory together;
copying a lone `.exe` does not install its dependencies. This recipe copies the
release DLLs from the chosen vcpkg installation, so a dedicated vcpkg checkout
keeps the local bundle smaller.

Open the calculator or a viewer from PowerShell:

```powershell
& "$env:LOCALAPPDATA/FDFD/bin/transmission-line-calculator.exe"
& "$env:LOCALAPPDATA/FDFD/bin/fem-waveguide-scattering-viewer.exe"
& "$env:LOCALAPPDATA/FDFD/bin/fem-periodic-mode-viewer.exe"
```

You can also open `%LOCALAPPDATA%\FDFD\bin` in File Explorer and create shortcuts
to those executables. `transmission-line-calculator-cli.exe` is the terminal UI;
run it from a terminal. The two companion `*-inspect.exe` programs read HDF5
archives without opening a GUI.

#### 5. Connect the viewers to Python

Set these **user-level** environment variables once so `result.show()` finds the
installed viewers from any checkout or wheel environment:

```powershell
[Environment]::SetEnvironmentVariable("FEM_WAVEGUIDE_SCATTERING_VIEWER_EXECUTABLE", "$env:LOCALAPPDATA/FDFD/bin/fem-waveguide-scattering-viewer.exe", "User")
[Environment]::SetEnvironmentVariable("FEM_PERIODIC_MODE_VIEWER_EXECUTABLE", "$env:LOCALAPPDATA/FDFD/bin/fem-periodic-mode-viewer.exe", "User")
```

Restart your terminal or IDE to inherit these settings, activate your Python environment, and run
a FEM periodic or scattering example. The variables must name the executable,
not its containing folder. To open saved results directly, pass the `.h5` path
as an argument to either viewer.

### Native installation troubleshooting

| Symptom | Check or fix |
|---|---|
| `git`, `cmake`, or `cl` is not recognized | Install the tools in step 1 and reopen Developer PowerShell for VS 2022. |
| CMake cannot find Qt, HDF5, Gmsh, or VTK | Check that vcpkg finished successfully, `$vcpkgRoot` is correct, and both the vcpkg triplet and CMake architecture are x64. Pass the toolchain file on the first configure. |
| Calculator reports OpenCASCADE is unavailable | Install `gmsh[occ]`, rebuild, and repeat runtime deployment. |
| `windeployqt.exe` is missing | Install the `qtbase[windeployqt]` feature and use the tool from that same vcpkg checkout. |
| Missing DLL, or Qt cannot load its Windows platform plugin | Repeat step 4; keep `bin/platforms/qwindows.dll` and all copied DLLs. Avoid Qt plugin paths pointing at another installation. Install the [x64 MSVC runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) if it is missing. |
| Python reports that a viewer cannot be found | Check step 5's executable paths and restart the terminal/IDE. |

### Other native build options

App-specific documentation covers standalone builds, MSYS2/MinGW, macOS, and Linux.
The bundled PowerShell `install.ps1` scripts are **MSYS2/MinGW installers**; use
the CMake install and deployment steps above for MSVC.

To build only selected apps from the root, add the appropriate `OFF` switches to
the configure command (all default to `ON`):

```text
-DCEM_BUILD_TRANSMISSION_LINE_CALCULATOR=OFF
-DCEM_BUILD_FEM_WAVEGUIDE_SCATTERING_VIEWER=OFF
-DCEM_BUILD_FEM_PERIODIC_MODE_VIEWER=OFF
```

- [FEM Waveguide Scattering Viewer](apps/fem_waveguide_scattering_viewer/README.rst)
- [FEM Periodic Mode Viewer](apps/fem_periodic_mode_viewer/README.rst)
- [Transmission Line Calculator](apps/transmission_line_calculator/README.rst)

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
python scripts/build_wheels.py --no-build-isolation
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
rerun your source installation command.

## Analytical benchmarks

The [benchmark index](benchmarks/README.md) provides executable comparisons of
solver results with analytical waveguide and electrostatic solutions. These save
numerical error tables and plots under `outputs/benchmarks/analytical/`.

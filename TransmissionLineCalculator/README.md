# Transmission Line Calculator

Transmission Line Calculator is a native C++20/Qt 6 application for fast
quasi-TEM extraction of coaxial, microstrip, stripline, and coplanar-waveguide
cross-sections. It is a standalone rewrite of the Python calculator: the GUI,
mesher, sparse finite-element solve, field plots, and command-line benchmark
run without Python, NumPy, SciPy, scikit-fem, or Matplotlib. Geometry and
conforming triangular meshing use the native Gmsh 4 C++ API.

The numerical core is also available as the reusable `tl-core` CMake target.

## What it computes

The solver asks Gmsh to create a conforming triangular mesh and solves two scalar,
first-order (P1) finite-element potential problems on it:

1. the physical complex dielectric gives the electric field and complex
   capacitance per unit length;
2. the same cross-section filled with vacuum gives vacuum capacitance and the
   unit-current magnetic-field dual.

The result contains the native mesh and transverse E/H samples as well as
`n_eff`, propagation constant, circuit characteristic impedance, field wave
impedance, power, and the complete R/L/G/C per-length model. Optional finite
bulk metal conductivity uses a first-order good-conductor surface-impedance
correction on the named signal and ground conductors. The artificial remote
PEC truncation boundary is never counted as lossy metal.

The phasor convention is `exp(+j omega t - j beta z)`. Passive attenuation
therefore appears as a non-positive imaginary part of `beta` and `n_eff`.

## Requirements

The build requires:

- a C++20 compiler (MSVC, AppleClang, GCC, or Clang);
- CMake 3.24 or newer;
- Qt 6.2 or newer (`Widgets` and `Concurrent`);
- Eigen 3.4 or newer (Eigen 5 is also accepted);
- Gmsh 4, including `gmsh.h` and its C++ library.

Keep every dependency built for the same compiler, architecture, and runtime
as the application. For example, an MSVC build cannot link to MinGW libraries.

### Windows with MSVC and vcpkg

Install Visual Studio 2022 with **Desktop development with C++**, CMake, and
[vcpkg](https://learn.microsoft.com/vcpkg/get_started/get-started). Then run:

```powershell
C:\vcpkg\vcpkg.exe install qtbase eigen3 gmsh --triplet x64-windows
cmake --fresh -S . -B build-msvc -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build-msvc --config Release --parallel
ctest --test-dir build-msvc -C Release --output-on-failure
```

Replace `C:\vcpkg` with the location of your vcpkg checkout. Visual Studio
places the executables under `build-msvc\Release`.

### Windows with MinGW-w64 (MSYS2)

Install or update one consistent MinGW64 environment from PowerShell:

```powershell
& C:\msys64\usr\bin\pacman.exe -S --needed `
  mingw-w64-x86_64-toolchain `
  mingw-w64-x86_64-cmake `
  mingw-w64-x86_64-ninja `
  mingw-w64-x86_64-qt6-base `
  mingw-w64-x86_64-eigen3 `
  mingw-w64-x86_64-gmsh
```

```powershell
$env:Path = "C:\msys64\mingw64\bin;$env:Path"
cmake --fresh -S . -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH=C:/msys64/mingw64
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### macOS with AppleClang and Homebrew

Install the Xcode Command Line Tools and dependencies, then configure with the
Homebrew prefixes visible to CMake:

```bash
xcode-select --install
brew install cmake ninja qt eigen gmsh
cmake --fresh -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(brew --prefix qt);$(brew --prefix)"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The GUI is produced as `build/transmission-line-calculator.app`; the CLI and
tests are regular Mach-O executables in `build/`.

### Linux with GCC (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build qt6-base-dev \
  libeigen3-dev libgmsh-dev
cmake --fresh -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Clang can be selected instead by installing it and setting
`CC=clang CXX=clang++` on the first configure command. Other distributions
use the equivalent development packages.

### Generic CMake build

The same source directory can use another CMake generator. Single-config
generators such as Ninja and Unix Makefiles use `-DCMAKE_BUILD_TYPE=Release`.
Multi-config generators such as Visual Studio and Ninja Multi-Config use
`--config Release` when building and `-C Release` when testing. If dependencies
are in a custom prefix, add `-DCMAKE_PREFIX_PATH=/path/to/prefix`.

## Build outputs

The build creates:

- `transmission-line-calculator` — Qt desktop calculator (`.exe` on Windows or
  `.app` on macOS);
- `transmission-line-calculator-cli` — headless solve/benchmark tool;
- `tl-solver-tests` — numerical regression tests;
- `tl-core` — reusable static solver library.

With a standalone build these are under `build/` (or a configuration
subdirectory for Visual Studio). A repository-root build places them under
`build/TransmissionLineCalculator/`.

CTest exercises all four default templates, the exact ideal-coax TEM solution,
lossy passive branches, analytic coax and projected microstrip metal loss,
mesh refinement, invalid inputs, and an offscreen GUI startup smoke test. It
also compares `n_eff`,
`Zc`, capacitance, vacuum capacitance, and inductance for every 1 mm GUI
default against frozen Python FEM reference values with a 4% relative-error
limit.

## Use

Launch the desktop calculator:

```powershell
.\build\transmission-line-calculator.exe
```

On Linux use `./build/transmission-line-calculator`; on macOS use
`open build/transmission-line-calculator.app`. Add `Release/` after the build
directory for a Visual Studio build.

Choose one of Coaxial, Microstrip, Stripline, or CPW, edit its dimensions, and
select **Calculate FEM**. **Refine x2** halves the effective maximum element
size and resolves the line again. **Display mesh** overlays the native
triangles without recomputing the result. The E/H backgrounds encode field
magnitude; the arrows encode instantaneous transverse direction.

Run all four defaults headlessly:

```powershell
.\build\transmission-line-calculator-cli.exe
```

Select one line, refine it, or benchmark repeated full solves:

```powershell
.\build\transmission-line-calculator-cli.exe --type microstrip --refine 2
.\build\transmission-line-calculator-cli.exe --type coaxial --benchmark 5
.\build\transmission-line-calculator-cli.exe --help
```

The CLI prints node/triangle counts; mesh, solve, and total timings; `n_eff`,
`beta`, `Zc`, and wave impedance; R/L/G/C and vacuum capacitance; and power.

## Units and defaults

The core API uses SI units: metres, hertz, siemens per metre, farads per metre,
henries per metre, ohms, and watts. The GUI converts the following displayed
units to SI:

| Input | GUI unit | Default |
| --- | ---: | ---: |
| Frequency | GHz | 10 |
| Mesh size | mm | 1.00 |
| Metal conductivity | MS/m | blank (ideal PEC) |

Geometry defaults match the Python calculator:

- Coaxial: inner radius 0.5 mm, shield inner radius 1.67 mm, shield thickness
  150 um, relative permittivity 2.1, loss tangent 0.0002.
- Microstrip: trace 3 mm, substrate 1.524 mm, metal 35 um, padding 1,
  relative permittivity 3.55, loss tangent 0.0027.
- Stripline: trace 0.8 mm, ground spacing 1.524 mm, metal 35 um, padding 1,
  relative permittivity 3.55, loss tangent 0.0027.
- CPW: centre conductor 0.6 mm, slot 0.25 mm, each ground 1.5 mm, substrate
  0.8 mm, metal 35 um, padding 1, relative permittivity 3.55, loss tangent
  0.0027.

All dimensions, frequency, relative permittivity, mesh size, padding, and a
provided conductivity must be finite and positive. Loss tangent may be zero
but cannot be negative. Coax shield radius must exceed its inner-conductor
radius; stripline metal thickness must be smaller than the ground spacing.

Results are reported as `Zc`/`Zwave` in ohms, `R'` in ohms/m, `L'` in H/m,
`G'` in S/m, `C'` in F/m, attenuation in 1/m, and power in W.

## Performance and accuracy

Both potential systems reuse the same mesh topology. The native executable
avoids Python interpreter and object-allocation overhead, while the GUI runs
meshing and solving away from its event thread. The result and CLI expose
separate mesh and solve timings; use `--benchmark N` for measurements on the
current machine instead of relying on a fixed speedup claim.

`Refine x2` approximately doubles planar resolution, so triangle count and
memory can grow by roughly four times. Reported engineering values should be
checked by successive refinement until the quantities of interest stop
changing. For microstrip and CPW, also increase domain padding to check the
independent error caused by the remote zero-potential truncation wall.

The automated parity test fixes the GUI's default 1 mm mesh and requires the
five principal extracted quantities for all four templates to remain within
4% of the established Python implementation. This guards native performance
work from silently changing the calculator's engineering results.

This is a frequency-tagged quasi-static TEM/quasi-TEM model. It does not model
radiation, rough conductors, fields inside metal, higher-order modes, or
full-wave dispersion. The conductor correction assumes a smooth,
nonmagnetic good conductor for which a first-order surface impedance is
appropriate.

## Licensing note

This application's MIT-licensed source links to Gmsh, which is distributed
under GPL-2.0-or-later. The source licenses are compatible, but anyone who
redistributes a combined application binary or the Gmsh runtime must also
comply with Gmsh's GPL terms and provide the corresponding notices and source
offer required by that license.

## Windows MinGW install and uninstall

The provided PowerShell installer is specifically for the MSYS2 MinGW64 build.
It configures and builds Release binaries, runs CTest, installs to
`%LOCALAPPDATA%\TransmissionLineCalculator`, deploys Qt's platform plugin, and
copies Gmsh and the required MinGW runtime DLLs:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

Use `-SkipTests` only when the same build has already passed CTest. A custom
MSYS2 prefix or installation directory can be supplied with `-MsysPrefix` and
`-Destination`.

Launch the installed GUI with:

```powershell
& "$env:LOCALAPPDATA\TransmissionLineCalculator\bin\transmission-line-calculator.exe"
```

Remove the default installation with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\uninstall.ps1
```

For safety, the uninstall script only recursively removes a directory whose
final component is exactly `TransmissionLineCalculator`. Repository-local
build output is separate and is not removed by the uninstaller.

## Source layout

- `native/solver.*` — geometry, P1 mesh, two-potential FEM, and result model;
- `native/main_window.*` — Qt calculator workflow and asynchronous solve;
- `native/field_plot.*` — E/H field and mesh rendering;
- `native/cli.cpp` — headless validation and benchmarking;
- `tests/test_solver.cpp` — numerical regression suite;
- `scripts/install.ps1`, `scripts/uninstall.ps1` — Windows deployment helpers.

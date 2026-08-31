# WaveFEM Viewer

WaveFEM Viewer is a native C++20/Qt 6 desktop application for inspecting
WaveFEM schema-v1 HDF5 results. It is independent of the Python solver and
does not import WaveFEM, NumPy, h5py, Tk, or Matplotlib.

The viewer uses a deliberately lazy loading path:

1. opening a file reads only its frequency index and S-parameter records;
2. the selected frequency's large field, mode, and scene arrays are loaded on
   a worker thread;
3. changing frequency loads only the newly selected result;
4. material triangles are rasterized into a cached image, while no more than
   1,200 field arrows are drawn per frame.

This keeps sweep files responsive and avoids loading every frequency's field
arrays into memory at once.

## Features

- Single-result and frequency-sweep HDF5 files.
- Directory picker with an in-window selector for every `.h5`/`.hdf5` file
  in the chosen directory, alongside the single-file picker.
- Indexed modal S-parameter table and sweep plots for dB magnitude, linear
  magnitude, phase, real part, and imaginary part.
- Modal E and H plots with x/y/z/norm and abs/real/imaginary selectors.
- Total, incident, and scattered 2D E/H vector fields.
- Physical plotting convention: z is the horizontal axis and x is the
  vertical axis.
- Every nonzero vector arrow has the same screen length; viridis colour and a
  colorbar carry the original in-plane magnitude.
- Grey dielectric background, yellow PEC, blue PMC, red wave ports, and green
  dashed PML interfaces.
- Mouse-wheel zoom, left-button pan, and double-click reset.
- Native support for HDF5 complex datatypes and older r/i compound-complex
  datasets.

## Requirements

The build requires:

- a C++20 compiler with a complete `<format>` implementation (a current MSVC,
  AppleClang, GCC, or Clang release);
- CMake 3.24 or newer;
- Qt 6.2 or newer (`Widgets` and `Concurrent`);
- HDF5 1.10 or newer.

HDF5 2.x native-complex datasets are supported when building against HDF5
2.x. Builds against HDF5 1.x can read the older `r`/`i` compound-complex and
real-valued datasets. Keep Qt and HDF5 built for the same compiler,
architecture, and runtime as the viewer.

### Windows with MSVC and vcpkg

Install Visual Studio 2022 with **Desktop development with C++**, CMake, and
[vcpkg](https://learn.microsoft.com/vcpkg/get_started/get-started). Then run:

```powershell
C:\vcpkg\vcpkg.exe install qtbase hdf5 --triplet x64-windows
cmake --fresh -S . -B build-msvc -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build-msvc --config Release --parallel
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
  mingw-w64-x86_64-hdf5
```

```powershell
$env:Path = "C:\msys64\mingw64\bin;$env:Path"
cmake --fresh -S . -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH=C:/msys64/mingw64
cmake --build build --parallel
```

### macOS with AppleClang and Homebrew

```bash
xcode-select --install
brew install cmake ninja qt hdf5
cmake --fresh -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(brew --prefix qt);$(brew --prefix)"
cmake --build build --parallel
```

The GUI is produced as `build/wavefem-viewer.app`; the inspect utility is a
regular Mach-O executable in `build/`.

### Linux with GCC (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build qt6-base-dev libhdf5-dev
cmake --fresh -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Ubuntu 24.04's GCC 13 and Qt 6.4 satisfy the source requirements. Clang can be
selected instead by installing it and setting `CC=clang CXX=clang++` on the
first configure command. Other distributions use the equivalent development
packages.

### Generic CMake build

Single-config generators such as Ninja and Unix Makefiles use
`-DCMAKE_BUILD_TYPE=Release`. Multi-config generators such as Visual Studio
and Ninja Multi-Config use `--config Release` when building. If dependencies
are in a custom prefix, add `-DCMAKE_PREFIX_PATH=/path/to/prefix`.

## Build outputs

The build creates:

- `wavefem-viewer` — the Qt GUI (`.exe` on Windows or `.app` on macOS);
- `wavefem-viewer-inspect` — a headless schema/benchmark utility.

With a standalone build these are under `build/` (or a configuration
subdirectory for Visual Studio). A repository-root build places them under
`build/WaveFEMViewer/`.

## Windows MinGW install

The provided PowerShell installer is specifically for the MSYS2 MinGW64 build.
It builds Release binaries, deploys the Qt platform plugin, and copies the
required MinGW/HDF5 runtime DLLs. The default location is
`%LOCALAPPDATA%\WaveFEMViewer`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

Choose another directory when needed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1 `
  -Destination D:\Tools\WaveFEMViewer
```

If the old Python package was previously installed, remove its command first
to avoid a PATH-name collision:

```powershell
python -m pip uninstall wavefem-viewer
```

## Use

Open a file from the GUI:

```powershell
& "$env:LOCALAPPDATA\WaveFEMViewer\bin\wavefem-viewer.exe"
```

For an uninstalled build, use `./build/wavefem-viewer` on Linux or
`open build/wavefem-viewer.app` on macOS. On Windows, run
`build\wavefem-viewer.exe` for MinGW or
`build-msvc\Release\wavefem-viewer.exe` for Visual Studio.

Use **Open directory…** to choose a results folder. The **File** selector is
then populated with every readable `.h5` and `.hdf5` file in that directory;
selecting another entry loads it immediately. **Open HDF5…** remains
available for choosing one file directly. A file supplied on the command
line also populates the selector from its parent directory. Supplying a
directory on the command line scans it and loads its first listed result.

Or pass an HDF5 path directly:

```powershell
& "$env:LOCALAPPDATA\WaveFEMViewer\bin\wavefem-viewer.exe" `
  D:\results\device_sweep.h5
```

The frequency selector changes the active sweep result. S-parameter curves
remain indexed across the whole sweep; only the selected result's full arrays
are resident.

The status bar reports native loading time and the number of field samples,
modes, and material triangles loaded.

From Python, `result.visualize(gui=True)`,
`sweep.visualize_with_gui()`, and `wavefem.launch_viewer(path)` find this
executable in repository build, `PATH`, and installed locations. Set
`WAVEFEM_VIEWER_EXECUTABLE` for an explicit override. Run
`wavefem-inspect-h5 --gui` (or `python examples/inspect_h5.py --gui`) to open
the current directory and choose one of its HDF5 files in this window.

### Headless validation and timing

Use the companion utility to validate a file and measure native HDF5 loading:

```powershell
& "$env:LOCALAPPDATA\WaveFEMViewer\bin\wavefem-viewer-inspect.exe" `
  D:\results\device_sweep.h5
```

Inspect a specific zero-based sweep result:

```powershell
wavefem-viewer-inspect.exe D:\results\device_sweep.h5 4
```

It prints result counts, selected frequency, sample/mode/S-parameter/scene
counts, and separate index/result loading times.

## Uninstall

For the default installation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\uninstall.ps1
```

For a custom location, remove that installation directory directly or pass a
directory named `WaveFEMViewer` to the script. The uninstall script refuses to
recursively remove a directory with any other final name.

Build outputs are not installed state. Remove the repository-local `build`
directory separately if desired.

## Source layout

- `CMakeLists.txt` — portable Qt/HDF5 build and install rules.
- `native/h5_reader.*` — schema-v1 HDF5 index and lazy result loader.
- `native/model.hpp` — in-memory result structures.
- `native/plot_widget.*` — cached QPainter plots, interaction, and overlays.
- `native/main_window.*` — asynchronous GUI and frequency/result management.
- `native/inspect.cpp` — headless validation and loading benchmark.
- `scripts/install.ps1`, `scripts/uninstall.ps1` — Windows deployment helpers.

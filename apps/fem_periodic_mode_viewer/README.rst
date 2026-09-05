FEM Periodic Mode Viewer
========================

Windows x64 / Python 3.12 users can install this app with all solvers using the
`single FDFD release wheel <../../README.md#installation>`_. Launch it with
``python -m fdfd periodic-viewer``; no C++ build is needed. The build instructions
below are for source installations, including Linux and macOS.

FEM Periodic Mode Viewer is a standalone C++20 desktop application for inspecting ``fem-periodic-modes`` HDF5 results. It does not import or link the Python solver, NumPy, h5py, Matplotlib, or the repository's FEM Waveguide Scattering viewer.

Qt 6 and QPainter provide the native 2D triangle view. VTK is optional and adds a tetrahedral 3D mesh view with surface edges, an optional movable plane cut, and decimated E/H vector glyphs. The separate ``fem-periodic-mode-inspect`` executable validates and summarizes a selected case and mode without Qt or a display server.

Large arrays are loaded lazily:

#. opening a file reads only root metadata and ``/index``;

#. selecting a case loads and caches its referenced mesh and material state;

#. selecting a mode reads one HDF5 hyperslab from ``E`` and ``H``;

#. raw FEM coefficients are read only when the inspector is given ``--coefficients``.

The viewer presents stored cell-owned visualization samples. Expanded P1 or canonically oriented Nédélec coefficients remain the scientific representation for exact post-processing.

Features
--------

* Single-result and multi-case frequency-sweep archives.

* Separate 2D and 3D mode panels with their own relevant controls, each opening on a material tab before its field tab. Controls from the other dimension are not shown or disabled in the active panel.

* Complex propagation constant, raw/folded effective index, Bloch multiplier, residual, PML participation, polarization, direction, and normalization metadata.

* Native QPainter 2D triangle plots with E/H, x/y/z, and magnitude/real/imaginary/phase controls.

* Optional VTK tetrahedral 3D surface, movable orthogonal heat-map slice, and bounded sample-point vector glyphs. With slicing off, glyphs span the sampled volume. With slicing on, only the scalar heat map on the exact cut is drawn: the volume surface and vector glyphs are hidden.

* Every 3D viewport includes an XYZ orientation-arrow triad and a labelled scalar colour bar for the active material or field quantity.

* Material tabs derive local ``|n_eff|`` from both diagonal epsilon and mu and use a blue-white-red colour scale with no yellow; yellow remains reserved for PEC/metal boundary geometry.

* VTK arrows use the real phasor vector (the ``t=0`` field) for magnitude, real, and phase coloring, and the imaginary quadrature vector for imaginary coloring; component-wise magnitudes or phases are never used as directions.

* Portable HDF5 compound ``r``/``i`` complex values and, when the viewer is linked against HDF5 2.x, HDF5 native complex values.

* Fixed-length and variable-length UTF-8 HDF5 strings.

* Read-only asynchronous loading, stale-request suppression, and truthful load-error reporting.

* Directory picker and sibling-file selector for readable ``.h5``/``.hdf5`` results, including directory paths supplied on the command line.

* Mouse-wheel zoom, left-button pan, and double-click reset in 2D.

Build and installer commands below run from ``apps/fem_periodic_mode_viewer``.

Configure options
-----------------

``FEM_PERIODIC_MODE_VIEWER_WITH_VTK`` is a string cache option:

* ``AUTO`` (default) enables 3D when all requested VTK modules are found;

* ``ON`` requires VTK and makes a missing component a configuration error;

* ``OFF`` builds the Qt/QPainter viewer without a VTK viewport.

The VTK package must itself be built against Qt 6. A build without VTK still indexes and validates both 2D and 3D archives and renders 2D fields; opening a 3D result displays a clear unavailable-view message.

HDF5 schema accepted by the reader
----------------------------------

Schema major version 1 has the following contract. Brackets mark optional, additive data. Object names are zero-padded six-digit indices.

.. code-block:: text

   /
     attrs:
       format = "cem-fem-results"
       schema = "1.0"
       solver_family = "periodic_modes"
       units = "SI"
       dimension = 2 | 3 | 0  # 0: cases with mixed mesh dimensions
       result_kind = "modes" | "sweep"
       schema_major = 1
       schema_minor >= 0
       kind = "single" | "sweep"
       case_count
       time_convention = "exp(+i*omega*t)"
       field_representation = "periodic-envelope"
       [producer], [producer_version]

     /index
       frequency_hz[C]
       mode_offsets[C + 1]
       mesh_index[C]
       material_state_index[C]
       gamma_per_m[K], neff[K], neff_folded[K], bloch_multiplier[K]
       alpha_per_m[K], beta_per_m[K], beta_folded_per_m[K]
       residual[K], [gauss_residual[K]], [gauss_available[K]], pml_fraction[K]
       polarization[K], direction[K], normalization[K]

     /meshes/000000
       attrs: dimension = 2 | 3
              topology = "triangle3" | "tetra4"
              periodic_axis = "z"
              period_m, reference_z_m
       points[Np, 3]
       cells[Nc, 3 | 4]
       [cell_region_id[Nc]]
       /samples/points[Ns, 3]
       /samples/owner_cell[Ns]
       [/boundary/facets[Nf, 2 | 3]]
       [/boundary/tag[Nf]]
       [/periodic/node_pairs[Npair, 2]]       # slave, master
       [/periodic/affine[4, 4]]
       3D: edge_nodes[Ne, 2]                  # ascending global node indices
           cell_edges[Nc, 6]                  # local (01),(12),(02),(03),(13),(23)
           cell_edge_sign[Nc, 6]              # local direction vs ascending global
           [/periodic/edge_pairs[Nepair, 2]]  # slave edge, master edge
           [/periodic/edge_sign[Nepair]]      # mapped slave vs master orientation

     /material_states/000000
       attrs: mesh_index
       epsilon_r[Nc, 3]
       mu_r[Nc, 3]
       pml_fraction[Nc]

     /cases/000000
       [attrs: frequency_hz, omega, k0, mesh_index,
               material_state_index, mode_count, backend]
       /coefficients
         attrs: space, full_expanded = 1
         values[M, Ndof]
         primary_unknown[M]                   # or scalar text attribute
       [/mode_metadata/has_power[M]]          # integer 0 or 1
       [/mode_metadata/power[M]]              # paired complex128 values
       /visualization/E[M, Ns, 3]
       /visualization/H[M, Ns, 3]

``C`` is the number of cases, ``K`` is the total number of modes across all cases, and ``M`` is the mode count in one case. ``mode_offsets`` maps each case to a contiguous range in the flattened ``/index`` mode arrays. The reader accepts complex values as an HDF5 compound with members named exactly ``r`` and ``i``. It also accepts the HDF5 2.x native complex type when the viewer itself is linked against HDF5 2.x.

Dataset shapes, finite values, material and visualization references, connectivity, canonical edge orientation, and orientation signs are checked before use. Unknown schema major versions and incompatible field or phasor conventions are rejected. Additive minor-version members are ignored. When present, ``mode_metadata/has_power`` and ``power`` are a paired optional contract: both must be one-dimensional with exactly ``M`` entries.

Requirements
------------

Always required:

* CMake 3.24 or newer;

* a C++20 compiler with a complete ``<format>`` implementation;

* Qt 6.2 or newer with ``Widgets`` and ``Concurrent``;

* HDF5 1.10 or newer.

Optional 3D support requires VTK 9.2 or newer with ``GUISupportQt``, OpenGL2, and a Qt 6 build. Qt, HDF5, VTK, and the compiler must use the same architecture and runtime.

Standalone commands in the following sections run from the repository root.

Windows: MSYS2 MinGW64
----------------------

Install one consistent MinGW64 toolchain. The extra header-only packages are dependencies of the current MSYS2 VTK package:

.. code-block:: powershell

   & C:\msys64\usr\bin\pacman.exe -S --needed `
     mingw-w64-x86_64-toolchain `
     mingw-w64-x86_64-cmake `
     mingw-w64-x86_64-ninja `
     mingw-w64-x86_64-qt6-base `
     mingw-w64-x86_64-hdf5 `
     mingw-w64-x86_64-vtk `
     mingw-w64-x86_64-nlohmann-json `
     mingw-w64-x86_64-fast_float `
     mingw-w64-x86_64-utf8cpp `
     mingw-w64-x86_64-exprtk

Build and test from PowerShell:

.. code-block:: powershell

   $env:Path = "C:\msys64\mingw64\bin;$env:Path"
   cmake --fresh -S apps/fem_periodic_mode_viewer -B outputs/build-periodic-mingw -G Ninja `
     -DCMAKE_BUILD_TYPE=Release `
     -DCMAKE_PREFIX_PATH=C:/msys64/mingw64 `
     -DFEM_PERIODIC_MODE_VIEWER_WITH_VTK=AUTO
   cmake --build outputs/build-periodic-mingw --parallel
   ctest --test-dir outputs/build-periodic-mingw --output-on-failure

The provided installer builds, installs, runs ``windeployqt``, and copies MinGW/HDF5/VTK runtime DLLs into ``%LOCALAPPDATA%\FEMPeriodicModeViewer``:

.. code-block:: powershell

   powershell -ExecutionPolicy Bypass -File .\apps\fem_periodic_mode_viewer\scripts\install.ps1

This bundle is a local development convenience. Before redistributing it, review the licenses and deployment obligations of the bundled Qt, HDF5, VTK, and toolchain runtime libraries and include their required third-party notices.

Pass ``-WithoutVtk`` for a 2D-only build or ``-Destination D:\Tools\FEMPeriodicModeViewer`` for another destination. Remove an installation with:

.. code-block:: powershell

   powershell -ExecutionPolicy Bypass -File .\apps\fem_periodic_mode_viewer\scripts\uninstall.ps1

The uninstaller refuses to recursively remove a directory whose final name is not ``FEMPeriodicModeViewer``.

Windows: MSVC and vcpkg
-----------------------

Follow the `root Windows installation walkthrough <../../README.md#windows-msvc-and-vcpkg-step-by-step>`_
for downloading MSVC and vcpkg, installing dependencies, compiling, testing,
deploying DLLs and Qt plugins, and connecting the native viewers to Python.
The walkthrough's commands run from the repository root and install all three
applications under ``%LOCALAPPDATA%\FDFD``. Its CMake options also allow you to
select individual apps. The bundled ``scripts/install.ps1`` is for MSYS2/MinGW;
use the root walkthrough's CMake install/deployment commands for MSVC.

macOS: AppleClang and Homebrew
------------------------------

.. code-block:: bash

   xcode-select --install
   brew install cmake ninja qt hdf5 vtk
   cmake --fresh -S apps/fem_periodic_mode_viewer -B outputs/build-periodic-macos -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_PREFIX_PATH="$(brew --prefix qt);$(brew --prefix hdf5);$(brew --prefix vtk)" \
     -DFEM_PERIODIC_MODE_VIEWER_WITH_VTK=AUTO
   cmake --build outputs/build-periodic-macos --parallel
   ctest --test-dir outputs/build-periodic-macos --output-on-failure

The GUI target is emitted as a macOS application bundle; the inspector remains a normal command-line executable. A recent Xcode/libc++ is required for ``std::format``.

Linux: GCC or Clang
-------------------

On Ubuntu/Debian, a typical 2D-only dependency set is:

.. code-block:: bash

   sudo apt update
   sudo apt install build-essential cmake ninja-build qt6-base-dev libhdf5-dev

For 3D, additionally install the distribution's VTK development package with Qt 6 support. Package names vary by release; commonly they include ``libvtk9-dev`` and ``libvtk9-qt-dev``. Then configure with ``AUTO`` or ``ON``:

.. code-block:: bash

   cmake --fresh -S apps/fem_periodic_mode_viewer -B outputs/build-periodic-linux -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DFEM_PERIODIC_MODE_VIEWER_WITH_VTK=AUTO
   cmake --build outputs/build-periodic-linux --parallel
   ctest --test-dir outputs/build-periodic-linux --output-on-failure

Select Clang on the first configure with ``-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++``. Use a recent GCC or Clang/libstdc++ combination that implements ``std::format``.

Use and install
---------------

Launch the GUI with or without an initial file:

.. code-block:: text

   fem-periodic-mode-viewer [result.h5|results-directory]

Use **Open directory…** to populate the in-window **File** selector with every readable ``.h5`` and ``.hdf5`` file in that directory. Opening one file directly populates the same selector from its parent directory. Supplying a directory on the command line scans it and loads the first valid result in name order; an invalid earlier file does not hide valid siblings from the selector.

The complete Windows FDFD wheel includes this viewer. Python ``result.show()``
finds the bundled executable before repository builds, ``PATH``, and local
installations. Set ``FEM_PERIODIC_MODE_VIEWER_EXECUTABLE`` to an absolute
executable path to override that search.

Inspect the first case and mode:

.. code-block:: text

   fem-periodic-mode-inspect result.h5

Inspect another zero-based case and mode and read its coefficient hyperslab:

.. code-block:: text

   fem-periodic-mode-inspect result.h5 3 1 --coefficients

The inspector prints machine-readable ``key=value`` lines and returns nonzero for invalid files. A generic CMake install is also available:

.. code-block:: bash

   cmake --install outputs/build-periodic-linux --prefix install

Tests
-----

The CTest suite creates all native fixtures through the HDF5 C API; it does not require Python or FEM Waveguide Scattering. It covers:

* fixed and variable-length UTF-8 strings;

* compound complex encodings, plus native complex when built with HDF5 2.x;

* single archives and shared-object two-case sweeps;

* index, mesh, material, selected-mode field, and coefficient hyperslabs;

* empty optional topology arrays;

* rejection of unsupported schema versions, array-valued scalar attributes, inconsistent cross-object references, and malformed optional mode metadata;

* truthful headless inspector and asynchronous GUI failure status;

* offscreen directory discovery, invalid-first fallback, and first-result loading;

* native removal of Python-created temporary visualization archives on normal and immediate-close exits, after outstanding asynchronous reads finish;

* 3D archive reading in both VTK and no-VTK builds;

* Qt offscreen 2D startup and, when enabled, VTK offscreen 3D rendering.

The repository's Python writer was also checked end-to-end with the native inspector and offscreen GUI; its compound-complex single and sweep archives load without conversion.

Source layout
-------------

* ``native/h5_reader.*`` — schema validation and one-mode hyperslab reader.

* ``native/field_plot_2d.*`` — native QPainter triangle renderer.

* ``native/vtk_field_view.*`` — optional VTK tetrahedral viewport.

* ``native/main_window.*`` — asynchronous UI, cache, controls, and metadata.

* ``native/inspect.cpp`` — headless validator and timing utility.

* ``tests/`` — standalone HDF5 C-API fixtures and reader tests.

* ``scripts/`` — MinGW64 install and guarded uninstall helpers.

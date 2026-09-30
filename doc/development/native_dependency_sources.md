# Native dependency sources on Windows

Windows builds use x64 MSVC and vcpkg in `C:\opt\vcpkg`. MinGW/MSYS2 builds are unsupported. The root [README](../../README.md) lists the required vcpkg features and the command to build all applications.

vcpkg records dependency versions and source URLs in each installed port's `vcpkg.json` and `portfile.cmake`. The project-local [Gmsh overlay](../../vcpkg-ports/README.rst) enables meshing, Eigen, and OpenCASCADE compatibility; pass `--overlay-ports=./vcpkg-ports` when installing dependencies. Cached upstream archives are in `C:\opt\vcpkg\downloads`; installed binaries and headers are in `C:\opt\vcpkg\installed\x64-windows`.

To rebuild the native dependencies, run the vcpkg install command in the root README from the repository root. Then enter the MSVC environment with `scripts/setup_msvc_windows.ps1` and configure a fresh CMake build. The packaging script `scripts/package_native_windows.py` records the installed dependency recipes and licenses in the release bundle, and preserves cached source archives with verifiable hashes in `SOURCE_INDEX.md`. Some installed vcpkg SPDX records contain unexpanded source variables rather than a usable hash; the index flags those entries explicitly. `build-manifest.json` identifies the toolchain and vcpkg revision.

Application source and packaging code are in the [FDFD repository](https://github.com/SolverNotConverging/FDFD); release notes identify the source commit for each wheel.

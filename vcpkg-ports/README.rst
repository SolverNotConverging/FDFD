FDFD vcpkg overlays
===================

The ``gmsh`` port is based on vcpkg's Gmsh 4.15.2 port at revision
``aa40adda5352e87655b8583cfb2451d5e9e276fd``. It enables
``ENABLE_MESH`` because the transmission-line calculator calls Gmsh's native
mesh generator. The upstream port disables meshing. The overlay also builds
Gmsh with Eigen for the matrix inversion used by that mesh generator and with
C++17, as required by OpenCASCADE 8's headers. Other upstream options
and source checksums are preserved. The overlay updates Gmsh's OpenCASCADE 8
header and type names to the corresponding library aliases and enables MSVC's
``/bigobj`` for Eigen-heavy translation units. Use
``--overlay-ports=./vcpkg-ports`` with
the Windows dependency installation command in the root README.

The upstream port files are covered by ``LICENSE.vcpkg`` in this directory;
Gmsh's own license is installed by vcpkg with the library.

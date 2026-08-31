FEM Periodic Solver
===================

``FEM_Periodic_Solver`` solves for the complex propagation constant of one
periodic electromagnetic cell at a fixed real frequency.  It uses the
``exp(+j omega t)`` convention and represents the physical field as a strictly
periodic finite-element envelope times ``exp(-j k0 neff z)``.

The two-dimensional backend assembles independent scalar P1 TE and TM
quadratic eigenproblems on a conforming Gmsh mesh.  Periodic nodes are reduced
by a sparse equality prolongation before the pencil is linearized.  Small
systems use homogeneous dense QZ; large systems use the shared
``periodic-eigensolver`` refined shift-and-invert Arnoldi implementation.
The three-dimensional backend uses first-kind ``ElementTetN1`` Nedelec edge
fields, signed periodic edge constraints, and a weak transformed Gauss-law
filter.  Its magnetic field is reconstructed from the shifted curl.

Installation
------------

From the repository root, install the shared eigensolver followed by this
package.  A source installation may use the Python Arnoldi fallback when a
native compiler is unavailable; published ``periodic-eigensolver`` wheels
include the Cython extension.

.. code-block:: bash

   python -m pip install -e ./periodic_eigensolver
   python -m pip install -e ./FEM_Periodic_Solver

Quick start
-----------

.. code-block:: python

   from FEM_Periodic_Solver import PeriodicModeSolver2D

   solver = PeriodicModeSolver2D(
       frequency=20e9,
       x_range=(0.0, 10e-3),
       z_range=(0.0, 8e-3),
       num_modes=4,
       neff_guess=1.5,
       polarization="TM",
       boundary="pec",
       eigensolver="auto",
   )
   solver.add_rectangle(
       epsilon=10.2, mu=1.0,
       x_range=(0.0, 1.27e-3), z_range=(0.0, 8e-3),
       name="grounded_dielectric_slab",
   )
   solver.add_pec(
       x_range=(1.27e-3, 1.32e-3),
       z_range=(1.0e-3, 2.0e-3),
       name="top_pec_perturbation",
   )
   solver.add_pml(2.5e-3, direction="x+")
   solver.discretize(max_element_size=0.65e-3)
   modes = solver.solve(direction="all", max_pml_fraction=None)
   print(modes.neff)
   solver.visualize(1, component="Hy", quantity="real")
   solver.visualize_with_gui()  # open every solved mode in the native viewer

The complete runnable version is
``examples/leaky_wave_antenna_2d.py``.  Static ``visualize`` calls show their
Matplotlib window by default; use ``show=False`` only when embedding the
returned figure and axes.

Geometry is added before ``discretize`` with ``add_rectangle``, ``add_circle``,
``add_polygon``, or ``add_triangle``.  Material and conductor objects that
touch the periodic seam must have identical split topology on both seam faces;
Gmsh rejects unmatched cells.  PML is transverse only (``x-``, ``x+``, or
``x``); a PML in the propagation direction is intentionally invalid.

The result keeps both the raw branch nearest ``neff_guess`` and the folded
first-Brillouin-zone values through ``mode.folded_beta`` and
``mode.folded_neff``.  It also exposes ``gamma``, the Bloch multiplier,
original-QEP residual, longitudinal power, normalization, and PML energy
fraction.  ``solve(max_pml_fraction=0.5)`` rejects PML-dominated candidates by
default; pass ``None`` to disable this filter.  Three-dimensional
``gauss_residual`` is the squared, normalized weak Gauss-defect energy, so the
default ``divergence_tolerance=1e-6`` applies to that energy ratio rather than
its square-root amplitude.

Three-dimensional workflow
--------------------------

.. code-block:: python

   from FEM_Periodic_Solver import Box, PeriodicModeSolver3D

   solver = PeriodicModeSolver3D(
       12e9,
       x_range=(0.0, 22.86e-3),
       y_range=(0.0, 10.16e-3),
       z_range=(0.0, 8e-3),
       num_modes=2,
       neff_guess=0.7,
       boundary="pec",
       eigensolver="dense",
   )
   solver.add_pec(
       Box((0.0, 4e-3), (0.0, 10.16e-3), (3.6e-3, 4.4e-3)),
       name="left_iris",
   )
   solver.add_pec(
       Box((18.86e-3, 22.86e-3), (0.0, 10.16e-3), (3.6e-3, 4.4e-3)),
       name="right_iris",
   )
   solver.discretize(max_element_size=4e-3)
   modes = solver.solve(direction="all")
   solver.visualize(1, component="E", quantity="abs")
   solver.visualize_with_gui()

This is the periodic iris-loaded WR-90 filter cell from
``examples/iris_loaded_waveguide_filter_3d.py`` rather than a uniform-guide
fixture.

``add_cylinder`` and ``add_sphere`` provide the other v1 volume primitives.
PEC and PMC objects are explicit surfaces/volumes constructed before meshing.
Periodic objects must already be split so that the ``z-`` and ``z+`` traces
match; imported meshes and automatic seam wrapping are intentionally outside
v1.

HDF5 results and native viewer
------------------------------

HDF5 is the only persistent result format for this FEM package.  The versioned
``fem-periodic-modes`` schema stores an eagerly readable index and deduplicated
meshes/material states, while fields and coefficients remain mode-chunked.

.. code-block:: python

   from FEM_Periodic_Solver import (
       load_periodic_h5, open_periodic_h5, save_periodic_sweep_h5,
       validate_periodic_h5, launch_viewer,
   )

   modes.save_h5("one-case.h5")
   save_periodic_sweep_h5([modes_at_f1, modes_at_f2], "sweep.h5")
   validate_periodic_h5("sweep.h5", deep=True)
   with open_periodic_h5("sweep.h5") as archive:
       one_mode = archive.load_case(1, modes=0)
   restored = load_periodic_h5("one-case.h5")
   launch_viewer("one-case.h5")

``launch_viewer`` first searches repository CMake build directories (including
Release, RelWithDebInfo, Debug, and MinSizeRel layouts), then ``PATH`` and the
default local Windows installation.  Standard macOS ``.app`` bundles are
recognized in both single- and multi-configuration builds.
It accepts either one archive or a directory for the viewer's HDF5 selector;
omitting the path selects the current directory.  Set
``FEM_PERIODIC_VIEWER_EXECUTABLE`` to an absolute executable path to override
that search.
For MinGW build-tree executables, Python derives the matching DLL and Qt plugin
paths from ``CMakeCache.txt`` before starting the child process, so launching
from an IDE does not depend on the IDE inheriting an MSYS2 ``PATH``.  An early
native-loader exit raises ``PersistenceError`` instead of disappearing
silently.
``solver.visualize(...)`` always creates a Matplotlib figure. The separate,
zero-argument ``solver.visualize_with_gui()`` writes all currently available
modes to an implicit temporary archive, launches the native viewer, and
removes the archive after the window closes even if the launching Python
process has already exited.

The native viewer has separate 2D and 3D mode panels with dimension-specific
controls. Each opens on its material tab before the field tab. In 3D the slice
is optional: with the slice disabled, vector glyphs are shown throughout the
sampled volume; with it enabled, only a scalar heat map on the exact cut is
drawn. The off-plane surface and all vector glyphs are hidden. Each 3D viewport
also provides an XYZ orientation triad and a labelled colour bar.

Gmsh sizing is material- and conductor-aware. The requested maximum element
size is retained in low-index background regions, while high-index regions are
locally reduced in proportion to their refractive-index scale. Internal PEC
perturbations receive an additional smooth distance refinement. Explicit
``refine`` regions remain available and take the finest requested size.

The Python inspector keeps a headless summary mode and can launch the same
directory-first GUI:

.. code-block:: bash

   python -m FEM_Periodic_Solver.inspect_h5 periodic-guide.h5 --deep
   python -m FEM_Periodic_Solver.inspect_h5 --gui .

``--deep`` applies only to headless validation and is rejected with ``--gui``.

Writes use a sibling temporary file followed by atomic replacement.  Complex
arrays use HDF5 compound ``{r,i}`` complex128 storage; readers linked through
HDF5 2 also accept its native complex encoding.  Field chunks are mode-first, gzip level 4,
shuffled, and Fletcher32 protected.  See ``../FEMPeriodicViewer/README.md`` for
portable MinGW, MSVC, AppleClang, and GCC builds of the standalone C++20
viewer and ``fem-periodic-inspect``.

Supported scope
---------------

Materials may be complex isotropic values or diagonal anisotropic triples.
Transverse PML supports ``x-``/``x+`` in 2D and ``x-``/``x+``/``y-``/``y+``
in 3D.  The package rejects propagation-axis PML, off-diagonal tensors, SIBC,
unmatched seam topology, and seam-crossing objects without an explicit split.
The v1 solver is fixed-frequency, single-``z``-period and workstation-local;
band diagrams, multi-axis periodicity, PETSc/SLEPc, higher-order tetrahedra,
and imported meshes are deferred.

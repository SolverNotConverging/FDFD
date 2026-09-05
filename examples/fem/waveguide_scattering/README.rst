FEM waveguide scattering examples
=================================

Install the packages first; see `setup <../../README.rst>`_.
The `user guide <../../../doc/solvers/fem/waveguide_scattering/guide.rst>`_ and
`public API <../../../doc/solvers/fem/waveguide_scattering/API_REFERENCE.rst>`_ explain supported controls.

The ``fem-waveguide-scattering-viewer`` native application is required for ``show()``.
These examples use the ``mesh / solve / show`` workflow. Geometry and material
configuration precede meshing; the typed result is returned by ``solve()``.

Recommended order
-----------------

Runtime depends on hardware and mesh size. Single solves are the starting point;
dispersion and band-structure scripts perform many eigenproblems and can take
substantially longer. 3D cases also require more memory.

1. `uniform_waveguide_2d.py <uniform_waveguide_2d.py>`_ — Port modes and transmission through a uniform guide. Single solve.
2. `dielectric_insert_2d.py <dielectric_insert_2d.py>`_ — Reflection, transmission, and power balance for a weak insert. Single solve.
3. `dielectric_insert_2d_frequency_sweep.py <dielectric_insert_2d_frequency_sweep.py>`_ — A frequency sweep saved as a multi-case HDF5 archive. Frequency sweep.
4. `slab_waveguide_2d_oblique_incidence.py <slab_waveguide_2d_oblique_incidence.py>`_ — Oblique incidence with nonzero invariant-direction wavenumber. Single solve.
5. `grounded_slab_slot_2d.py <grounded_slab_slot_2d.py>`_ — A PEC slot in a grounded slab. Single solve.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``outputs/examples/fem/waveguide_scattering/<example>/`` in the checkout.

FDFD scattering examples
========================

The scripts run directly from top to bottom. Shared materials and shapes are
imported from the solver package alongside the solver classes.

Install the Python dependencies first; see `setup <../../README.md>`_.
The `user guide <../docs/guide.rst>`_ and
`public API <../docs/API_REFERENCE.rst>`_ explain supported controls.

A Matplotlib GUI backend is required to display interactive figures.
These examples retain the FDFD-specific workflow: configure the grid and
materials, solve, then inspect stored fields using the matching viewer.

Recommended order
-----------------

Runtime depends on hardware and mesh size. Single solves are the starting point;
dispersion and band-structure scripts perform many eigenproblems and can take
substantially longer. 3D cases also require more memory.

1. `dielectric_cylinder_2d.py <dielectric_cylinder_2d.py>`_ — Scattering from a dielectric cylinder. Single solve.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``fdfd_scattering/outputs/examples/<example>/`` in the repository.

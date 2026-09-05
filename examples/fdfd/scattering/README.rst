FDFD scattering examples
========================

Install the packages first; see `setup <../../README.rst>`_.
The `user guide <../../../doc/solvers/fdfd/scattering/guide.rst>`_ and
`public API <../../../doc/solvers/fdfd/scattering/API_REFERENCE.rst>`_ explain supported controls.

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
``outputs/examples/fdfd/scattering/<example>/`` in the checkout.

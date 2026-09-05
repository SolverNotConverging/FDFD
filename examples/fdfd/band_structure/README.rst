FDFD band structure examples
============================

Install the packages first; see `setup <../../README.rst>`_.
The `user guide <../../../doc/solvers/fdfd/band_structure/guide.rst>`_ and
`public API <../../../doc/solvers/fdfd/band_structure/API_REFERENCE.rst>`_ explain supported controls.

A Matplotlib GUI backend is required to display interactive figures.
These examples retain the FDFD-specific workflow: configure the grid and
materials, solve, then inspect stored fields using the matching viewer.

Recommended order
-----------------

Runtime depends on hardware and mesh size. Single solves are the starting point;
dispersion and band-structure scripts perform many eigenproblems and can take
substantially longer. 3D cases also require more memory.

1. `square_lattice_2d.py <square_lattice_2d.py>`_ — TE/TM bands along a square-lattice Bloch path. Bloch-path sweep.
2. `rectangular_unit_cell_2d.py <rectangular_unit_cell_2d.py>`_ — Bands of a rectangular unit cell. Bloch-path sweep.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``outputs/examples/fdfd/band_structure/<example>/`` in the checkout.

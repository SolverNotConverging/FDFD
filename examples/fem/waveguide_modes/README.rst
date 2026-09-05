FEM waveguide modes examples
============================

Install the packages first; see `setup <../../README.rst>`_.
The `user guide <../../../doc/solvers/fem/waveguide_modes/guide.rst>`_ and
`public API <../../../doc/solvers/fem/waveguide_modes/API_REFERENCE.rst>`_ explain supported controls.

A Matplotlib GUI backend is required to display interactive figures.
These examples use the ``mesh / solve / show`` workflow. Geometry and material
configuration precede meshing; the typed result is returned by ``solve()``.

Recommended order
-----------------

Runtime depends on hardware and mesh size. Single solves are the starting point;
dispersion and band-structure scripts perform many eigenproblems and can take
substantially longer. 3D cases also require more memory.

1. `parallel_plate_waveguide_1d.py <parallel_plate_waveguide_1d.py>`_ — 1D modes and analytic cutoff checks. Single solve.
2. `rectangular_waveguide_2d.py <rectangular_waveguide_2d.py>`_ — Second-order vector elements and the TE10 cutoff. Single solve.
3. `dielectric_slab_1d.py <dielectric_slab_1d.py>`_ — Guided modes in a dielectric slab. Single solve.
4. `ridge_waveguide_2d.py <ridge_waveguide_2d.py>`_ — A ridge waveguide cross section. Single solve.
5. `microstrip_2d_surface_impedance.py <microstrip_2d_surface_impedance.py>`_ — A copper microstrip with a surface-impedance boundary. Single solve.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``outputs/examples/fem/waveguide_modes/<example>/`` in the checkout.

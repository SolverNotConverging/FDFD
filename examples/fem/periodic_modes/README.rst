FEM periodic modes examples
===========================

Install the packages first; see `setup <../../README.rst>`_.
The `user guide <../../../doc/solvers/fem/periodic_modes/guide.rst>`_ and
`public API <../../../doc/solvers/fem/periodic_modes/API_REFERENCE.rst>`_ explain supported controls.

The ``fem-periodic-mode-viewer`` native application is required for ``show()``.
These examples use the ``mesh / solve / show`` workflow. Geometry and material
configuration precede meshing; the typed result is returned by ``solve()``.

Recommended order
-----------------

Runtime depends on hardware and mesh size. Single solves are the starting point;
dispersion and band-structure scripts perform many eigenproblems and can take
substantially longer. 3D cases also require more memory.

1. `uniform_cell_2d.py <uniform_cell_2d.py>`_ — TEM effective index in a uniform 2D cell. Single solve.
2. `uniform_cell_3d.py <uniform_cell_3d.py>`_ — TE10 effective index in a uniform 3D cell. Single solve.
3. `leaky_wave_antenna_2d.py <leaky_wave_antenna_2d.py>`_ — A leaky-wave cell with an outgoing PML. Single solve.
4. `iris_loaded_waveguide_filter_3d.py <iris_loaded_waveguide_filter_3d.py>`_ — An iris-loaded rectangular waveguide cell. Single solve.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``outputs/examples/fem/periodic_modes/<example>/`` in the checkout.

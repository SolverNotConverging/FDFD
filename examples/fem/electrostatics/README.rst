FEM electrostatics examples
===========================

Install the packages first; see `setup <../../README.rst>`_.
The `user guide <../../../doc/solvers/fem/electrostatics/guide.rst>`_ and
`public API <../../../doc/solvers/fem/electrostatics/API_REFERENCE.rst>`_ explain supported controls.

A Matplotlib GUI backend is required to display interactive figures.
These examples use the ``mesh / solve / show`` workflow. Geometry and material
configuration precede meshing; the typed result is returned by ``solve()``.

Recommended order
-----------------

Runtime depends on hardware and mesh size. Single solves are the starting point;
dispersion and band-structure scripts perform many eigenproblems and can take
substantially longer. 3D cases also require more memory.

1. `parallel_plate_capacitor_1d.py <parallel_plate_capacitor_1d.py>`_ — Electrodes and a dielectric inclusion in a 1D capacitor. Single solve.
2. `layered_capacitor_1d.py <layered_capacitor_1d.py>`_ — Analytic energy for two dielectric layers. Single solve.
3. `parallel_plate_2d_space_charge.py <parallel_plate_2d_space_charge.py>`_ — Uniform space charge and an analytic potential check. Single solve.
4. `embedded_electrode_2d_anisotropic.py <embedded_electrode_2d_anisotropic.py>`_ — An embedded electrode and anisotropic dielectric. Single solve.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``outputs/examples/fem/electrostatics/<example>/`` in the checkout.

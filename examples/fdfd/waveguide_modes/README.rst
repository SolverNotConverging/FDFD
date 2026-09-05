FDFD waveguide modes examples
=============================

Install the packages first; see `setup <../../README.rst>`_.
The `user guide <../../../doc/solvers/fdfd/waveguide_modes/guide.rst>`_ and
`public API <../../../doc/solvers/fdfd/waveguide_modes/API_REFERENCE.rst>`_ explain supported controls.

A Matplotlib GUI backend is required to display interactive figures.
These examples retain the FDFD-specific workflow: configure the grid and
materials, solve, then inspect stored fields using the matching viewer.

Recommended order
-----------------

Runtime depends on hardware and mesh size. Single solves are the starting point;
dispersion and band-structure scripts perform many eigenproblems and can take
substantially longer. 3D cases also require more memory.

1. `parallel_plate_waveguide_1d.py <parallel_plate_waveguide_1d.py>`_ — TE/TM modes between parallel plates. Single solve.
2. `grounded_slab_1d.py <grounded_slab_1d.py>`_ — A grounded dielectric slab. Single solve.
3. `rectangular_waveguide_2d.py <rectangular_waveguide_2d.py>`_ — Modes in a rectangular metal waveguide. Single solve.
4. `circular_dielectric_waveguide_2d.py <circular_dielectric_waveguide_2d.py>`_ — A circular dielectric core. Single solve.
5. `ridge_dielectric_waveguide_2d.py <ridge_dielectric_waveguide_2d.py>`_ — A dielectric ridge cross section. Single solve.
6. `microstrip_2d.py <microstrip_2d.py>`_ — A microstrip with dielectric loss. Single solve.
7. `layered_waveguide_1d_dispersion.py <layered_waveguide_1d_dispersion.py>`_ — Frequency-dependent TE/TM propagation and attenuation. Frequency sweep.
8. `dielectric_waveguide_2d_dispersion.py <dielectric_waveguide_2d_dispersion.py>`_ — Dispersion of a rectangular dielectric core. Frequency sweep.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``outputs/examples/fdfd/waveguide_modes/<example>/`` in the checkout.

Postprocessing
--------------

Run the producing example first. These scripts accept an optional input path
and otherwise load from its standard output directory; use ``--help`` for usage.
Plots are saved beside their source data.

* `plot_dispersion_1d.py <postprocessing/plot_dispersion_1d.py>`_
* `plot_dispersion_2d.py <postprocessing/plot_dispersion_2d.py>`_

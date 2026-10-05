FDFD waveguide modes examples
=============================

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

1. `1d_parallel_plate_waveguide.py <1d_parallel_plate_waveguide.py>`_ — TE/TM modes between parallel plates. Single solve.
2. `1d_grounded_slab.py <1d_grounded_slab.py>`_ — A grounded dielectric slab. Single solve.
3. `2d_rectangular_waveguide.py <2d_rectangular_waveguide.py>`_ — Modes in a rectangular metal waveguide. Single solve.
4. `2d_circular_dielectric_waveguide.py <2d_circular_dielectric_waveguide.py>`_ — A circular dielectric core. Single solve.
5. `2d_ridge_dielectric_waveguide.py <2d_ridge_dielectric_waveguide.py>`_ — A dielectric ridge cross section. Single solve.
6. `2d_microstrip.py <2d_microstrip.py>`_ — A microstrip with dielectric loss. Single solve.
7. `1d_layered_waveguide_dispersion.py <1d_layered_waveguide_dispersion.py>`_ — Frequency-dependent TE/TM propagation and attenuation. Frequency sweep.
8. `2d_dielectric_waveguide_dispersion.py <2d_dielectric_waveguide_dispersion.py>`_ — Dispersion of a rectangular dielectric core. Frequency sweep.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``fdfd_waveguide_modes/outputs/examples/<example>/`` in the repository.

Postprocessing
--------------

Run the producing example first. These scripts accept an optional input path
and otherwise load from its standard output directory; use ``--help`` for usage.
Plots are saved beside their source data.

* `1d_plot_dispersion.py <postprocessing/1d_plot_dispersion.py>`_
* `2d_plot_dispersion.py <postprocessing/2d_plot_dispersion.py>`_

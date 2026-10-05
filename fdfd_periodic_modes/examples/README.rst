FDFD periodic modes examples
============================

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

1. `2d_surface_wave_antenna.py <2d_surface_wave_antenna.py>`_ — A dielectric-loaded periodic surface-wave cell. Single solve.
2. `3d_image_guide_leaky_wave_antenna.py <3d_image_guide_leaky_wave_antenna.py>`_ — A 3D image-guide cell with an outgoing PML. Single solve.
3. `2d_surface_wave_antenna_dispersion.py <2d_surface_wave_antenna_dispersion.py>`_ — A 2D periodic frequency sweep. Frequency sweep.
4. `3d_image_guide_leaky_wave_antenna_dispersion.py <3d_image_guide_leaky_wave_antenna_dispersion.py>`_ — A 3D periodic frequency sweep. Frequency sweep.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``fdfd_periodic_modes/outputs/examples/<example>/`` in the repository.

Postprocessing
--------------

Run the producing example first. These scripts accept an optional input path
and otherwise load from its standard output directory; use ``--help`` for usage.
Plots are saved beside their source data.

* `3d_inspect_results.py <postprocessing/3d_inspect_results.py>`_
* `2d_plot_dispersion.py <postprocessing/2d_plot_dispersion.py>`_

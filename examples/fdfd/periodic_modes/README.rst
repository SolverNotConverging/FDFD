FDFD periodic modes examples
============================

Install the packages first; see `setup <../../README.rst>`_.
The `user guide <../../../doc/solvers/fdfd/periodic_modes/guide.rst>`_ and
`public API <../../../doc/solvers/fdfd/periodic_modes/API_REFERENCE.rst>`_ explain supported controls.

A Matplotlib GUI backend is required to display interactive figures.
These examples retain the FDFD-specific workflow: configure the grid and
materials, solve, then inspect stored fields using the matching viewer.

Recommended order
-----------------

Runtime depends on hardware and mesh size. Single solves are the starting point;
dispersion and band-structure scripts perform many eigenproblems and can take
substantially longer. 3D cases also require more memory.

1. `surface_wave_antenna_2d.py <surface_wave_antenna_2d.py>`_ — A dielectric-loaded periodic surface-wave cell. Single solve.
2. `image_guide_leaky_wave_antenna_3d.py <image_guide_leaky_wave_antenna_3d.py>`_ — A 3D image-guide cell with an outgoing PML. Single solve.
3. `surface_wave_antenna_2d_dispersion.py <surface_wave_antenna_2d_dispersion.py>`_ — A 2D periodic frequency sweep. Frequency sweep.
4. `image_guide_leaky_wave_antenna_3d_dispersion.py <image_guide_leaky_wave_antenna_3d_dispersion.py>`_ — A 3D periodic frequency sweep. Frequency sweep.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``outputs/examples/fdfd/periodic_modes/<example>/`` in the checkout.

Postprocessing
--------------

Run the producing example first. These scripts accept an optional input path
and otherwise load from its standard output directory; use ``--help`` for usage.
Plots are saved beside their source data.

* `inspect_results_3d.py <postprocessing/inspect_results_3d.py>`_
* `plot_dispersion_2d.py <postprocessing/plot_dispersion_2d.py>`_

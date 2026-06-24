Periodic Solver 3D
==================

``PeriodicModeSolver3D`` solves full-vector Bloch-periodic eigenmodes in three-dimensional unit cells. The solver is intended for periodically loaded waveguides, leaky-wave antenna unit cells, and other structures periodic along ``z``.

What It Solves
--------------

The solver builds sparse finite-difference operators on a 3D Yee grid, applies periodicity along the unit-cell direction, and solves a generalized eigenproblem for modal propagation constants. It can reconstruct and store volumetric field arrays for plotting and export.

Main Class
----------

.. code-block:: python

   PeriodicModeSolver3D(Nx, Ny, Nz, x_range, y_range, z_range, freq, num_modes, sigma_guess=None, tol=0, ncv=None)

Parameters:

* ``Nx``, ``Ny``, ``Nz``: grid cells in each direction.
* ``x_range``, ``y_range``, ``z_range``: physical domain spans in metres.
* ``freq``: operating frequency in Hz.
* ``num_modes``: number of modes to compute.
* ``sigma_guess``: optional sparse-eigensolver shift.
* ``tol`` and ``ncv``: optional eigensolver controls.

Material And Boundary API
-------------------------

.. code-block:: python

   add_block(er, mr, x_range, y_range, z_range, subpixels=8)
   add_pec(x_range, y_range, z_range, components=None, epsilon=1e8)
   add_pmc(x_range, y_range, z_range, components=None, mu=1e8)
   add_UPML(sides=('-x', '+x', '-y', '+y'), width=10, max_loss=5, n=3)

Notes:

* ``er`` and ``mr`` can be scalar or anisotropic material values accepted by the implementation.
* Geometry is assigned on a cell material grid with shape ``(Nx, Ny, Nz)`` and averaged onto Yee component locations internally.
* Geometry regions are supplied as ``(min, max)`` pairs using integer grid indices or float physical positions in metres. Python slices are also accepted for index-based regions.
* ``add_block`` uses subpixel fill ratios on the cell material grid before Yee-component averaging.
* ``add_pec`` and ``add_pmc`` apply large material penalties and prevent averaging across those cells.
* ``add_UPML`` accepts side labels such as ``'+y'`` to absorb selected faces.

Solve And Field Storage
-----------------------

.. code-block:: python

   solve()
   store_fields()

After solving, important attributes include:

* ``gammas``: complex propagation constants normalized by ``k0``.
* ``eigenvalues`` and ``eigenvectors``: eigensolver outputs.
* Stored field arrays after ``store_fields`` or visualization routines.

Visualization And Export
------------------------

.. code-block:: python

   plot_field_plane(axis, index, mode_index=0, field='Ex')
   plot(mode=0, x=None, y=None, z=None, save=None, show=True)
   visualize_with_gui()
   save_results(path, include_eigenvectors=False, compressed=True)
   PeriodicModeSolver3D.load_results(path)

Use ``save_results`` for NPZ export and ``load_results`` for post-processing previously computed modes.

Minimal Example
---------------

.. code-block:: python

   from Periodic_Solver_3D import PeriodicModeSolver3D

   solver = PeriodicModeSolver3D(
       Nx=24, Ny=20, Nz=16,
       x_range=6e-3, y_range=6e-3, z_range=8e-3,
       freq=22e9,
       num_modes=2,
       tol=0.1,
   )

   solver.add_block(6.0, 1.0, (6, 18), (13, 19), (0, 16), subpixels=8)
   solver.add_UPML(['+y'], width=6, max_loss=5)
   solver.solve()
   solver.visualize_with_gui()

Examples
--------

* ``example_image_guide_leaky_wave_antenna.py``
* ``Periodic_3D_Dispersion.py``
* ``Load_Results.py``

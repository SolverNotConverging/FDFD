Periodic Solver 2D
==================

``PeriodicModeSolver2D`` solves two-dimensional Bloch-periodic TE or TM modes. The grid spans transverse ``x`` and periodic ``z``. Geometry is assigned on a cell material grid with shape ``(Nx, Nz)`` and then averaged onto Yee-staggered component locations internally.

Main Class
----------

.. code-block:: python

   PeriodicModeSolver2D(
       polarization,
       freq,
       x_range,
       z_range,
       Nx,
       Nz,
       num_modes,
       mode_filter=True,
       guess=0,
       tol=0,
       ncv=None,
   )

Parameters:

* ``polarization``: ``"TE"`` or ``"TM"``.
* ``freq``: operating frequency in Hz.
* ``x_range``, ``z_range``: physical unit-cell spans in metres.
* ``Nx``, ``Nz``: grid cells in transverse and periodic directions.
* ``num_modes``: number of modes requested.
* ``mode_filter``: reserved for API consistency with ``ModeSolver2D``.
* ``guess``: shift target for the sparse eigensolver.
* ``tol`` and ``ncv``: optional eigensolver controls.

Material And Boundary API
-------------------------

.. code-block:: python

   add_rectangle(epsilon, mu, x_range, z_range, subpixels=8)
   add_pec(x_range, z_range, components=None, epsilon=1e8)
   add_pmc(x_range, z_range, components=None, mu=1e8)
   add_pml(pml_width=20, n=3, sigma_max=5.0, direction="all")

Notes:

* ``epsilon`` and ``mu`` can be scalars or length-3 values ordered as ``(xx, yy, zz)``.
* Region bounds accept grid-index pairs or physical coordinate pairs in metres.
* ``add_rectangle`` uses subpixel fill ratios on the cell material grid before Yee-component averaging.
* ``add_pec`` applies a large-permittivity material penalty instead of eliminating DOFs.
* ``add_pmc`` applies a large-permeability material penalty instead of eliminating DOFs.
* ``components`` can select tensor components; ``None`` applies all three.
* PML ``direction`` accepts ``"x-"``, ``"x+"``, ``"x"``, or ``"all"``.

Solve API
---------

.. code-block:: python

   solve(guess=None, tol=None, ncv=None)

``guess``, ``tol``, and ``ncv`` override the instance eigensolver settings for that call. If omitted, the constructor values are used.

After solving, common outputs are:

* ``neff``: complex propagation constants normalized by ``k0``.
* ``propagation_constant``: imaginary part of ``neff`` under the existing plotting convention.
* ``attenuation_constant``: real part of ``neff`` under the existing plotting convention.
* ``eigenvalues`` and ``eigenvectors``: sparse eigensolver outputs.
* ``Ex`` and ``Hy`` for ``"TM"`` polarization.
* ``Hx`` and ``Ey`` for ``"TE"`` polarization.

Visualization
-------------

.. code-block:: python

   visualize_with_gui()

The GUI displays the material map and the active field components for the selected polarization and mode.
PEC/PMC penalty regions are excluded from the material colormap and drawn only on the material subplot so their large values do not dominate the plot scale.

Minimal Example
---------------

.. code-block:: python

   from Periodic_Solver_2D import PeriodicModeSolver2D

   solver = PeriodicModeSolver2D(
       "TM",
       freq=25e9,
       x_range=10e-3,
       z_range=8e-3,
       Nx=200,
       Nz=80,
       num_modes=6,
       guess=0,
   )

   solver.add_rectangle(8.0, 1.0, (10, 25), (0, 80), subpixels=8)
   solver.add_pec((9, 10), (0, 80))
   solver.add_pml(pml_width=30, sigma_max=5, direction="x+")
   solver.solve()

   print(solver.neff)
   solver.visualize_with_gui()

Examples
--------

* ``example_surface_wave_antenna.py``
* ``Periodic_2D_Dispersion.py``

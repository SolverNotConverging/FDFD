Periodic Solver 2D
==================

``PeriodicModeSolver2D`` solves two-dimensional Bloch-periodic TE or TM modes
on a compact Yee grid. The grid spans transverse ``x`` and periodic ``z``.
User-defined materials live on cell centers with shape ``(Nx, Nz)`` and are
interpolated to the actual Yee component locations before solving.

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
       guess=5,
       tol=0,
       ncv=None,
   )

Parameters:

* ``polarization``: ``"TE"`` or ``"TM"``.
* ``freq``: operating frequency in Hz.
* ``x_range``, ``z_range``: physical unit-cell spans in metres.
* ``Nx``, ``Nz``: grid cells in transverse and periodic directions.
* ``num_modes``: number of modes requested.
* ``guess``: shift target for the sparse eigensolver. A nonzero shift is
  recommended for PEC/PML problems.
* ``tol`` and ``ncv``: optional eigensolver controls.

Yee Layout
----------

The periodic ``z`` direction uses only ``Nz`` unique samples. The duplicated
``z = L`` node is not stored; derivative and averaging operators wrap from
``Nz - 1`` to ``0``.

TM fields:

* ``Ex``: shape ``(Nx, Nz, num_modes)``.
* ``Ez``: shape ``(Nx + 1, Nz, num_modes)``.
* ``Hy``: shape ``(Nx, Nz, num_modes)``.

TE fields:

* ``Ey``: shape ``(Nx + 1, Nz, num_modes)``.
* ``Hx``: shape ``(Nx + 1, Nz, num_modes)``.
* ``Hz``: shape ``(Nx, Nz, num_modes)``.

Material And Boundary API
-------------------------

.. code-block:: python

   add_rectangle(epsilon, mu, x_range, z_range, subpixels=8)
   add_pec(x_range, z_range, components=None)
   add_pmc(x_range, z_range, components=None)
   add_pml(pml_width=30, n=3, sigma_max=5.0, direction="all")
   add_UPML(pml_width=30, n=3, sigma_max=5.0, direction="all")

Notes:

* ``epsilon`` and ``mu`` can be scalars or length-3 values ordered as
  ``(xx, yy, zz)``.
* Region bounds accept grid-index pairs or physical coordinate pairs in metres.
* ``add_rectangle`` first computes fractional per-cell coverage on a
  ``subpixels`` by ``subpixels`` grid, blends into cell-centred material arrays,
  then refreshes the component-location ``eps_r_*`` and ``mu_r_*`` arrays.
* ``add_pec`` and ``add_pmc`` are cell-based and expand the selected region to
  component-specific Yee masks.
* ``components=None`` applies PEC or PMC to all tensor components.
* PML ``direction`` accepts ``"x-"``, ``"x+"``, ``"x"``, or ``"all"``.

Solve API
---------

.. code-block:: python

   solve(guess=None, tol=None, ncv=None)

``guess``, ``tol``, and ``ncv`` override the instance eigensolver settings for
that call. If omitted, the constructor values are used.

After solving, common outputs are:

* ``neff``: complex propagation constants normalized by ``k0``.
* ``propagation_constant``: imaginary part of ``neff`` under the existing
  plotting convention.
* ``attenuation_constant``: real part of ``neff`` under the existing plotting
  convention.
* ``eigenvalues`` and ``eigenvectors``: sparse eigensolver outputs.
* ``Ex``, ``Ez``, and ``Hy`` for ``"TM"`` polarization.
* ``Ey``, ``Hx``, and ``Hz`` for ``"TE"`` polarization.

Visualization
-------------

.. code-block:: python

   visualize_with_gui()

The GUI displays the material map and the active staggered field components for
the selected polarization and mode.

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
       guess=5,
   )

   solver.add_rectangle(8.0, 1.0, (10, 25), (0, 80))
   solver.add_pec((9, 10), (0, 80))
   solver.add_pml(pml_width=30, sigma_max=5, direction="x+")
   solver.solve()

   print(solver.neff)
   solver.visualize_with_gui()

Examples
--------

* ``example_surface_wave_antenna.py``
* ``Periodic_2D_Dispersion.py``

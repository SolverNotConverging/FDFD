Periodic Solver 2D
==================

``Periodic_Solver_2D.py`` contains separate TE and TM mode solvers for two-dimensional Bloch-periodic unit cells. The grid spans ``x`` and periodic ``z``; the solver finds complex propagation constants for periodic structures such as leaky-wave antenna cells and periodically loaded waveguides.

Available Solvers
-----------------

.. code-block:: python

   PeriodicTMModeSolver(freq, x_range, z_range, Nx, Nz, num_modes, guess=0, tol=0, ncv=None)
   PeriodicTEModeSolver(freq=30e9, x_range=20e-3, z_range=5e-3, Nx=200, Nz=50, num_modes=4, guess=0, tol=0, ncv=None)

Use ``PeriodicTMModeSolver`` for TM-like modes with ``Hy``, ``Ex``, and ``Ez``. Use ``PeriodicTEModeSolver`` for TE-like modes with ``Ey``, ``Hx``, and ``Hz``.

Parameters:

* ``freq``: operating frequency in Hz.
* ``x_range``, ``z_range``: physical unit-cell spans in metres.
* ``Nx``, ``Nz``: grid cells in the transverse and periodic directions.
* ``num_modes``: number of modes requested.
* ``guess``: shift target for the sparse eigensolver.
* ``tol`` and ``ncv``: optional eigensolver controls.

Material And Boundary API
-------------------------

Both TE and TM classes expose the same public setup methods:

.. code-block:: python

   add_object(epsilon, mu, x_indices, z_indices)
   add_UPML(pml_width=20, n=3, sigma_max=5.0, direction="top")

Notes:

* ``epsilon`` and ``mu`` can be scalars or length-3 anisotropic values.
* ``x_indices`` and ``z_indices`` are Python indices, lists, ranges, or slices accepted by NumPy assignment.
* Large negative or positive material constants are commonly used in examples as metal approximations.
* ``add_UPML`` stretches the transverse boundary. Check the example scripts for accepted direction conventions.

Solve API
---------

.. code-block:: python

   solve()

After solving, common outputs are:

* ``gammas``: complex propagation constants normalized by ``k0``.
* ``eigenvalues`` and ``eigenvectors``: sparse eigensolver results.
* Polarization-specific field arrays used by ``visualize_with_gui``.

The examples interpret the real and imaginary parts of ``gammas`` as attenuation and phase constants according to their plotting convention.

Visualization
-------------

.. code-block:: python

   visualize_with_gui()

The GUI displays the available field components for the selected polarization and mode.

Minimal Example
---------------

.. code-block:: python

   from Periodic_Solver_2D import PeriodicTMModeSolver

   solver = PeriodicTMModeSolver(
       freq=25e9,
       x_range=10e-3,
       z_range=8e-3,
       Nx=200,
       Nz=80,
       num_modes=6,
       guess=0,
   )

   solver.add_object(8.0, 1.0, x_indices=range(10, 25), z_indices=range(80))
   solver.add_UPML(pml_width=30, sigma_max=5, direction="top")
   solver.solve()

   print(solver.gammas)
   solver.visualize_with_gui()

Examples
--------

* ``example_surface_wave_leaky_wave_antenna.py``
* ``Periodic_2D_Dispersion.py``

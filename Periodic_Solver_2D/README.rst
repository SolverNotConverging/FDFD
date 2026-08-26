Periodic Solver 2D
==================

``PeriodicModeSolver2D`` solves two-dimensional Bloch-periodic TE or TM modes. The grid spans transverse ``x`` and periodic ``z``. Geometry is assigned on a cell material grid with shape ``(Nx, Nz)`` and then averaged onto Yee-staggered component locations internally.

The solver uses the ``exp(+j*omega*t)`` phasor convention. Its raw
eigenvalue is the dimensional spatial exponent ``gamma = alpha + j*beta``
in ``exp(-gamma*z)``. The standard effective index is therefore
``neff = -j*gamma/k0 = beta/k0 - j*alpha/k0``. Passive forward modes have
``Im(neff) <= 0``. The paired forward Fourier-transform kernel is
``exp(-j*omega*t)``.

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
* ``guess``: shift target for the dimensional spatial exponent ``gamma`` in inverse metres, where ``gamma = j*k0*neff``.
* ``tol`` and ``ncv``: optional eigensolver controls.

Material And Boundary API
-------------------------

.. code-block:: python

   add_rectangle(epsilon, mu, x_range, z_range, subpixels=8)
   add_pec(x_range, z_range, components=None)
   add_pmc(x_range, z_range, components=None)
   add_pml(pml_width=20, n=3, sigma_max=5.0, direction="all")

Notes:

* ``epsilon`` and ``mu`` can be scalars or length-3 values ordered as ``(xx, yy, zz)``.
* Under ``exp(+j*omega*t)``, passive lossy permittivity and permeability use negative imaginary parts.
* Region bounds accept grid-index pairs or physical coordinate pairs in metres.
* ``add_rectangle`` uses subpixel fill ratios on the cell material grid before Yee-component averaging.
* ``add_pec`` and ``add_pmc`` expand cell regions onto the periodic Yee grids and eliminate the constrained DOFs from both generalized-eigenproblem matrices.
* PEC faces constrain tangential electric fields and normal magnetic fields, leaving normal electric and tangential magnetic components free. PMC faces apply the electromagnetic-dual constraints.
* Face orientation is retained on both axes. For example, a z-normal PEC face constrains tangential electric fields plus normal ``Hz``; the dual PMC face constrains tangential magnetic fields plus normal ``Ez``. At a corner, the constraints required by either incident face are combined.
* Adjacent regions are merged on the cell grid before interfaces are derived, so no false internal wall is introduced.
* Longitudinal ``Ez`` PEC and ``Hz`` PMC constraints are enforced exactly by zeroing their entries in the inverse material operators used for Schur elimination.
* ``components`` can select tensor components; ``None`` applies all three.
* PEC/PMC regions do not modify the material tensors or introduce a large-value penalty.
* PML ``direction`` accepts ``"x-"``, ``"x+"``, ``"x"``, or ``"all"``; ``sigma_max`` must be finite and nonnegative.

Solve API
---------

.. code-block:: python

   solve(guess=None, tol=None, ncv=None)

``guess``, ``tol``, and ``ncv`` override the instance eigensolver settings for that call. If omitted, the constructor values are used.

After solving, common outputs are:

* ``gammas``: normalized spatial exponents ``gamma/k0`` for ``exp(-gamma*z)``.
* ``neff``: complex effective indices, calculated as ``-j*gamma/k0``.
* ``propagation_constant``: normalized phase constants ``Re(neff) = beta/k0``.
* ``attenuation_constant``: normalized positive-forward attenuation ``-Im(neff) = alpha/k0``.
* ``eigenvalues``: dimensional spatial exponents ``gamma`` in inverse metres.
* ``eigenvectors``: sparse eigensolver field vectors.
* ``Ex`` and ``Hy`` for ``"TM"`` polarization.
* ``Hx`` and ``Ey`` for ``"TE"`` polarization.

Visualization
-------------

.. code-block:: python

   visualize_with_gui()

The GUI displays the material map and the active field components for the selected polarization and mode.
PEC/PMC regions are excluded from the material colormap and drawn as boundary overlays on the material subplot.

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

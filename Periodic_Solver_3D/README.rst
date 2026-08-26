Periodic Solver 3D
==================

``PeriodicModeSolver3D`` solves full-vector Bloch-periodic eigenmodes in three-dimensional unit cells. The solver is intended for periodically loaded waveguides, leaky-wave antenna unit cells, and other structures periodic along ``z``.

The solver uses the ``exp(+j*omega*t)`` phasor convention. Its raw
eigenvalue is the dimensional spatial exponent ``gamma = alpha + j*beta``
in ``exp(-gamma*z)``. The standard effective index is therefore
``neff = -j*gamma/k0 = beta/k0 - j*alpha/k0``. Passive forward modes have
``Im(neff) <= 0``. The paired forward Fourier-transform kernel is
``exp(-j*omega*t)``.

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
* ``sigma_guess``: optional shift for the dimensional spatial exponent ``gamma`` in inverse metres.
* ``tol`` and ``ncv``: optional eigensolver controls.

Material And Boundary API
-------------------------

.. code-block:: python

   add_block(er, mr, x_range, y_range, z_range, subpixels=8)
   add_pec(x_range, y_range, z_range, components=None)
   add_pmc(x_range, y_range, z_range, components=None)
   add_UPML(sides=('-x', '+x', '-y', '+y'), width=10, max_loss=5, n=3)

Notes:

* ``er`` and ``mr`` can be scalar or anisotropic material values accepted by the implementation.
* Under ``exp(+j*omega*t)``, passive lossy permittivity and permeability use negative imaginary parts.
* Geometry is assigned on a cell material grid with shape ``(Nx, Ny, Nz)`` and averaged onto Yee component locations internally.
* Geometry regions are supplied as ``(min, max)`` pairs using integer grid indices or float physical positions in metres. Python slices are also accepted for index-based regions.
* ``add_block`` uses subpixel fill ratios on the cell material grid before Yee-component averaging.
* ``add_pec`` and ``add_pmc`` expand cell regions onto the staggered component grids and eliminate constrained transverse DOFs from both generalized-eigenproblem matrices.
* PEC faces constrain tangential electric fields and normal magnetic fields, leaving normal electric and tangential magnetic components free. PMC faces apply the electromagnetic-dual constraints.
* Face orientation is retained along x, y, and periodic z. At edges and corners the masks use union semantics, so every component required by any incident face is constrained. For example, a z-normal PEC face constrains ``Ex``, ``Ey``, and normal ``Hz``; a z-normal PMC face constrains ``Hx``, ``Hy``, and normal ``Ez``.
* Schur-eliminated longitudinal fields remain exact: PEC ``Ez`` and PMC ``Hz`` entries are removed by zeroing their inverse-material entries.
* PEC/PMC regions do not modify the material tensors or introduce a large-value penalty.
* ``add_UPML`` accepts side labels such as ``'+y'`` to absorb selected faces; ``max_loss`` must be finite and nonnegative.

Solve And Field Storage
-----------------------

.. code-block:: python

   solve()
   store_fields()

After solving, important attributes include:

* ``eigenvalues``: dimensional spatial exponents ``gamma`` in inverse metres.
* ``gammas``: normalized spatial exponents ``gamma/k0``.
* ``neff``: complex effective indices ``-j*gamma/k0``.
* ``propagation_constant``: normalized phase constants ``Re(neff) = beta/k0``.
* ``attenuation_constant``: normalized positive-forward attenuation ``-Im(neff) = alpha/k0``.
* ``eigenvectors``: eigensolver field vectors.
* Stored field arrays after ``store_fields`` or visualization routines.

Reduced eigenvectors are expanded back to the full staggered field ordering after each solve, with every constrained entry restored as an exact zero. Saved result files include all six PEC/PMC masks so a loaded model retains its boundary constraints.

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

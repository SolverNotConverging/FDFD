Mode Solver 1D
==============

``ModeSolver1D`` solves one-dimensional slab-waveguide eigenmodes on a true staggered Yee grid. The structure varies along ``x`` and is uniform along the propagation direction.

What It Solves
--------------

Use this solver for dielectric slabs, grounded slabs, impedance-loaded sheets, and quick modal dispersion sweeps.

The solver supports:

* TE modes with primary ``Ey`` and reconstructed ``Hx`` and ``Hz``.
* TM modes with primary ``Hy`` and reconstructed ``Ex`` and ``Ez``.
* Isotropic or diagonal-anisotropic relative ``epsilon`` and ``mu``.
* Cell-centred PEC and PMC regions expanded to component-specific Yee masks.
* Simple uniaxial PML stretching at the left and/or right boundary.
* Electric impedance-sheet perturbations.

Grid Layout
-----------

The source material grid has shape ``(Nx,)`` and is cell-centred.

After ``solve()``, fields are stored on their native staggered locations:

* ``Ex``, ``Hy``, ``Hz``: cell-centred arrays with shape ``(Nx, num_modes)``.
* ``Ey``, ``Ez``, ``Hx``: node arrays with shape ``(Nx + 1, num_modes)``.

Component-location material arrays use the same locations:

* ``eps_r_xx``: shape ``(Nx,)``.
* ``eps_r_yy`` and ``eps_r_zz``: shape ``(Nx + 1,)``.
* ``mu_r_xx``: shape ``(Nx + 1,)``.
* ``mu_r_yy`` and ``mu_r_zz``: shape ``(Nx,)``.

Main Class
----------

.. code-block:: python

   ModeSolver1D(frequency, x_range, Nx, num_modes, guess=None)

Parameters:

* ``frequency``: operating frequency in Hz.
* ``x_range``: physical domain width in metres.
* ``Nx``: number of grid cells.
* ``num_modes``: number of TE and TM modes to request.
* ``guess``: shift-invert target passed to ``scipy.sparse.linalg.eigs`` when ``solve(sigma=None)`` is used. If ``None``, the solver uses the maximum magnitude of the cell-centred material tensors.

Material And Boundary API
-------------------------

.. code-block:: python

   add_layer(epsilon, mu, x_range, subpixels=8)
   add_pec(x_range, components=None)
   add_pmc(x_range, components=None)
   add_pml(pml_width=50, n=3, sigma_max=25, direction="all")
   add_impedance_surface(Zs, position, thickness_cells=1, eps_components=("xx", "yy", "zz"))

Notes:

* ``epsilon`` and ``mu`` can be scalars or length-3 values ordered as ``(xx, yy, zz)``.
* ``x_range`` accepts grid-index pairs or physical coordinate pairs in metres.
* ``add_layer`` computes fractional per-cell coverage on a ``subpixels`` sample grid, blends into cell-centred source arrays named ``cell_eps_r_*`` and ``cell_mu_r_*``, then refreshes the Yee-grid component arrays.
* ``components`` can be ``"xx"``, ``"yy"``, ``"zz"`` or an iterable of those names. ``None`` applies all three.
* PML ``direction`` accepts ``"x-"``, ``"x+"``, ``"x"``, or ``"all"``.

Solve API
---------

.. code-block:: python

   solve(sigma=None)

``sigma`` overrides the constructor ``guess`` for that solve. If both are ``None``, the automatic material-magnitude target is recomputed before calling ``eigs``.

After ``solve()``, the main outputs are:

* ``neff_TE`` and ``neff_TM``: complex effective indices for TE and TM modes.
* ``propagation_constant_TE`` and ``propagation_constant_TM``: real parts of ``neff``.
* ``attenuation_constant_TE`` and ``attenuation_constant_TM``: imaginary parts of ``neff``.
* ``Ey``, ``Hx``, ``Hz``, ``Hy``, ``Ex``, ``Ez``: fields on native staggered locations.

Visualization
-------------

.. code-block:: python

   visualize(mode=1, ey=True, hz=True)
   visualize_with_gui()

``visualize`` plots selected 1D field profiles. If no fields are selected, it plots all TE and TM components. Field plots include inferno material-layer backgrounds and yellow/blue PEC/PMC layer overlays. ``visualize_with_gui`` opens an interactive mode selector.

Minimal Example
---------------

.. code-block:: python

   from Mode_Solver_1D import ModeSolver1D

   solver = ModeSolver1D(frequency=30e9, x_range=10e-3, Nx=1000, num_modes=4)
   solver.add_layer(epsilon=10.2, mu=1.0, x_range=(3e-3, 4.27e-3))
   solver.add_pec((2.9e-3, 3.0e-3))
   solver.solve()

   print(solver.neff_TE)
   print(solver.neff_TM)
   solver.visualize_with_gui()

Examples
--------

* ``example_grounded_isotropic_slab.py``
* ``example_anisotropic_slab.py``
* ``Modal_1D_Dispersion.py``

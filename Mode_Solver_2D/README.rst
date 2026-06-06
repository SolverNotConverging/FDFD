Mode Solver 2D
==============

``ModeSolver2D`` solves full-vector electromagnetic modes of two-dimensional waveguide cross-sections. The structure varies in ``x`` and ``y`` and is uniform along the propagation direction.

What It Solves
--------------

Use this solver for dielectric waveguides, ridge guides, loaded cross-sections, and bounded or open waveguide mode calculations. It assembles transverse electric and magnetic operators, solves a sparse eigenproblem, and reconstructs all six field components.

The solver supports:

* Diagonal-anisotropic relative ``epsilon`` and ``mu`` tensors.
* PEC and PMC masks for selected Yee-grid components.
* Component masks generated from cell-centred PEC/PMC regions.
* Simple uniaxial PML in ``x``, ``y``, or all boundaries.
* Impedance sheets normal to ``x`` or ``y``.
* True staggered Yee-grid field storage and rectangular curl operators.

Main Class
----------

.. code-block:: python

   ModeSolver2D(frequency, x_range, y_range, Nx, Ny, num_modes, guess=None)

Parameters:

* ``frequency``: operating frequency in Hz.
* ``x_range``, ``y_range``: physical cross-section spans in metres.
* ``Nx``, ``Ny``: grid cells in ``x`` and ``y``.
* ``num_modes``: number of modes to retain.
* ``guess``: shift-invert target passed to ``scipy.sparse.linalg.eigs`` when ``solve(sigma=None)`` is used. If ``None``, the solver uses ``-max(abs(cell_eps_r_xx), abs(cell_eps_r_yy), abs(cell_eps_r_zz), abs(cell_mu_r_xx), abs(cell_mu_r_yy), abs(cell_mu_r_zz))`` from the current cell-centred material tensors.

Material And Boundary API
-------------------------

.. code-block:: python

   add_rectangle(epsilon, mu, x_range, y_range, average=True)
   add_pec(x_range, y_range, components=None)
   add_pmc(x_range, y_range, components=None)
   add_pml(pml_width=50, n=3, sigma_max=5, direction="all")
   add_UPML(pml_width=50, n=3, sigma_max=5, direction="all")
   add_impedance_surface(Zs, position, orientation="x", thickness_cells=1, eps_components=("xx", "yy", "zz"))

Notes:

* ``epsilon`` and ``mu`` can be scalars or length-3 values ordered as ``(xx, yy, zz)``.
* Region bounds can be integer grid indices or physical coordinates in metres.
* ``add_rectangle`` writes material to cell-centred source arrays named ``cell_eps_r_*`` and ``cell_mu_r_*``.
* The solver interpolates source materials onto component-location arrays named ``eps_r_*`` and ``mu_r_*``.
* With ``average=True``, component materials are averaged from neighbouring cells.
* With ``average=False``, the cell material is stamped directly onto all surrounding Yee component material locations.
* PEC/PMC ``components=None`` treats the region as cell-centred and expands it to component-specific Yee masks.
* PML ``direction`` accepts ``"x-"``, ``"x+"``, ``"x"``, ``"y-"``, ``"y+"``, ``"y"``, or ``"all"``.

Solve API
---------

.. code-block:: python

   solve(sigma=None)

``sigma`` overrides the constructor ``guess`` for that solve. If both are ``None``, the automatic material-magnitude target is recomputed before calling ``eigs``.

After ``solve()``, outputs include:

* ``neff``: complex effective index for each selected mode.
* ``propagation_constant`` and ``attenuation_constant``: real and imaginary parts of ``neff``.
* ``eigenvalues`` and ``eigenvectors``: selected sparse eigensystem outputs.
* ``Ex`` and ``Hy``: staggered field arrays of shape ``(Nx, Ny + 1, num_modes)``.
* ``Ey`` and ``Hx``: staggered field arrays of shape ``(Nx + 1, Ny, num_modes)``.
* ``Ez``: staggered field array of shape ``(Nx + 1, Ny + 1, num_modes)``.
* ``Hz``: cell-centred field array of shape ``(Nx, Ny, num_modes)``.

Visualization
-------------

.. code-block:: python

   visualize(mode=1, ex=True, ey=True, ez=True)
   visualize_with_gui()

``visualize`` plots selected components or all six components by default. It also supports ``eabs=True`` and ``habs=True`` for magnitude plots. ``visualize_with_gui`` opens a six-panel field viewer with a mode selector.

Minimal Example
---------------

.. code-block:: python

   from Mode_Solver_2D import ModeSolver2D

   solver = ModeSolver2D(30e9, 24e-3, 16e-3, 240, 160, num_modes=5)
   solver.add_rectangle(3.0, 1.0, (0, 240), (60, 80))
   solver.add_rectangle(12.0, 1.0, (100, 140), (80, 100))
   solver.add_UPML(pml_width=30, sigma_max=1, direction="x")
   solver.solve()

   print(solver.neff)
   solver.visualize_with_gui()

Examples
--------

* ``example_ridge_dielectric_waveguide.py``
* ``example_microstrip.py``
* ``Modal_2D_Dispersion.py``

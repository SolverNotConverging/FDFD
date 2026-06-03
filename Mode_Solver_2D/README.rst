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
* Simple uniaxial PML in ``x``, ``y``, or both directions.
* Impedance sheets normal to ``x`` or ``y``.
* Optional filtering of PEC-neighbour-dominated spurious candidates.

Main Class
----------

.. code-block:: python

   ModeSolver2D(frequency, x_range, y_range, Nx, Ny, num_modes, mode_filter=True, guess=None)

Parameters:

* ``frequency``: operating frequency in Hz.
* ``x_range``, ``y_range``: physical cross-section spans in metres.
* ``Nx``, ``Ny``: grid cells in ``x`` and ``y``.
* ``num_modes``: number of modes to retain.
* ``mode_filter``: whether to request extra candidates and filter likely spurious PEC-localized modes.
* ``guess``: shift-invert target passed to ``scipy.sparse.linalg.eigs`` when ``solve(sigma=None)`` is used. If ``None``, the solver uses ``-max(abs(eps_r_xx), abs(eps_r_yy), abs(eps_r_zz), abs(mu_r_xx), abs(mu_r_yy), abs(mu_r_zz))`` from the current material tensors.

Material And Boundary API
-------------------------

.. code-block:: python

   add_rectangle(epsilon, mu, x_range, y_range)
   add_pec(x_range, y_range, components=None)
   add_pmc(x_range, y_range, components=None)
   add_pml(pml_width=50, n=3, sigma_max=5, direction="both")
   add_UPML(pml_width=50, n=3, sigma_max=5, direction="both")
   add_impedance_surface(Zs, position, orientation="x", thickness_cells=1, eps_components=("xx", "yy", "zz"))

Notes:

* ``epsilon`` and ``mu`` can be scalars or length-3 values ordered as ``(xx, yy, zz)``.
* Region bounds can be integer grid indices or physical coordinates in metres.
* PEC/PMC ``components=None`` treats the region as cell-centred and expands it to component-specific Yee masks.
* PML ``direction`` accepts ``"x-"``, ``"x+"``, ``"x"``, ``"y-"``, ``"y+"``, ``"y"``, or ``"both"``.

Solve API
---------

.. code-block:: python

   solve(sigma=None, extra_modes=8, max_pec_neighbor_energy_fraction=0.35)

``sigma`` overrides the constructor ``guess`` for that solve. If both are ``None``, the automatic material-magnitude target is recomputed before calling ``eigs``.

After ``solve()``, outputs include:

* ``neff``: complex effective index for each selected mode.
* ``propagation_constant`` and ``attenuation_constant``: real and imaginary parts of ``neff``.
* ``eigenvalues`` and ``eigenvectors``: selected sparse eigensystem outputs.
* ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``: flattened field arrays of shape ``(Nx * Ny, num_modes)``.
* ``spurious_scores`` and candidate-index arrays when filtering is enabled.

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

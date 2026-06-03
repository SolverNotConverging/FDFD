Mode Solver 1D
==============

``ModeSolver1D`` solves one-dimensional slab-waveguide eigenmodes on a Yee-style finite-difference grid. It computes separate TE and TM scalar eigenproblems and reconstructs the available transverse and longitudinal field components.

What It Solves
--------------

Use this solver for structures that vary along one transverse coordinate ``x`` and are uniform along the propagation direction. Typical examples are grounded slabs, dielectric slabs, impedance-loaded sheets, and quick modal dispersion sweeps.

The solver supports:

* TE modes with primary field ``Ey`` and reconstructed ``Hx`` and ``Hz``.
* TM modes with primary field ``Hy`` and reconstructed ``Ex`` and ``Ez``.
* Isotropic or diagonal-anisotropic relative ``epsilon`` and ``mu``.
* PEC masks for electric-field components.
* PMC masks for magnetic-field components.
* Simple uniaxial PML stretching at the left and/or right boundary.
* Electric impedance-sheet perturbations.

Main Class
----------

.. code-block:: python

   ModeSolver1D(frequency, x_range, Nx, num_modes, guess=None)

Parameters:

* ``frequency``: operating frequency in Hz.
* ``x_range``: physical domain width in metres.
* ``Nx``: number of grid cells.
* ``num_modes``: number of TE and TM modes to request.
* ``guess``: shift-invert target passed to ``scipy.sparse.linalg.eigs`` when ``solve(sigma=None)`` is used. If ``None``, the solver uses ``-max(abs(eps_r_xx), abs(eps_r_yy), abs(eps_r_zz), abs(mu_r_xx), abs(mu_r_yy), abs(mu_r_zz))`` from the current material tensors.

Material And Boundary API
-------------------------

.. code-block:: python

   add_layer(epsilon, mu, x_range)
   add_pec(x_range, components=None)
   add_pmc(x_range, components=None)
   add_pml(pml_width=50, n=3, sigma_max=25, direction="both")
   add_UPML(pml_width=50, n=3, sigma_max=25, direction="both")
   add_impedance_surface(Zs, position, thickness_cells=1, eps_components=("xx", "yy", "zz"))

Notes:

* ``epsilon`` and ``mu`` can be scalars or length-3 values ordered as ``(xx, yy, zz)``.
* ``x_range`` accepts grid-index pairs or physical coordinate pairs in metres.
* PEC/PMC ``components`` can be ``"xx"``, ``"yy"``, ``"zz"`` or an iterable of those names. ``None`` applies all three.
* ``add_UPML`` is a compatibility alias for ``add_pml``.
* PML ``direction`` accepts ``"x-"``, ``"x+"``, ``"x"``, ``"top"``, ``"bottom"``, or ``"both"``.

Solve API
---------

.. code-block:: python

   solve(sigma=None)

``sigma`` overrides the constructor ``guess`` for that solve. If both are ``None``, the automatic material-magnitude target is recomputed before calling ``eigs``.

After ``solve()``, the main outputs are:

* ``neff_TE`` and ``neff_TM``: complex effective indices for TE and TM modes.
* ``propagation_constant_TE`` and ``propagation_constant_TM``: real parts of ``neff``.
* ``attenuation_constant_TE`` and ``attenuation_constant_TM``: imaginary parts of ``neff``.
* ``Ey``, ``Hx``, ``Hz``, ``Hy``, ``Ex``, ``Ez``: field arrays with shape ``(Nx, num_modes)``.
* ``fields``: nested compatibility dictionary with ``fields["TE"]`` and ``fields["TM"]``.

``beta_TE``, ``beta_TM``, ``alpha_TE``, and ``alpha_TM`` remain as legacy aliases for effective-index real and imaginary parts.

Visualization
-------------

.. code-block:: python

   visualize(mode=1, ey=True, hz=True)
   visualize_with_gui()

``visualize`` plots selected 1D field profiles. If no fields are selected, it plots all TE and TM components. ``visualize_with_gui`` opens an interactive mode selector.

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

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
* Opaque scalar surface-impedance boundaries compiled from cell regions.
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

   add_rectangle(epsilon, mu, x_range, y_range, subpixels=8)
   add_circle(epsilon, mu, center, r1, r2=None, subpixels=8)
   add_triangle(epsilon, mu, p1, p2, p3, subpixels=8)
   add_pec(x_range, y_range, components=None)
   add_pmc(x_range, y_range, components=None)
   add_pml(pml_width=50, n=3, sigma_max=5, direction="all")
   add_UPML(pml_width=50, n=3, sigma_max=5, direction="all")
   add_impedance_surface(Zs=None, *, preset=None, x_range, y_range)

Notes:

* ``epsilon`` and ``mu`` can be scalars or length-3 values ordered as ``(xx, yy, zz)``.
* Region bounds can be integer grid indices or physical coordinates in metres.
* ``add_rectangle``, ``add_circle``, and ``add_triangle`` first compute fractional per-cell coverage on a ``subpixels`` by ``subpixels`` grid, blend the material into cell-centred source arrays named ``cell_eps_r_*`` and ``cell_mu_r_*``, then refresh the Yee-grid component arrays.
* ``add_circle`` accepts ``r1`` as the outer radius and optional ``r2`` as the inner radius for annuli. Float radii are metres; integer radii are cell counts.
* Shape points and centres can be integer grid coordinates or physical coordinates in metres.
* The solver interpolates source materials onto component-location arrays named ``eps_r_*`` and ``mu_r_*``.
* PEC/PMC ``components=None`` treats the region as cell-centred and expands it to component-specific Yee masks.
* Transverse boundary masks are cross-constrained on collocated Yee pairs: PEC ``Ex`` implies PMC ``Hy`` (and vice versa), while PEC ``Ey`` implies PMC ``Hx`` (and vice versa).
* PML ``direction`` accepts ``"x-"``, ``"x+"``, ``"x"``, ``"y-"``, ``"y+"``, ``"y"``, or ``"all"``.

Time, Propagation, And Loss Convention
--------------------------------------

Fields use

.. math::

   F(z,t)=\operatorname{Re}\{\widetilde F\exp(+j\omega t-j\beta z)\}.

The corresponding forward Fourier-transform kernel is ``exp(-j*omega*t)``;
the ``+j*omega*t`` factor above is the inverse-transform/phasor synthesis.
Consequently, passive bulk materials have ``Im(epsilon) <= 0`` and
``Im(mu) <= 0``.  For example, electric conductivity is represented by

.. math::

   \epsilon_r=\epsilon_r'-j\frac{\sigma}{\omega\epsilon_0}.

A forward passive mode has ``Im(neff) <= 0``, and the solver reports the
positive normalized attenuation as ``-Im(neff)``.  The uniaxial PML uses
``Sx = 1 - j sigma_x/(omega epsilon0)`` and the analogous ``Sy``.  A positive
imaginary bulk constitutive parameter therefore represents gain, not loss.

Surface Impedance Boundaries
----------------------------

``add_impedance_surface`` marks an opaque rectangular cell region.  Fields in
the marked cells are excluded, while every exposed face between an opaque and
retained cell receives a one-sided scalar Leontovich boundary condition.  The
ranges describe the solid region; they are not a numerical film thickness.

Provide exactly one of a complex ``Zs`` in ohms or a named metal ``preset``:

.. code-block:: python

   # Constant impedance at the solver frequency.
   solver.add_impedance_surface(
       Zs=0.025 + 0.030j,
       x_range=(0, 2),
       y_range=(0, solver.Ny),
   )

   # Good-conductor copper impedance evaluated at solver.frequency.
   solver.add_impedance_surface(
       preset="Cu",
       x_range=(solver.Nx - 2, solver.Nx),
       y_range=(0, solver.Ny),
   )

Under the solver convention
``F(z, t) = Re{F~ exp(+j omega t - j beta z)}``, the normal points from the
opaque region into the retained medium and ``E_t = Zs (n x H_t)``.  A preset
is evaluated directly as

.. math::

   Z_s=(1+j)\sqrt{\frac{\omega\mu_0}{2\sigma}}
      =(1+j)\sqrt{\pi f\mu_0\rho}.

Preset names and chemical symbols are case-insensitive. ``aluminum`` is also
accepted as an alias for ``aluminium``.

.. list-table:: Metal surface-impedance presets at the cited reference condition
   :header-rows: 1

   * - Preset (symbol)
     - Resistivity (ohm metre)
     - Conductivity (S/m)
     - Source
   * - aluminium (Al)
     - ``2.650e-8``
     - ``3.774e7``
     - DES1984A_
   * - copper (Cu)
     - ``1.676e-8``
     - ``5.967e7``
     - MAT1979_
   * - gold (Au)
     - ``2.192e-8``
     - ``4.562e7``
     - MAT1979_
   * - molybdenum (Mo)
     - ``5.340e-8``
     - ``1.873e7``
     - DES1984S_
   * - palladium (Pd)
     - ``1.054e-7``
     - ``9.488e6``
     - MAT1979_
   * - silver (Ag)
     - ``1.586e-8``
     - ``6.305e7``
     - MAT1979_
   * - tungsten (W)
     - ``5.280e-8``
     - ``1.894e7``
     - DES1984S_
   * - zinc (Zn)
     - ``5.964e-8``
     - ``1.677e7``
     - DES1984S_

.. _DES1984A: https://doi.org/10.1063/1.555725
.. _MAT1979: https://doi.org/10.1063/1.555614
.. _DES1984S: https://doi.org/10.1063/1.555723

``Zs`` must be finite, nonzero, scalar, and passive
(``Re(Zs) >= 0``); use ``add_pec`` for the exact ``Zs=0`` limit.  Identical
definitions may overlap and union, while different definitions may not
overlap. PEC/PMC regions must not constrain a Yee component used by an SIBC;
this excludes both cell overlaps and immediately adjacent row conflicts. PML
cells must neither overlap an impedance region nor touch one of its exposed
faces. Point-only diagonal contacts are rejected as non-manifold. The public
``impedance_surface_mask`` property returns a copy of the marked-cell mask.

Solve API
---------

.. code-block:: python

   solve(sigma=None)

``sigma`` overrides the constructor ``guess`` for that solve. If both are ``None``, the automatic material-magnitude target is recomputed before calling ``eigs``.

After ``solve()``, outputs include:

* ``neff``: complex effective index for each selected mode.
* ``propagation_constant``: real part of ``neff``.
* ``attenuation_constant``: positive normalized attenuation ``-Im(neff)`` for passive forward modes.
* ``eigenvalues`` and ``eigenvectors``: selected sparse eigensystem outputs.
* ``Ex`` and ``Hy``: staggered field arrays of shape ``(Nx, Ny + 1, num_modes)``.
* ``Ey`` and ``Hx``: staggered field arrays of shape ``(Nx + 1, Ny, num_modes)``.
* ``Ez``: staggered field array of shape ``(Nx + 1, Ny + 1, num_modes)``.
* ``Hz``: cell-centred field array of shape ``(Nx, Ny, num_modes)``.

The magnetic arrays use the solver normalization
``H_stored = -j*eta0*H_physical``. Recover physical magnetic fields with
``H_physical = j*H_stored/eta0``. This keeps the dimensionless curl equations
compact while preserving the ``exp(+j*omega*t - j*beta*z)`` Maxwell phases.

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

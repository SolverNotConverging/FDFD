Mode Solver 1D
==============

``ModeSolver1D`` solves one-dimensional slab-waveguide eigenmodes on a true staggered Yee grid. The structure varies along ``x`` and is uniform along the propagation direction.

What It Solves
--------------

Use this solver for dielectric slabs, grounded slabs, and quick modal dispersion sweeps.

The solver supports:

* TE modes with primary ``Ey`` and reconstructed ``Hx`` and ``Hz``.
* TM modes with primary ``Hy`` and reconstructed ``Ex`` and ``Ez``.
* Isotropic or diagonal-anisotropic relative ``epsilon`` and ``mu``.
* Cell-centred PEC and PMC regions expanded to component-specific Yee masks.
* Opaque scalar surface-impedance boundaries compiled at retained/solid cell interfaces.
* Simple uniaxial PML stretching at the left and/or right boundary.

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
   add_impedance_surface(Zs=None, *, preset=None, x_range)

Notes:

* ``epsilon`` and ``mu`` can be scalars or length-3 values ordered as ``(xx, yy, zz)``.
* ``x_range`` accepts grid-index pairs or physical coordinate pairs in metres.
* ``add_layer`` computes fractional per-cell coverage on a ``subpixels`` sample grid, blends into cell-centred source arrays named ``cell_eps_r_*`` and ``cell_mu_r_*``, then refreshes the Yee-grid component arrays.
* ``components`` can be ``"xx"``, ``"yy"``, ``"zz"`` or an iterable of those names. ``None`` applies all three.
* Transverse boundary masks are closed on collocated Yee pairs: PEC ``Ex`` implies PMC ``Hy`` (and vice versa), while PEC ``Ey`` implies PMC ``Hx`` (and vice versa).
* PML ``direction`` accepts ``"x-"``, ``"x+"``, ``"x"``, or ``"all"``.

Time, Propagation, And Loss Convention
--------------------------------------

Fields use

.. math::

   F(z,t)=\mathrm{Re}\{\widetilde F\exp(+j\omega t-j\beta z)\}.

The corresponding forward Fourier-transform kernel is ``exp(-j*omega*t)``;
the ``+j*omega*t`` factor above is the inverse-transform/phasor synthesis.
Consequently, passive bulk materials have ``Im(epsilon) <= 0`` and
``Im(mu) <= 0``.  For example, electric conductivity is represented by

.. math::

   \epsilon_r=\epsilon_r'-j\frac{\sigma}{\omega\epsilon_0}.

A forward passive mode has ``Im(neff) <= 0``, and the solver reports the
positive normalized attenuation as ``-Im(neff)``.  The uniaxial PML uses
``S = 1 - j sigma/(omega epsilon0)`` under the same convention.  A positive
imaginary bulk constitutive parameter therefore represents gain, not loss.

Surface Impedance Boundaries
----------------------------

``add_impedance_surface`` marks an opaque cell interval. Fields supported only
by marked cells are excluded, and every transition between an opaque cell and
a retained cell receives a one-sided scalar Leontovich boundary condition. The
range describes solid cells; it is not a numerical thin-film thickness, and
the source ``epsilon`` and ``mu`` arrays are not modified.

Provide exactly one of a complex ``Zs`` in ohms or a named metal ``preset``:

.. code-block:: python

   # One-cell copper walls around a retained parallel-plate aperture.
   solver.add_impedance_surface(
       preset="Cu",
       x_range=(0, 1),
   )
   solver.add_impedance_surface(
       preset="copper",
       x_range=(solver.Nx - 1, solver.Nx),
   )

   # A constant impedance at solver.frequency.
   solver.add_impedance_surface(
       Zs=0.025 + 0.030j,
       x_range=(20, 22),
   )

Under the solver convention
``F(z, t) = Re{F~ exp(+j omega t - j beta z)}``, the normal points from the
opaque region into the retained medium and ``E_t = Zs (n x H_t)``. At each
opaque/retained transition, the half-cell Ampere rule changes both tangential
electric coefficients according to

.. math::

   \epsilon_{p,\mathrm{eff}}
   = \epsilon_{p,\mathrm{retained}}
   + \frac{2}{j\omega\epsilon_0\Delta x Z_s},
   \qquad p\in\{yy,zz\}.

The matching normalized curl row is also clipped to the retained half-cell.
If ``c`` is the retained magnetic-cell index at interface node ``i``, its only
nonzero entry is

.. math::

   (D_{H\rightarrow E})_{i,c}
   = \begin{cases}
       +2/(k_0\Delta x), & \text{opaque cell on the left},\\
       -2/(k_0\Delta x), & \text{opaque cell on the right}.
     \end{cases}

The preset good-conductor impedance is evaluated directly at
``solver.frequency``:

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
(``Re(Zs) >= 0``); use ``add_pec`` for the exact ``Zs=0`` limit. Identical
definitions may overlap and union. Different definitions may be disjoint or
adjacent, but they may not overlap. Adjacent opaque cells do not create an
internal boundary row; the definition on the opaque cell touching an exposed
face controls that face.

A terminal opaque run has one exposed in-domain interface. An internal opaque
run has two interfaces and separates the retained domain into independent
apertures. Multiple runs are supported, but modes from all disconnected
apertures share one returned spectrum; degenerate modes can mix numerically.
A wall must occupy at least one cell, and an all-opaque domain is invalid.

PEC and PMC cells may not overlap an impedance interval. At a shared exposed
interface, constraints are checked after closing the collocated Yee pairs
``PEC Ex <-> PMC Hy`` and ``PEC Ey <-> PMC Hx``. Therefore an ``xx``-only
request is not automatically normal-only: adjacent PEC ``xx`` constrains the
``Hy`` referenced by the TM boundary row, while adjacent PMC ``xx`` constrains
the TE boundary ``Ey``. Direct PEC ``yy``/``zz`` and PMC ``zz``/``yy``
constraints likewise conflict with the SIBC ``Ey``/``Ez`` rows or their
referenced ``Hz``/``Hy`` fields. Consequently no PEC or PMC component interval
may touch an exposed impedance face. PML cells may neither overlap an
impedance interval nor touch one of its exposed interfaces. Conflicting API
calls are rejected without changing the existing boundary or PML state. The
public ``impedance_surface_mask`` property returns a copy of the opaque-cell
mask.

Solve API
---------

.. code-block:: python

   solve(sigma=None)

``sigma`` overrides the constructor ``guess`` for that solve. If both are ``None``, the automatic material-magnitude target is recomputed before calling ``eigs``.

After ``solve()``, the main outputs are:

* ``neff_TE`` and ``neff_TM``: complex effective indices for TE and TM modes.
* ``propagation_constant_TE`` and ``propagation_constant_TM``: real parts of ``neff``.
* ``attenuation_constant_TE`` and ``attenuation_constant_TM``: positive normalized attenuation ``-Im(neff)`` for passive forward modes.
* ``Ey``, ``Hx``, ``Hz``, ``Hy``, ``Ex``, ``Ez``: fields on native staggered locations.

The magnetic arrays use the solver normalization
``H_stored = -j*eta0*H_physical``. Recover physical magnetic fields with
``H_physical = j*H_stored/eta0``. This keeps the dimensionless curl equations
compact while preserving the ``exp(+j*omega*t - j*beta*z)`` Maxwell phases.

Visualization
-------------

.. code-block:: python

   visualize(mode=1, ey=True, hz=True)
   visualize_with_gui()

``visualize`` plots selected 1D field profiles. If no fields are selected, it plots all TE and TM components. Field plots include inferno material-layer backgrounds and yellow/blue/magenta PEC/PMC/SIBC layer overlays. ``visualize_with_gui`` opens an interactive mode selector.

Minimal Example
---------------

.. code-block:: python

   from fdfd_waveguide_modes import ModeSolver1D

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

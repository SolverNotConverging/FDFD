Periodic (Bloch) modes
======================

:class:`~fdfd.PeriodicModeSolver2D` solves an (x, z) cell that is invariant
in y; :class:`~fdfd.PeriodicModeSolver3D` a full (x, y, z) cell. The z range
is one period. Both return complex Bloch effective indices, which makes them
suitable for leaky-wave antennas and other periodic open structures.

Example
-------

.. code-block:: python

   import fdfd
   from fdfd import materials

   solver = fdfd.PeriodicModeSolver2D(frequency=20e9, x_range=(0, 10e-3), z_range=(0, 8e-3),
                                      boundary=materials.PEC)
   solver.add_rectangle(x_range=(0, 1.27e-3), z_range=(0, 8e-3), material=fdfd.Material(name="sub", epsilon=10.2))
   solver.add_rectangle(x_range=(1.27e-3, 1.37e-3), z_range=(1e-3, 2e-3), material=materials.PEC)
   solver.add_pml(thickness=2.5e-3, sides="x+")
   solver.mesh(resolution=(100, 80))
   result = solver.solve(num_modes=4, polarization="TM", guess=0.0, tol=1e-9)

Modes are returned nearest to ``guess`` first; ``guess=None`` uses 1.01 times
the largest material index. Leaky (fast-wave) modes have
:math:`\operatorname{Re} n_{\rm eff} < 1`, so ``guess=0`` is the usual choice
for antennas. Because each Bloch mode appears with its reciprocal partner
:math:`-n_{\rm eff}`, expect pairs. In 2D, ``polarization="both"`` solves TE
(:math:`H_x, E_y, H_z`) and TM (:math:`E_x, H_y, E_z`) and keeps the
``num_modes`` nearest overall; ``metadata["polarizations"]`` labels each mode.
For difficult 3D cells use ``method="refined"`` (see :doc:`../acceleration`).

Boundaries
----------

The 2D x ends are PEC (default) or PMC via ``boundary=``. The 3D transverse
boundary is the natural Yee closure (PMC); use conductor objects for walls.
The PML may be placed on x (and y in 3D) sides; z is periodic.
A residual check raises :class:`~fdfd.SolverError` when any eigenpair's
backward error exceeds ``max(1e-8, 100*tol)``.

Formulation
-----------

Fields are written as :math:`\mathbf F(\mathbf r) e^{-\gamma z}` with a
periodic envelope :math:`\mathbf F`. Unknowns are the transverse fields in SI
units (TE: :math:`H_x, E_y`; TM: :math:`E_x, H_y`; 3D: all four), and the
longitudinal fields are eliminated with the longitudinal Maxwell equations,
e.g. :math:`E_z = (-i/\omega\varepsilon_0)\,\varepsilon_{zz}^{-1}(D_x H_y - D_y H_x)`.
On the periodic z lattice the transverse curl equations become the pencil

.. math::

   A\,v = \gamma\,B\,v,

where :math:`A` contains the periodic z differences and the transverse terms,
and :math:`B` is the forward z average that places the :math:`\gamma` term on
the same half-cell as the difference. :math:`B` is neither Hermitian nor
positive, and is singular for an even number of cells, so the solver applies
Arnoldi to :math:`(A-\sigma B)^{-1}B` with :math:`\sigma = i k_0\,{\rm guess}`
and maps back with :math:`\gamma = \sigma + 1/\theta`. Equation rows are
ordered like the unknowns so the same mask removes constrained rows and
columns. Finally :math:`n_{\rm eff} = -i\gamma/k_0`.

Ex, Ey and Hz lie on z nodes :math:`z_0 + j\,\Delta z`; Ez, Hx and Hy on z
centres. Materials, conductor footprints and ``field_coordinates`` follow
these locations, including the periodic seam.

Comparison with FEM
-------------------

The grounded-slab antenna of ``examples/periodic/2d_surface_wave_antenna.py``
(TM, :math:`\varepsilon_r = 10.2` slab 0-1.27 mm, PEC patch 1.27-1.32 mm in x
and 1-2 mm in z, period 8 mm, PEC ground, 2.5 mm x+ PML of order 3 and
strength 5) was compared with a first-order FEM solution of the same cell. The
four roots nearest :math:`n_{\rm eff}=0` were matched by minimum complex
distance:

.. list-table::
   :header-rows: 1

   * - Geometry
     - Frequency (GHz)
     - FDFD cells (x, z)
     - FEM max. edge (µm)
     - Max. complex-neff difference
   * - Antenna
     - 18
     - 1000, 320
     - 100
     - 2.19%
   * - Antenna
     - 20
     - 1000, 320
     - 50
     - 2.05%
   * - Antenna
     - 20
     - 2000, 640
     - 50
     - 1.39%
   * - Antenna
     - 22
     - 1000, 320
     - 100
     - 1.12%
   * - Slab without patch
     - 20
     - 1000, 320
     - 100
     - 0.88%

At 20 GHz the nearest reciprocal pair is :math:`\pm(0.0145459 + 0.0932326i)`
in FEM and :math:`\pm(0.0149422 + 0.0944872i)` on the finer FDFD grid. Both
meshes still affect these values: refining the FEM edge from 350 to 50 µm
moves its root from :math:`0.013886 + 0.090630i` to
:math:`0.014546 + 0.093233i`. The more distant pair carries appreciable PML
energy and is included to compare the discrete pencils, not as a guided
mode. Repeat the comparison with
``benchmarks/periodic/leaky_wave_fem_reference.py`` (stored FEM references, no
FEM installation needed).

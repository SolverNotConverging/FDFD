Periodic antenna: FDFD and FEM
==============================

The FDFD antenna example now uses the same cell as
``fem_periodic_modes/examples/2d_leaky_wave_antenna.py``. At 18e9, 20e9, and
22e9 Hz, the four nearest complex effective indices agree with refined FEM
within 2.2% on a 1000-by-320 Yee grid. At 20e9 Hz, a 2000-by-640 grid reduces
the largest difference to 1.4%. The grounded slab without its patch agrees
within 0.9% on the 1000-by-320 grid.

What caused the disagreement
----------------------------

The previous examples represented different antennas. FDFD used a different
frequency, dielectric, slab thickness, period, patch, and PML placement.
Matching those inputs also exposed numerical problems:

* The periodic mass matrix is neither Hermitian nor positive semidefinite.
  Passing it as ``M`` to SciPy's generalized ``eigs`` violated the
  `documented requirements <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`_.
  The reported modes had residuals near 1 in the original Maxwell pencil.
* Equation rows and field columns had different component orders. Applying
  the same PEC mask to both removed the wrong equations.
* Material values used the wrong periodic half-cell offsets. Some magnetic
  field samples inside PEC were kept as though they were boundary traces.
* The outer transverse boundary and PML strength/profile differed from FEM.

The solver now applies ordinary Arnoldi to ``(A - sigma*B)^-1 B``, restores
the physical eigenvalues, and evaluates residuals in the original reduced
pencil. Equation rows follow the same component order as the unknowns.
Material values, conductor masks, and returned coordinates follow the actual
Yee lattice, including its periodic seam. The antenna uses PEC outer walls
and the same dimensionless PML profile as FEM.

Matched geometry
----------------

.. list-table:: Parameters
   :header-rows: 1

   * - Quantity
     - Value
   * - Polarization
     - TM: Ex, Ez, Hy
   * - x extent
     - 0 to 10e-3 m
   * - z period
     - 8e-3 m
   * - Substrate
     - epsilon_r = 10.2, mu_r = 1
   * - Substrate x extent
     - 0 to 1.27e-3 m, spanning the period
   * - PEC patch x extent
     - 1.27e-3 to 1.32e-3 m
   * - PEC patch z extent
     - 1e-3 to 2e-3 m
   * - Outer x faces
     - PEC; the lower face is the ground plane
   * - PML
     - x+, thickness 2.5e-3 m, order 3, sigma_max = 5
   * - Eigenvalues
     - Four nearest neff = 0, retaining both reciprocal directions

Measured agreement
------------------

Roots are matched by minimum complex distance, rather than their returned
indices. The error is ``abs(neff_FDFD - neff_FEM) / abs(neff_FEM)``. The table
shows the maximum over all four matched roots.

.. list-table:: FEM comparison
   :header-rows: 1

   * - Geometry
     - Frequency (Hz)
     - FDFD cells (x, z)
     - FEM maximum edge (m)
     - Maximum complex-neff difference
   * - Antenna
     - 18e9
     - 1000, 320
     - 100e-6
     - 2.19%
   * - Antenna
     - 20e9
     - 1000, 320
     - 50e-6
     - 2.05%
   * - Antenna
     - 20e9
     - 2000, 640
     - 50e-6
     - 1.39%
   * - Antenna
     - 22e9
     - 1000, 320
     - 100e-6
     - 1.12%
   * - Slab without patch
     - 20e9
     - 1000, 320
     - 100e-6
     - 0.88%

For example, at 20e9 Hz the nearest reciprocal pair is
``+/- (0.0145459 + 0.0932326j)`` in FEM and
``+/- (0.0149422 + 0.0944872j)`` on the finer FDFD grid. FDFD's original-pencil
residuals are below 2e-9 for that solve and below 5e-10 for the three-frequency
antenna comparison. The slab solve has residuals below 9e-8.

Complex-neff agreement measures both parts together. Small imaginary parts
can have larger relative differences: at 18e9 Hz the nearest positive-real
root is ``0.417986 + 0.025202j`` in FEM and ``0.409565 + 0.028814j`` in FDFD.
Both meshes still affect these values. At 20e9 Hz, refining the FEM maximum
edge from 350e-6 to 175e-6, 100e-6, and 50e-6 m changes the nearest positive
root from ``0.013886 + 0.090630j`` to ``0.014110 + 0.091817j``,
``0.014348 + 0.092604j``, and ``0.014546 + 0.093233j``.

The more distant pair has appreciable PML energy. It is included to compare
the two discrete pencils, without classifying it as a guided antenna mode.
The low algebraic residual confirms an eigenpair solves its discrete equations;
mesh refinement measures its spatial accuracy.

Repeat the comparison
---------------------

The benchmark stores FEM references and requires only the FDFD dependencies:

.. code-block:: console

   python benchmarks/periodic/leaky_wave_fem_reference.py --check
   python benchmarks/periodic/leaky_wave_fem_reference.py --case slab --frequencies 20e9 --check
   python benchmarks/periodic/leaky_wave_fem_reference.py --frequencies 20e9 --grids 1000x320 2000x640 --check

CSV data and a figure are written under
``fdfd_periodic_modes/outputs/benchmarks/leaky_wave_fem_reference/``.
The tolerance is 3% complex-neff difference at every selected frequency on
the finest grid, with original-pencil residuals below 1e-6.

FEM references were generated with first-order triangles using the local FEM
sources at revision ``9c70ad81``. To regenerate the antenna reference in that
project, run the following with its dependencies installed:

.. code-block:: python

   from fem_periodic_modes import Material, PeriodicModeSolver2D, materials

   solver = PeriodicModeSolver2D(
       frequency=20e9, x_range=(0., 10e-3), z_range=(0., 8e-3),
       polarization="TM", boundary=materials.PEC,
   )
   solver.add_rectangle(
       x_range=(0., 1.27e-3), z_range=(0., 8e-3),
       material=Material(name="substrate", epsilon=10.2),
   )
   solver.add_rectangle(
       x_range=(1.27e-3, 1.32e-3), z_range=(1e-3, 2e-3),
       material=materials.PEC,
   )
   solver.add_pml(thickness=2.5e-3, direction="x+", order=3, sigma_max=5.)
   solver.mesh(max_element_size=50e-6)
   result = solver.solve(
       num_modes=4, neff_guess=0., direction="all", eigensolver="auto",
       max_pml_fraction=None, max_refinements=0,
   )
   print(result.neff)

Use 100e-6 m maximum edge for the 18e9 and 22e9 Hz references. Omit the patch
for the slab reference at 20e9 Hz, also using a 100e-6 m maximum edge.

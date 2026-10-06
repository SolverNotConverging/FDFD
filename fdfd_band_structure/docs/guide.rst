FDFD Band Structure Solver
==========================

``BandStructureSolver2D`` computes TE and TM eigenfrequencies along a physical
Bloch-vector path. Frequency is an output, so construction specifies only the
unit-cell ranges and background material. Phasors use ``exp(+i*omega*t)``.

Material-first workflow
-----------------------

.. code-block:: python

   import numpy as np
   from fdfd_band_structure import BandStructureSolver2D, load_result, Material

   rod = Material(name="dielectric rod", epsilon=8.9)
   solver = BandStructureSolver2D(x_range=1e-3, y_range=1e-3)
   solver.add_circle(center=(0.5e-3, 0.5e-3), radius=0.2e-3, material=rod)
   solver.mesh(resolution=(32, 32))

   path = solver.make_bloch_path(
       points=((0.0, 0.0), (np.pi / 1e-3, 0.0),
               (np.pi / 1e-3, np.pi / 1e-3), (0.0, 0.0)),
       num_points=40,
   )
   result = solver.solve(beta_path=path, num_modes=5,
                         polarizations=("TE", "TM"))
   result.save("fdfd_band_structure/outputs/bands.h5")
   loaded = load_result("fdfd_band_structure/outputs/bands.h5")
   loaded.plot(component="TE", quantity="real", mode=None)

The generic geometry API accepts compatible shared shapes; rectangle, circle,
and polygon convenience methods use physical coordinates. The current backend
supports isotropic bulk ``epsilon`` and ``mu``. Conductor, SIBC, dispersive, and
anisotropic assignments are rejected explicitly.

``make_bloch_path()`` distributes samples over a polyline of wavevectors in
radians per metre. ``solve()`` returns frequency arrays in hertz indexed by
polarization, raw eigenvalues, normalized frequencies, solver metadata, and
plot/show/save operations. Loading never solves again.

The dedicated band viewer displays all bands together. Its controls select
TE/TM polarizations, real part, imaginary part, or magnitude, and frequency
in GHz or normalized ``fa/c``.

The operator uses separate Yee sites: Ex and Hy are at x cell centres / y
nodes, Ey and Hx at x nodes / y cell centres, Ez at nodes in both directions,
and Hz at cell centres in both directions. Periodicity gives each lattice
``Nx * Ny`` distinct samples; equal array sizes do not put the fields at the
same positions. Each material tensor is sampled at its corresponding site.
Lossy complex materials use an ordinary eigenproblem after diagonal material
inversion, so they do not require a Hermitian generalized mass matrix.

Examples and API
----------------

Run the `square-lattice example <../examples/2d_square_lattice.py>`_
and see the `family example index <../examples/README.rst>`_.
The curated signatures are in `API_REFERENCE.rst <API_REFERENCE.rst>`_.

FDFD Periodic Mode Solvers
==========================

``PeriodicModeSolver2D`` solves scalar TE or TM periodic envelopes on an x/z
cell. ``PeriodicModeSolver3D`` solves full-vector envelopes on x/y/z cells. The
z axis is periodic; x and y may use physical PML regions. Phasors use
``exp(+i*omega*t)`` and guided propagation uses ``exp(-i*beta*z)``.

Material-first workflow
-----------------------

.. code-block:: python

   from fdfd_periodic_modes import PeriodicModeSolver2D, load_result, Material, materials

   substrate = Material(name="substrate", epsilon=10.2)
   solver = PeriodicModeSolver2D(
       frequency=20e9,
       x_range=(0.0, 10e-3),
       z_range=(0.0, 8e-3),
       polarization="TM",
       background_material=materials.air,
       boundary=materials.PEC,
   )
   solver.add_rectangle(
       x_range=(0.0, 1.27e-3),
       z_range=solver.z_range,
       material=substrate,
       name="slab",
   )
   solver.add_pml(thickness=2.5e-3, direction="x+")
   solver.mesh(resolution=(40, 32))
   result = solver.solve(num_modes=4, neff_guess=0.5)
   result.save("fdfd_periodic_modes/outputs/fdfd_periodic.h5")
   loaded = load_result("fdfd_periodic_modes/outputs/fdfd_periodic.h5")
   loaded.plot(component="Hy", quantity="magnitude", mode=1)

For 2D, ``boundary=materials.PEC`` is the default at both x faces; use
``materials.PMC`` explicitly for magnetic walls. The upper PEC face may sit
behind an x+ PML. ``sigma_max`` is a dimensionless coordinate-stretch strength,
matching the FEM periodic solver; it does not scale with frequency.

Both eigensolvers work on the full periodic pencil without assuming a positive
mass matrix. ``result.solve_info['residuals']`` reports the relative residual
of each eigenpair in the original constrained Maxwell equations.

The 3D class uses ``add_box()``, ``add_sphere()``, and ``add_cylinder()``
convenience methods. Both dimensions also accept compatible shared shapes through
``add_geometry()``. Bulk materials may be scalar or diagonal. PEC and PMC are
assigned to geometry as material presets; SIBC is not supported in this family.

``mesh()`` accepts Yee-cell ``resolution`` or a physical
``max_element_size``. ``solve()`` returns periodic envelopes with explicit
staggered coordinates. It exposes ``neff``, ``beta``, fields, residual metadata,
plotting, interactive viewing, and atomic HDF5 persistence. Geometry edits
invalidate mesh and result while retaining explicit meshing settings.

The 2D result includes all six components: ``Ex``, ``Ey``, ``Ez``, ``Hx``,
``Hy``, and ``Hz``. TM modes have ``Ex``, ``Ez``, and ``Hy``; TE modes have
``Ey``, ``Hx``, and ``Hz``. The other components are zero for that polarization.
The longitudinal fields are reconstructed from Maxwell's equations on their
staggered grids. In 2D plots, ``z`` is horizontal and ``x`` is vertical.

Ex, Ey, and Hz use z nodes ``z0 + j*dz``; Ez, Hx, and Hy use z centres
``z0 + (j+0.5)*dz``. Both periodic lattices have Nz samples, with the upper
node identified with the lower one. In 3D, Ex uses x centres / y nodes, Ey
uses x nodes / y centres, Ez uses x/y nodes, Hx uses x nodes / y centres,
Hy uses x centres / y nodes, and Hz uses x/y centres. Materials and conductor
constraints follow these locations. Saved fields retain all native samples;
use ``result.field_coordinates[name]`` for each component's physical axes.

The 2D viewer shows the three active TE or TM components together with the
material geometry in the background. The 3D viewer shows all six components.
Select a mode from the dropdown and magnitude, real part, imaginary part, or phase. Mode numbers
start at 1. The title displays the selected mode's complex ``neff``.
The 3D viewer also selects an xy, xz, or yz plane and its physical slice position.
Both dimensions retain material backgrounds in saved results.

For frequency sweeps, import ``plot_dispersion`` from ``fdfd_periodic_modes``
and call ``plot_dispersion(frequencies, neff_sweep)`` after the loop. Frequencies
are in hertz; append ``result.neff`` to ``neff_sweep`` at each frequency.
The figure shows real and imaginary ``neff`` on two panels using unconnected
scatter markers. Mode numbers refer to the returned order at each frequency;
points are not joined because modes can exchange places. Use ``show=False``
to return the figure for saving without opening it.

Examples and API
----------------

Run `2d_surface_wave_antenna.py <../examples/2d_surface_wave_antenna.py>`_
or the `3D image-guide example <../examples/3d_image_guide_leaky_wave_antenna.py>`_.
The `family example index <../examples/README.rst>`_
lists dispersion and postprocessing scripts. See
`API_REFERENCE.rst <API_REFERENCE.rst>`_ for the curated user surface.

The `FEM comparison <fem_comparison.rst>`_ records the matched antenna geometry,
frequency comparisons, numerical residuals, and a runnable benchmark.

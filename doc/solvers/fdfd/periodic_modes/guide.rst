FDFD Periodic Mode Solvers
==========================

``PeriodicModeSolver2D`` solves scalar TE or TM periodic envelopes on an x/z
cell. ``PeriodicModeSolver3D`` solves full-vector envelopes on x/y/z cells. The
z axis is periodic; x and y may use physical PML regions. Phasors use
``exp(+i*omega*t)`` and guided propagation uses ``exp(-i*beta*z)``.

Material-first workflow
-----------------------

.. code-block:: python

   from cem_common import Material, materials
   from fdfd_periodic_modes import PeriodicModeSolver2D, load_result

   substrate = Material(name="substrate", epsilon=4.0)
   solver = PeriodicModeSolver2D(
       frequency=20e9,
       x_range=(0.0, 10e-3),
       z_range=(0.0, 8e-3),
       polarization="TM",
       background_material=materials.air,
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
   result.save("outputs/fdfd_periodic.h5")
   loaded = load_result("outputs/fdfd_periodic.h5")
   loaded.plot(component="Hy", quantity="magnitude", mode=0)

The 3D class uses ``add_box()``, ``add_sphere()``, and ``add_cylinder()``
convenience methods. Both dimensions also accept compatible shared shapes through
``add_geometry()``. Bulk materials may be scalar or diagonal. PEC and PMC are
assigned to geometry as material presets; SIBC is not supported in this family.

``mesh()`` accepts Yee-cell ``resolution`` or a physical
``max_element_size``. ``solve()`` returns periodic envelopes with explicit
staggered coordinates. It exposes ``neff``, ``beta``, fields, residual metadata,
plotting, interactive viewing, and atomic HDF5 persistence. Geometry edits
invalidate mesh and result while retaining explicit meshing settings.

Examples and API
----------------

Run `surface_wave_antenna_2d.py <../../../../examples/fdfd/periodic_modes/surface_wave_antenna_2d.py>`_
or the `3D image-guide example <../../../../examples/fdfd/periodic_modes/image_guide_leaky_wave_antenna_3d.py>`_.
The `family example index <../../../../examples/fdfd/periodic_modes/README.rst>`_
lists dispersion and postprocessing scripts. See
`API_REFERENCE.rst <API_REFERENCE.rst>`_ for the curated user surface.

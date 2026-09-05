FDFD Waveguide Mode Solvers
===========================

``ModeSolver1D`` solves stratified guides and ``ModeSolver2D`` solves full-vector
cross sections on staggered Yee grids. Lengths use metres, frequency uses hertz,
and Python mode indices are zero-based. Electromagnetic phasors use
``exp(+i*omega*t)`` with propagation ``exp(-i*beta*z)``.

Material-first workflow
-----------------------

Define reusable material and shape objects before assigning geometry. The
following slab example uses an explicit mesh and saves a result without opening
a window:

.. code-block:: python

   from cem_common import Material, materials, shapes
   from fdfd_waveguide_modes import ModeSolver1D, load_result

   dielectric = Material(name="dielectric", epsilon=2.25)
   solver = ModeSolver1D(
       frequency=10e9,
       x_range=(-5e-3, 5e-3),
       background_material=materials.air,
   )
   solver.add_layer(x_range=(-1e-3, 1e-3), material=dielectric, name="core")
   solver.add_geometry(
       shape=shapes.Interval(bounds=(-5e-3, -4.5e-3)),
       material=materials.PEC,
       name="left wall",
   )
   solver.mesh(max_element_size=0.25e-3)
   result = solver.solve(num_modes=2, neff_guess=1.2)
   result.save("outputs/fdfd_modes.h5")
   loaded = load_result("outputs/fdfd_modes.h5")
   figure = loaded.plot(component="Ey", quantity="magnitude", mode=0)

``solve()`` meshes automatically if needed. Geometry edits through
``set_shape()``, ``set_material()``, or ``remove()`` invalidate the mesh and
result; automatic remeshing reuses the last explicit settings. ``solver.show()``
requires a completed solve. ``result.plot()`` returns a Matplotlib figure and
``result.show(block=True)`` opens the interactive viewer.

Materials, conductors, and shapes
---------------------------------

Bulk media support scalar or diagonal relative ``epsilon`` and ``mu``. Assign
``materials.PEC`` or ``materials.PMC`` to opaque shapes. The waveguide family
also accepts ``SurfaceImpedance`` and good-conductor presets such as
``materials.copper``; these exclude the conductor interior and apply a scalar
Leontovich boundary. A passive bulk material has nonpositive imaginary
constitutive values, while passive SIBC impedance has nonnegative real part.

The generic ``add_geometry(shape=..., material=...)`` accepts compatible shared
primitives, Boolean combinations, and transformed shapes. Convenience methods
cover layers, rectangles, circles, and polygons. Overlapping conductor cells and
unresolved subcell conductors raise explicit geometry errors. ``add_pml()`` uses
physical ``thickness``, axis ``direction``, polynomial ``order``, and optional
``sigma_max``.

Examples and API
----------------

Start with `rectangular_waveguide_2d.py <../../../../examples/fdfd/waveguide_modes/rectangular_waveguide_2d.py>`_.
The `family example index <../../../../examples/fdfd/waveguide_modes/README.rst>`_
then covers slab, microstrip, dielectric, dispersion, and postprocessing cases.
The `rectangular-waveguide benchmark <../../../../benchmarks/analytical/rectangular_waveguide_modes.py>`_
compares FDFD and FEM against TE10 theory.

See `API_REFERENCE.rst <API_REFERENCE.rst>`_ for supported signatures, defaults,
return values, and actionable exceptions.

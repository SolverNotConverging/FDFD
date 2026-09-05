FDFD Scattering Solver
======================

``ScatteringSolver2D`` solves scalar TE or TM total-field/scattered-field
problems on a two-dimensional Yee grid. Frequency is in hertz, coordinates are
in metres, source angles are measured in degrees from physical +x, and fields
use ``exp(+i*omega*t)``.

Material-first workflow
-----------------------

.. code-block:: python

   from cem_common import Material
   from fdfd_scattering import ScatteringSolver2D, load_result

   dielectric = Material(name="cylinder", epsilon=4.0)
   solver = ScatteringSolver2D(
       frequency=10e9,
       x_range=(-0.05, 0.05),
       y_range=(-0.05, 0.05),
       polarization="TE",
   )
   solver.add_circle(center=(0.0, 0.0), radius=0.01, material=dielectric)
   solver.add_pml(thickness=0.01, direction="all")
   solver.mesh(max_element_size=2.5e-3)
   solver.add_source(kind="plane_wave", angle=0.0)
   solver.set_source_region(inset=0.015)
   result = solver.solve()
   result.save("outputs/fdfd_scattering.h5")
   loaded = load_result("outputs/fdfd_scattering.h5")
   loaded.plot(component="Ez", quantity="magnitude")

Define the source and rectangular total-field region before solving. A point
source instead uses ``kind="point"`` and a physical ``location=(x, y)``.
The current scalar backend supports isotropic bulk materials. PEC, PMC, SIBC,
and anisotropic scattering objects raise explicit capability errors.

``mesh()`` accepts cell ``resolution`` or physical ``max_element_size``.
Geometry edits invalidate mesh and result while retaining the last explicit
mesh settings. ``solve()`` neither opens a window nor saves a file. Returned
fields carry their physical staggered coordinates and support static plotting,
interactive display, atomic saving, and loading without rerunning the solver.

Examples and API
----------------

The runnable `dielectric-cylinder example <../../../../examples/fdfd/scattering/dielectric_cylinder_2d.py>`_
shows the complete workflow. See `API_REFERENCE.rst <API_REFERENCE.rst>`_ for
supported signatures, defaults, and errors.

FDFD Scattering Solver
======================

``ScatteringSolver2D`` solves scalar TE or TM total-field/scattered-field
problems on a two-dimensional Yee grid. Frequency is in hertz, coordinates are
in metres, source angles are measured in degrees from physical +x, and fields
use ``exp(+i*omega*t)``.

Material-first workflow
-----------------------

The interactive viewer displays all three TE/TM fields together with magnitude,
real, imaginary, and phase options. A scattering result has no mode selector.

.. code-block:: python

   from fdfd_scattering import ScatteringSolver2D, load_result, Material

   dielectric = Material(name="cylinder", epsilon=4.0)
   solver = ScatteringSolver2D(
       frequency=10e9,
       x_range=(-50e-3, 50e-3),
       y_range=(-50e-3, 50e-3),
       polarization="TE",
   )
   solver.add_circle(center=(0.0, 0.0), radius=10e-3, material=dielectric)
   solver.add_pml(thickness=10e-3, direction="all")
   solver.mesh(max_element_size=2.5e-3)
   solver.add_source(kind="plane_wave", angle=0.0)
   solver.set_source_region(inset=15e-3)
   result = solver.solve()
   result.save("fdfd_scattering/outputs/fdfd_scattering.h5")
   loaded = load_result("fdfd_scattering/outputs/fdfd_scattering.h5")
   loaded.show()

Define the source and rectangular total-field region before solving. A point
source instead uses ``kind="point"`` and a physical ``location=(x, y)``.
The solver supports scalar or diagonal bulk materials and PEC objects.
Use ``material=materials.PEC`` for a perfectly conducting scatterer. PEC
electric fields are constrained on their Yee traces; TM enforces zero
tangential electric field at the metal surface. Put the whole PEC scatterer
inside the total-field source region. PMC and SIBC objects are unsupported.

``mesh()`` accepts cell ``resolution`` or physical ``max_element_size``.
Geometry edits invalidate mesh and result while retaining the last explicit
mesh settings. ``solve()`` neither opens a window nor saves a file. Returned
fields carry their physical staggered coordinates and support static plotting,
interactive display, atomic saving, and loading without rerunning the solver.
TE results contain ``Ez``, ``Hx``, and ``Hy``; TM results contain ``Ex``, ``Ey``,
and ``Hz``. ``result.show()`` displays all three fields together with material
geometry and a magnitude/real/imaginary/phase control. There is no mode selector.

Yee field locations
-------------------

For ``resolution=(Nx, Ny)``, nodes include both domain ends and cells use
half-cell centres. The last array axis is the single result index.

.. list-table:: Complete component lattices
   :header-rows: 1

   * - Polarization
     - Field
     - x / y locations
     - Spatial shape
   * - TE
     - Ez
     - node / node
     - (Nx+1, Ny+1)
   * - TE
     - Hx
     - node / cell
     - (Nx+1, Ny)
   * - TE
     - Hy
     - cell / node
     - (Nx, Ny+1)
   * - TM
     - Hz
     - cell / cell
     - (Nx, Ny)
   * - TM
     - Ex
     - cell / node
     - (Nx, Ny+1)
   * - TM
     - Ey
     - node / cell
     - (Nx+1, Ny)

Materials and the total-field rectangle are evaluated on each component's
lattice. Derivatives map between those lattices; fields retain their native
samples in saved results and plots. Use ``result.field_coordinates[name]``
when working with field arrays. The outer boundary is PEC, normally behind PML.

Examples and API
----------------

The runnable `dielectric-cylinder example <../examples/2d_dielectric_cylinder.py>`_
and `PEC-cylinder example <../examples/2d_pec_cylinder.py>`_
show the complete workflow. See `API_REFERENCE.rst <API_REFERENCE.rst>`_ for
supported signatures, defaults, and errors.

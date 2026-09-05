cem_common user API
===================

This reference lists the deliberately supported shared values for version 1.0.0.

Materials
---------

``Material(*, name='material', epsilon=1.0, mu=1.0)``
   Immutable bulk material. ``epsilon`` and ``mu`` are relative scalar,
   diagonal, or square tensor values. Solver backends validate supported tensor
   forms. Returns a reusable ``Material``.

``SurfaceImpedance(*, impedance, name='surface impedance')``
   Constant passive scalar surface impedance in ohms. ``impedance`` is required,
   finite, nonzero, and has nonnegative real part. ``at_frequency(frequency=...)``
   returns its complex impedance.

``GoodConductor(*, name, conductivity, mu=1.0)``
   Frequency-dependent good-conductor SIBC. ``conductivity`` is required in S/m;
   ``mu`` is positive relative permeability. ``at_frequency(frequency=...)``
   returns surface impedance in ohms.

``materials`` exports ``vacuum``, ``air``, ``PEC``, ``PMC``, and the metal SIBC
presets ``aluminium``, ``copper``, ``gold``, ``molybdenum``, ``palladium``,
``silver``, ``tungsten``, and ``zinc``. ``materials.SpatialMaterial`` is the
named callback value accepted by FEM waveguide scattering.

Shapes
------

All constructors are keyword-only and all coordinates and lengths use metres.

.. list-table:: Shared shape constructors
   :header-rows: 1

   * - Constructor
     - Required arguments
     - Meaning
   * - ``shapes.Interval``
     - ``bounds=(min, max)``
     - One-dimensional interval.
   * - ``shapes.Rectangle`` / ``shapes.Box``
     - ``bounds=((min, max), ...)``
     - Axis-aligned 2D / 3D region.
   * - ``shapes.RoundedRectangle``
     - ``bounds``, ``radius``
     - Rectangle with positive corner radius.
   * - ``shapes.Circle`` / ``shapes.Sphere``
     - ``center``, ``radius``
     - 2D / 3D round region.
   * - ``shapes.Annulus``
     - ``center``, ``inner_radius``, ``outer_radius``
     - Connected ring.
   * - ``shapes.Ellipse`` / ``shapes.Ellipsoid``
     - ``center``, ``radii``
     - Axis-aligned elliptical region.
   * - ``shapes.Polygon``
     - ``points``
     - Nonzero-area 2D polygon.
   * - ``shapes.Cylinder``
     - ``center``, ``radius``, ``z_range``
     - Circular 3D extrusion.
   * - ``shapes.Extrusion``
     - ``shape``, ``z_range``
     - Extruded 2D cross section.
   * - ``shapes.Segment``
     - ``start``, ``end``
     - Zero-thickness 2D boundary where supported.
   * - ``shapes.Union`` / ``shapes.Intersection``
     - ``shapes=(...)``
     - Boolean combination of same-dimensional shapes.
   * - ``shapes.Difference``
     - ``shape``, ``tool``
     - Boolean subtraction.

Every shape provides read-only ``bounds`` and ``dimension``. Use
``translated(offset=...)`` and ``rotated(angle=..., center=..., axis=...)`` to
create transformed shapes.

Errors and mesh snapshots
-------------------------

``CEMError`` is the common base class. Public subclasses are
``ConfigurationError``, ``GeometryError``, ``BackendCapabilityError``,
``MeshError``, ``SolverError``, ``NoResultError``, ``PersistenceError``, and
``ViewerError``.

``MeshSnapshot`` exposes read-only ``coordinates``, zero-based ``elements``,
physical ``axes``, ``info``, and ``metadata`` on stored FEM results. Users
normally receive it through ``result.mesh_data`` rather than constructing it.

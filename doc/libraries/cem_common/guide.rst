Shared materials, geometry, and errors
======================================

``cem_common`` provides the values shared by every Python solver. Numerical
assembly remains in each solver package. Define materials and shapes once, then
assign them through a solver's ``add_geometry`` or convenience methods.

.. code-block:: python

   from cem_common import Material, materials, shapes

   substrate = Material(
       name="microwave substrate",
       epsilon=3.55 * (1.0 - 0.0027j),
       mu=1.0,
   )
   core = shapes.RoundedRectangle(
       bounds=((-1e-3, 1e-3), (-0.5e-3, 0.5e-3)),
       radius=0.1e-3,
   )

   solver.add_geometry(shape=core, material=substrate, name="core")
   solver.add_geometry(shape=metal_shape, material=materials.copper,
                       name="copper")

Material values
---------------

``Material`` stores named relative ``epsilon`` and ``mu``. Scalars are
isotropic; one-dimensional sequences are diagonal tensors; square arrays are
full tensors. Each solver validates its supported form without silently dropping
components. Electromagnetic results use ``exp(+i*omega*t)``, so passive bulk
materials have nonpositive imaginary constitutive values.

``materials.vacuum`` and ``materials.air`` are lossless bulk presets.
``materials.PEC`` and ``materials.PMC`` are ideal boundary assignments. Metal
presets including ``materials.copper``, ``materials.aluminium``,
``materials.gold``, and ``materials.silver`` are frequency-dependent
good-conductor SIBC models. A solver that does not support the chosen material
form raises ``BackendCapabilityError``.

Geometry values
---------------

All geometry coordinates are in metres. The available primitives are
``Interval``, ``Rectangle``, ``RoundedRectangle``, ``Circle``, ``Annulus``,
``Ellipse``, ``Polygon``, ``Box``, ``Sphere``, ``Cylinder``, ``Ellipsoid``,
``Extrusion``, and ``Segment``. ``Union``, ``Intersection``, and ``Difference``
compose shapes. ``translated()`` and ``rotated()`` return transformed immutable
shapes. A solver checks dimensions and backend capabilities when geometry is
assigned.

Geometry methods return solver-owned handles. Read ``name``, ``shape``,
``material``, ``clip``, and ``background`` from a live handle. Use
``set_material(geometry=...)``, ``set_shape(geometry=...)``, or
``remove(geometry=...)`` to edit the model and trigger lifecycle invalidation.

Errors and result metadata
--------------------------

Applications may catch ``CEMError`` across solver families. Its public
subclasses distinguish configuration, geometry, backend capability, meshing,
solving, missing results, persistence, and viewer failures. ``MeshSnapshot`` is
the common read-only mesh view stored on completed FEM results.

See `API_REFERENCE.rst <API_REFERENCE.rst>`_ for supported constructors and
units. Solver-specific capabilities are documented in each solver guide.

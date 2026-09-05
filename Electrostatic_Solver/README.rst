Electrostatic FEM Solver
========================

``ElectrostaticSolver`` is a geometry-first 1D/2D finite-element solver for

.. math::

   -\nabla \cdot (\epsilon_0 \boldsymbol{\epsilon}_r \nabla \phi) = \rho.

It uses Gmsh for interface-conforming, locally refined line/triangle meshes and
scikit-fem P1 elements for sparse assembly.  It is not an FDFD solver.

The `API reference <API_REFERENCE.rst>`_ documents every public constructor,
method, property, and module helper with argument tables showing required or
optional status, expected types, defaults, and explanations.

All examples and tutorials use ``max_refinements=0`` for a reproducible
single-mesh solve. This does not change the library's adaptive defaults.

Simple capacitor tutorial
-------------------------

.. code-block:: python

   from Electrostatic_Solver import ElectrostaticSolver

   solver = ElectrostaticSolver(dim=1, domain=(0.0, 1e-3), outer_potential=None)
   solver.set_potential("left", 0.0)
   solver.set_potential("right", 1.0)
   solver.discretize(max_element_size=0.1e-3)
   result = solver.solve(max_refinements=0)
   print(result.energy)
   print(result.adaptive_history)
   solver.visualize()

This produces a linear potential and a uniform electric field. Use
``result.coordinates`` to locate the nodal values and
``result.element_electric_field`` for unsmoothed element fields.

Runnable cases
--------------

Run these modules from the repository root:

* ``python -m Electrostatic_Solver.examples.capacitor_1d``: basic capacitor.
* ``python -m Electrostatic_Solver.examples.anisotropic_2d``: tensor dielectric.
* ``python -m Electrostatic_Solver.examples.layered_and_charged``: layered
  dielectric energy check and a charged 2D domain with an analytic potential.

Every example opens Matplotlib GUI plots; the layered/charged example opens
both results together. Use an interactive Matplotlib backend for desktop runs.

Breaking 1.0 Change
-------------------

Version 1.0 replaces the old Cartesian relaxation implementation.  Geometry is
now complete before discretization:

1. construct a physical domain;
2. add dielectric, charge, and fixed-potential geometry;
3. call ``discretize()``;
4. call ``solve()``.

``solve()`` auto-discretizes for compatibility, but explicit ``discretize()``
makes the geometry/mesh boundary clear.  ``potential`` is now a vector on the
unstructured mesh nodes, not an ndarray with the constructor's grid shape.
Use ``coordinates`` and ``elements`` to interpret it.

Installation
------------

From the repository root:

.. code-block:: console

   python -m pip install -e "Electrostatic_Solver[test]"

Continuous Geometry
-------------------

The public primitives are ``Interval`` in 1D and ``Rectangle``, ``Circle``, and
``Polygon`` in 2D.  Coordinates are physical values; SI metres are recommended.

.. code-block:: python

   from Electrostatic_Solver import ElectrostaticSolver, Rectangle

   solver = ElectrostaticSolver(
       dim=2,
       domain=((0.0, 10e-3), (0.0, 5e-3)),
       outer_potential=None,
   )
   solver.add_object(
       Rectangle((4e-3, 6e-3), (1e-3, 4e-3)),
       erxx=10.2,
       eryy=9.8,
       name="ceramic",
   )
   solver.set_potential("left", 0.0, name="ground")
   solver.set_potential("right", 5.0, name="drive")

Later overlapping material and charge regions take precedence.  Permittivity
may be isotropic, diagonal, or a full real symmetric positive-definite tensor:

.. code-block:: python

   solver.add_object(shape, permittivity=((4.0, 0.3), (0.3, 2.0)))

For compatibility, ``add_object(shape, erxx=4.0)`` changes only the x entry and
leaves ``eryy=1.0``.  Use ``permittivity=4.0`` for an isotropic 2D dielectric.

Charge and Boundary Conditions
------------------------------

``set_potential(shape, value, name=...)`` imposes a strong Dirichlet condition
on every node inside a conductor shape.  Exterior selectors are ``left`` and
``right`` in 1D, plus ``bottom``, ``top``, and ``outer`` in 2D.  The complete
outer boundary is zero volts by default; pass ``outer_potential=None`` to leave
unselected exterior facets with the natural zero-flux condition.

``add_charge_density(shape, density)`` adds piecewise-constant free volume
charge in C/m3.  In 2D, energy and charge are reported per unit out-of-plane
depth.

Meshing and Refinement
----------------------

.. code-block:: python

   mesh = solver.discretize(
       max_element_size=0.5e-3,
       material_aware=True,
       interface_refinement=0.7,
       boundary_refinement=0.5,
   )

Gmsh fragments the domain at every material, conductor, and charge shape before
generating elements.  With ``material_aware=True`` (the default), each material
uses

.. math::

   h_m = h_\max \frac{\min_j\sqrt{\lambda_\max(\epsilon_{r,j})}}
                         {\sqrt{\lambda_\max(\epsilon_{r,m})}},

clipped to ``h_max``.  High-Dk regions therefore receive smaller elements.
``interface_refinement`` scales size again at true dielectric jumps, while
``boundary_refinement`` uses Gmsh distance fields at the exterior and internal
fixed-potential boundaries.  Set either factor to ``None`` to disable it;
optional ``*_refinement_width`` values control grading distance in physical
units.

Each ``solve()`` also performs solution-driven local refinement by default.
The estimator combines jumps in normal displacement with the Poisson volume
residual, so a physical electric-field jump at a dielectric interface alone
does not cause refinement.  Marked cells are bisected conformingly and inherit
their parent's material.  The defaults allow two refinement passes, stop at a
relative flux indicator of 0.05, and cap the final mesh at 200,000 elements.

.. code-block:: python

   result = solver.solve(max_refinements=0, adaptive_tolerance=0.02,
                         marking_fraction=0.5, max_elements=100_000)
   print(result.adaptive_history)
   fixed_mesh_result = solver.solve(max_refinements=0, adaptive=False)

``marking_fraction`` selects the smallest set of cells accounting for that
fraction of the squared indicator.  History records the mesh sizes, indicators,
and whether tolerance, refinement count, or element budget stopped adaptation.
This indicator is a refinement heuristic, not a certified solution-error bound.
Use the returned result's mesh when indexing fields; adaptation may replace
the mesh originally returned by ``discretize()``.

Solving and Results
-------------------

.. code-block:: python

   result = solver.solve(max_refinements=0)
   phi = result.potential
   ex, ey = solver.compute_electric_field()

``ElectrostaticResult`` exposes:

* ``coordinates`` and ``elements``: unstructured mesh topology;
* ``potential``: nodal voltage;
* ``electric_field`` and ``displacement_field``: measure-weighted nodal fields;
* ``element_electric_field`` and ``element_displacement_field``: unaveraged
  fields of shape ``(elements, dimension)``, preserving dielectric jumps;
* ``adaptive_history``: per-pass error indicators and stopping reason;
* ``energy``: electrostatic energy (per unit depth in 2D);
* ``reaction`` and ``conductor_charges``: Dirichlet reaction charge;
* ``residual_norm``: relative residual on unconstrained degrees of freedom.

The solve is a sparse direct P1 FEM solve.  Legacy ``tol`` and ``max_iter``
arguments are accepted, but ``max_iter`` no longer controls relaxation.

Compatibility Surface
---------------------

The familiar calls still exist:

.. code-block:: python

   solver = ElectrostaticSolver(mesh_size=(50, 50), dim=2)
   solver.add_object((slice(10, 20), slice(15, 35)), erxx=7, eryy=3)
   solver.set_potential((slice(24, 26), slice(10, 40)), 100)
   solver.solve(max_refinements=0)

Here slices are converted to physical rectangles first; ``mesh_size`` supplies
the implicit domain and target sizing, rather than allocating a finite-
difference grid.  New code should use explicit physical domains and primitives.

Examples and Tests
------------------

.. code-block:: console

   python -m Electrostatic_Solver.examples.capacitor_1d
   python -m Electrostatic_Solver.examples.anisotropic_2d
   pytest Electrostatic_Solver/tests

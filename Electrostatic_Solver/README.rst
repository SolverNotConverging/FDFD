Electrostatic Solver
====================

``ElectrostaticSolver`` solves simple one-dimensional and two-dimensional electrostatic potential problems with fixed-potential constraints and spatially varying permittivity.

What It Solves
--------------

Use this solver for quick static-potential problems, capacitor-like geometries, and educational finite-difference relaxation examples. It computes electric potential and derives electric field from the negative potential gradient.

Supported features:

* 1D and 2D grids.
* Fixed-potential regions.
* Spatially varying ``erxx`` and, in 2D, ``eryy``.
* Iterative relaxation solve.
* Potential and electric-field visualization.

Main Class
----------

.. code-block:: python

   ElectrostaticSolver(mesh_size, dim=2)

Parameters:

* ``mesh_size``: ``(nx,)`` for 1D or ``(nx, ny)`` for 2D.
* ``dim``: ``1`` or ``2``.

Boundary Conditions
-------------------

The outer boundary is fixed to zero potential by default. Additional fixed-potential regions can be added with ``set_potential``.

Setup API
---------

.. code-block:: python

   set_potential(region, potential_value)
   add_object(region, erxx=1.0, eryy=None)

Notes:

* ``region`` is a slice for 1D or a tuple of slices for 2D.
* ``eryy`` is only valid for 2D problems.
* ``fixed_mask`` stores which cells are constrained.

Solve API
---------

.. code-block:: python

   solve(tol=1e-8, max_iter=100000)
   compute_electric_field()

Outputs:

* ``potential``: solved potential array.
* ``compute_electric_field()`` returns ``Ex`` in 1D or ``(Ex, Ey)`` in 2D.

Visualization
-------------

.. code-block:: python

   visualize()

In 1D, visualization plots potential and electric field. In 2D, it plots potential contours and a quiver plot coloured by field magnitude.

1D Example
----------

.. code-block:: python

   from Electrostatic_Solver import ElectrostaticSolver

   solver = ElectrostaticSolver(mesh_size=(100,), dim=1)
   solver.set_potential(slice(59, 60), potential_value=10)
   solver.set_potential(slice(10, 20), potential_value=-10)
   solver.add_object(slice(30, 40), erxx=2)
   solver.solve()
   solver.visualize()

2D Example
----------

.. code-block:: python

   from Electrostatic_Solver import ElectrostaticSolver

   solver = ElectrostaticSolver(mesh_size=(50, 50), dim=2)
   solver.set_potential((slice(0, 50), slice(0, 1)), potential_value=-30)
   solver.set_potential((slice(0, 25), slice(30, 31)), potential_value=100)
   solver.add_object((slice(30, 40), slice(40, 50)), erxx=7, eryy=3)
   solver.solve()
   solver.visualize()

Examples
--------

* ``1D_Example.py``
* ``2D_example.py``

FEM Electrostatic Solver
========================

.. contents:: On this page
   :local:
   :depth: 2

First example
-------------

With the packages installed as described in the `project setup <../../../../README.md>`_,
run this example from the repository root::

    python examples/fem/electrostatics/parallel_plate_capacitor_1d.py

The script prints node count and electrostatic energy, then opens an interactive potential, field, and mesh viewer.

A Matplotlib GUI backend is needed to display the figures.
For numerical runs without windows, omit ``show()`` or the plotting call;
FEM results also offer ``plot()`` for static figures.

Open the `first example <../../../../examples/fem/electrostatics/parallel_plate_capacitor_1d.py>`_ to change
geometry and controls. The `example index <../../../../examples/fem/electrostatics/README.rst>`_
provides a learning order and more physical problems. Scripts run from any working
directory once the packages are installed in the same Python environment.

Scripts that save results write to
``outputs/examples/fem/electrostatics/<example>/`` relative to the checkout.
The first example may only display results; see its code for explicit save calls.

Working with the solver
-----------------------

Compare the solver with analytical capacitor and Poisson solutions using::

    python benchmarks/analytical/parallel_plate_electrostatics.py --check

The `benchmark guide <../../../../benchmarks/README.md>`_ explains capacitance
units, the no-fringing boundary conditions, and interpolation-error convergence.

1. Construct the solver with keyword arguments and physical lengths in metres.
2. Add geometry, material properties, and boundary or excitation conditions.
3. Call ``mesh(...)`` to choose spatial and element resolution.
4. Call ``solve(...)`` to obtain a typed result; ``max_refinements=0`` uses one fixed mesh.
5. Inspect diagnostics, then call ``show()``, ``plot()``, or ``save(path)`` explicitly.

Part of **Computational Electromagnetics**, version 1.0.0.

Workflow
--------

.. code-block:: python

    from fem_electrostatics import ElectrostaticSolver, load_result

    solver = ElectrostaticSolver(dim=1, x_range=.01, outer_potential=None)
    solver.set_potential(geometry="left", potential=0., name="ground")
    solver.set_potential(geometry="right", potential=1., name="signal")
    solver.mesh(max_element_size=.001)
    result = solver.solve(max_refinements=0)
    result.save("outputs/electrostatics.h5")
    loaded = load_result("outputs/electrostatics.h5")
    print(loaded.conductor_charge("signal"))
    loaded.show()

``solve()`` automatically meshes when needed and never saves or opens a window.
The adaptive defaults are two refinements and a relative tolerance of 0.05.
The example uses a fixed mesh for reproducibility. Geometry edits invalidate
``mesh_data`` and ``result``; an automatic rebuild reuses explicit mesh settings.

The interactive Matplotlib viewer selects potential, cell fields, and mesh.
Relative epsilon may be scalar, diagonal, or symmetric positive-definite.

Persistence and API
-------------------

Results use the ``cem-fem-results`` HDF5 envelope, schema ``1.0``. Old archives
are rejected. Loading supports inspection, plotting, and saving, not solver
restart or Python callback restoration. Mode indices are zero-based.

See `API_REFERENCE.rst <API_REFERENCE.rst>`_ for supported configuration,
results, units, defaults, and exceptions. Run bundled examples from the repository
root with ``python examples/fem/electrostatics/<example>.py``.

See the `family example index <../../../../examples/fem/electrostatics/README.rst>`_
for learning order, output locations, and viewer requirements.

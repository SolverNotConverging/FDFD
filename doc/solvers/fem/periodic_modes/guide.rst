FEM Periodic Mode Solver
========================

.. contents:: On this page
   :local:
   :depth: 2

First example
-------------

With the packages installed as described in the `project setup <../../../../README.md>`_,
run this example from the repository root::

    python examples/fem/periodic_modes/uniform_cell_2d.py

The computed TEM effective index should be close to 1.5. The FEM Periodic Mode Viewer opens the completed result.

This example uses the `native periodic viewer <../../../../apps/fem_periodic_mode_viewer/README.rst>`_ when calling ``show()``.
For numerical runs without windows, omit ``show()`` or the plotting call;
FEM results also offer ``plot()`` for static figures.

Open the `first example <../../../../examples/fem/periodic_modes/uniform_cell_2d.py>`_ to change
geometry and controls. The `example index <../../../../examples/fem/periodic_modes/README.rst>`_
provides a learning order and more physical problems. Scripts run from any working
directory once the packages are installed in the same Python environment.

Scripts that save results write to
``outputs/examples/fem/periodic_modes/<example>/`` relative to the checkout.
The first example may only display results; see its code for explicit save calls.

Working with the solver
-----------------------

Compare lossless and lossy TEM propagation with the homogeneous-medium solution::

    python benchmarks/analytical/uniform_periodic_medium.py --check

The `benchmark guide <../../../../benchmarks/README.md>`_ explains the analytical
complex effective index, passive attenuation, and output tables and plots.

1. Construct the solver with keyword arguments and physical lengths in metres.
2. Add geometry, material properties, and boundary or excitation conditions.
3. Call ``mesh(...)`` to choose spatial and element resolution.
4. Call ``solve(...)`` to obtain a typed result; ``max_refinements=0`` uses one fixed mesh.
5. Inspect diagnostics, then call ``show()``, ``plot()``, or ``save(path)`` explicitly.

Part of **FDFD**, version 1.0.0.

Workflow
--------

.. code-block:: python

    from cem_common import Material
    from fem_periodic_modes import PeriodicModeSolver2D, load_result

    dielectric = Material(name="uniform dielectric", epsilon=2.25)
    solver = PeriodicModeSolver2D(frequency=10e9, x_range=.02, z_range=.005,
                                  background_material=dielectric)
    solver.mesh(max_element_size=.003)
    result = solver.solve(num_modes=1, neff_guess=.66, max_refinements=0)
    result.save("outputs/periodic.h5")
    loaded = load_result("outputs/periodic.h5")
    figure = loaded.plot(component="Ey", mode=0)
    loaded.show()

``solve()`` automatically meshes when needed and never saves or opens a window.
The adaptive defaults are two refinements and a relative tolerance of 0.05.
The example uses a fixed mesh for reproducibility. Geometry edits invalidate
``mesh_data`` and ``result``; an automatic rebuild reuses explicit mesh settings.

``show()`` uses the native viewer included in the complete Windows wheel (or
built separately for source installations). Launch failures report
the executable discovery setting; saving and loading work without the viewer.

Electromagnetic fields use ``exp(+i*omega*t)`` and guided propagation
``exp(-i*beta*z)``. Passive constitutive values have nonpositive imaginary
parts; forward attenuation is ``-Im(beta)``.

Persistence and API
-------------------

Results use the ``cem-fem-results`` HDF5 envelope, schema ``1.0``. Old archives
are rejected. Loading supports inspection, plotting, and saving, not solver
restart or Python callback restoration. Mode indices are zero-based.

See `API_REFERENCE.rst <API_REFERENCE.rst>`_ for supported configuration,
results, units, defaults, and exceptions. Run bundled examples from the repository
root with ``python examples/fem/periodic_modes/<example>.py``.

See the `family example index <../../../../examples/fem/periodic_modes/README.rst>`_
for learning order, output locations, and viewer requirements.

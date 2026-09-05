FEM Waveguide Mode Solver
=========================

.. contents:: On this page
   :local:
   :depth: 2

First example
-------------

With the packages installed as described in the `project setup <../../../../README.md>`_,
run this example from the repository root::

    python examples/fem/waveguide_modes/rectangular_waveguide_2d.py

The script prints the computed and analytic TE10 effective indices and opens the Matplotlib mode viewer.

A Matplotlib GUI backend is needed to display the figures.
For numerical runs without windows, omit ``show()`` or the plotting call;
FEM results also offer ``plot()`` for static figures.

Open the `first example <../../../../examples/fem/waveguide_modes/rectangular_waveguide_2d.py>`_ to change
geometry and controls. The `example index <../../../../examples/fem/waveguide_modes/README.rst>`_
provides a learning order and more physical problems. Scripts run from any working
directory once the packages are installed in the same Python environment.

Scripts that save results write to
``outputs/examples/fem/waveguide_modes/<example>/`` relative to the checkout.
The first example may only display results; see its code for explicit save calls.

Working with the solver
-----------------------

For a quantitative comparison with theory, run::

    python benchmarks/analytical/rectangular_waveguide_modes.py --check

This compares FEM and FDFD TE10 effective indices with the rectangular PEC
waveguide solution at several resolutions. The `benchmark guide <../../../../benchmarks/README.md>`_
explains the geometry, error measures, and generated CSV/plot reports.

The `coaxial TEM benchmark <../../../../benchmarks/analytical/coaxial_waveguide_adaptivity.py>`_
compares adaptive fields against the analytical radial solution::

    python benchmarks/analytical/coaxial_waveguide_adaptivity.py --check

Assign ``materials.PEC`` to a ``shapes.Circle`` for a circular conductor.
``shapes.Annulus`` models the outer conductor and ``clip=True`` explicitly
intersects out-of-bounds geometry with the model domain.
The benchmark demonstrates both conductors and the mesh/solve workflow.
Its plots distinguish physical field errors, eigenproblem residuals, and the
adaptive estimator. The current 2D waveguide backend globally remeshes by a
factor of 1.5 when its estimator requests refinement; it does not mark local
elements. The benchmark records budget exhaustion separately from convergence.

1. Construct the solver with keyword arguments and physical lengths in metres.
2. Add geometry, material properties, and boundary or excitation conditions.
3. Call ``mesh(...)`` to choose spatial and element resolution.
4. Call ``solve(...)`` to obtain a typed result; ``max_refinements=0`` uses one fixed mesh.
5. Inspect diagnostics, then call ``show()``, ``plot()``, or ``save(path)`` explicitly.

Part of **FDFD**, version 1.0.0.

Workflow
--------

.. code-block:: python

    from cem_common import materials
    from fem_waveguide_modes import ModeSolver2D, load_result

    solver = ModeSolver2D(frequency=10e9, x_range=22.86e-3,
                         y_range=10.16e-3, boundary=materials.PEC)
    solver.mesh(max_element_size=1e-3)
    result = solver.solve(num_modes=4, neff_guess=.8, max_refinements=0)
    result.save("outputs/modes.h5")
    loaded = load_result("outputs/modes.h5")
    figure = loaded.plot(component="Ey", mode=0)
    solver.show()

``solve()`` automatically meshes when needed and never saves or opens a window.
The adaptive defaults are two refinements and a relative tolerance of 0.05.
The example uses a fixed mesh for reproducibility. Geometry edits invalidate
``mesh_data`` and ``result``; an automatic rebuild reuses explicit mesh settings.

``show()`` uses the interactive Matplotlib mode viewer. ``plot()`` returns
a Figure without opening a GUI. Both dimensions support scalar and diagonal
relative materials; the 2D solver also supports surface-impedance conductors.

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
root with ``python examples/fem/waveguide_modes/<example>.py``.

See the `family example index <../../../../examples/fem/waveguide_modes/README.rst>`_
for learning order, output locations, and viewer requirements.

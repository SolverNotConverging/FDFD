FEM Waveguide Scattering Solver
===============================

.. contents:: On this page
   :local:
   :depth: 2

First example
-------------

With the packages installed as described in the `project setup <../../../../README.md>`_,
run this example from the repository root::

    python examples/fem/waveguide_scattering/uniform_waveguide_2d.py

The uniform guide should have effective index near 1, reflection near 0, and transmission near 1. The FEM Waveguide Scattering Viewer opens the result.

This example uses the `native scattering viewer <../../../../apps/fem_waveguide_scattering_viewer/README.rst>`_ when calling ``show()``.
For numerical runs without windows, omit ``show()`` or the plotting call;
FEM results also offer ``plot()`` for static figures.

Open the `first example <../../../../examples/fem/waveguide_scattering/uniform_waveguide_2d.py>`_ to change
geometry and controls. The `example index <../../../../examples/fem/waveguide_scattering/README.rst>`_
provides a learning order and more physical problems. Scripts run from any working
directory once the packages are installed in the same Python environment.

Scripts that save results write to
``outputs/examples/fem/waveguide_scattering/<example>/`` relative to the checkout.
The first example may only display results; see its code for explicit save calls.

Working with the solver
-----------------------

1. Construct the solver with keyword arguments and physical lengths in metres.
2. Add geometry, material properties, and boundary or excitation conditions.
3. Call ``mesh(...)`` to choose spatial and element resolution.
4. Call ``solve(...)`` to obtain a typed result; ``max_refinements=0`` uses one fixed mesh.
5. Inspect diagnostics, then call ``show()``, ``plot()``, or ``save(path)`` explicitly.

Before the scattering solve, use ``solve_modes()`` and ``set_incident_mode(0)``
to configure the first lead mode. The mesh is 2D; the fields are 2.5D full-vector.

Part of **FDFD**, version 1.0.0.

A **2.5D full-vector** scattered-field solver on a two-dimensional x/z mesh.
The invariant-direction factor is ``exp(-i*ky*y)``.

Workflow
--------

.. code-block:: python

    from cem_common import materials
    from fem_waveguide_scattering import WaveguideScatteringSolver2D, load_result

    solver = WaveguideScatteringSolver2D(frequency=10e9, x_range=.04,
        z_range=(-.1, .1), boundary=materials.PEC)
    solver.add_pml(thickness=.03, direction="z")
    solver.mesh(max_element_size=.005)
    solver.solve_modes(num_modes=1, neff_guess=.9, max_refinements=0)
    solver.set_incident_mode(0)
    result = solver.solve(max_refinements=0)
    result.save("outputs/scattering.h5")
    loaded = load_result("outputs/scattering.h5")
    print(loaded.S11, loaded.S21)
    loaded.show()

``solve()`` automatically meshes when needed and never saves or opens a window.
The adaptive defaults are two refinements and a relative tolerance of 0.05.
The example uses a fixed mesh for reproducibility. Geometry edits invalidate
``mesh_data`` and ``result``; an automatic rebuild reuses explicit mesh settings.

``show()`` uses the native viewer included in the complete Windows wheel (or
built separately for source installations). Launch failures report
the executable discovery setting; saving and loading work without the viewer.

``slot = solver.add_slot(geometry=sheet, z_range=(z0, z1))`` cuts an opening
in a background PEC sheet. Use ``solver.remove(geometry=slot)`` to close it.
Remove dependent slots before removing their parent sheet.

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
root with ``python examples/fem/waveguide_scattering/<example>.py``.

See the `family example index <../../../../examples/fem/waveguide_scattering/README.rst>`_
for learning order, output locations, and viewer requirements.

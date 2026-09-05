FEM Periodic Mode Solver
========================

Part of **Computational Electromagnetics**, version 1.0.0.

Installation
------------

From the repository root, create the project environment and install the maintained packages::

    conda env create -f environment.yml
    conda activate cem
    python scripts/install_python.py

This family also builds as an independent wheel. Install its declared dependencies
along with its wheel; source packages use the standard ``src/`` layout.

Workflow
--------

.. code-block:: python

    from fem_periodic_modes import PeriodicModeSolver2D, load_result

    solver = PeriodicModeSolver2D(frequency=10e9, x_range=.02, z_range=.005)
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

``show()`` uses the separately built native viewer. Launch failures report
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
root with ``python solvers/fem/periodic_modes/examples/<example>.py``.

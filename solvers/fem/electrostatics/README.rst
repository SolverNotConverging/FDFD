FEM Electrostatic Solver
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

    from fem_electrostatics import ElectrostaticSolver, load_result

    solver = ElectrostaticSolver(dim=1, x_range=.01, outer_potential=None)
    solver.set_potential(region="left", potential=0., name="ground")
    solver.set_potential(region="right", potential=1., name="signal")
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
root with ``python solvers/fem/electrostatics/examples/<example>.py``.

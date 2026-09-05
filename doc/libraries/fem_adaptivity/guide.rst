FEM adaptivity policy
=====================

Version 1.0.0 of the shared adaptive policy used by Computational
Electromagnetics. This independently installable library supplies error
estimation and refinement policy to the FEM families. Assembly remains in
the individual solvers.

Users configure adaptation through their solver's ``solve`` method::

    result = solver.solve(max_refinements=2, adaptive_tolerance=0.05)

Zero refinements means one solve on the initial mesh. Results expose stopping
status, element counts, and discretization residuals through ``solve_info``.
Algebraic residuals remain separate. Mesh and numerical failures are visible.

See the `project setup <../../../README.md>`_ for installation and
`API_REFERENCE.rst <API_REFERENCE.rst>`_ for the supported user surface.

Examples
--------

Configure adaptation through a solver's ``solve(max_refinements=2,
adaptive_tolerance=0.05)``. The FEM examples under ``examples/fem/`` demonstrate the
physical solver workflows; ``max_refinements=0`` selects a fixed initial mesh.

Tests
-----

Adaptive convergence, budget exhaustion, failures, and state restoration are
tested across solver families in ``tests/test_fem_adaptivity.py`` at the repository
root. Package-specific residual estimators are additionally exercised by their
own numerical suites.

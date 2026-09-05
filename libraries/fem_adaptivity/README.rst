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

Install through ``python scripts/install_python.py`` from the repository
root, or install this wheel with its declared dependencies. See
`API_REFERENCE.rst <API_REFERENCE.rst>`_ for the supported user surface.

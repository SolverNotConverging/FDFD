FEM adaptivity user configuration
===================================

This library has no independent end-user solver. Its assembly and estimator
functions are implementation interfaces used by the FEM packages. Users
configure the following keyword arguments on a solver's ``solve()`` method:

.. list-table:: Adaptive controls
   :header-rows: 1

   * - Argument
     - Type / default
     - Meaning
   * - ``max_refinements``
     - int / 2
     - Optional nonnegative number of mesh refinements after the initial solve.
   * - ``adaptive_tolerance``
     - float / 0.05
     - Optional positive relative discretization-residual threshold.

The solver returns its own typed result. ``solve_info`` describes convergence,
the refinement history, and exhausted budgets; numerical residuals are
reported independently. Invalid controls raise the solver family's
configuration error. Use the corresponding solver's API reference for
physics-specific filtering and mesh budget controls.

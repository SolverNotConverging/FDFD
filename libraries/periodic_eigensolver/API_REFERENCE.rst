Periodic eigensolver user API
=============================

Version 1.0.0. This is a numerical library for applications that already
have a generalized pencil. Solver-family users normally configure the
eigensolver through their FEM or FDFD solve method.

``solve_generalized``
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    solve_generalized(A, B, *, sigma, num_modes, tol=1e-10, ncv=None, max_restarts=12, random_seed=0, backend: 'PublicBackend' = 'auto') -> 'ArnoldiResult'

Solve ``A x = lambda B x`` with refined shift-invert Arnoldi.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``A``
     - ``array-like | scipy.sparse.spmatrix``
     - Required
     - ``—``
     - Square sparse or dense matrix of the generalized eigenproblem.
   * - ``B``
     - ``array-like | scipy.sparse.spmatrix``
     - Required
     - ``—``
     - Matrix with the same shape as A; may be singular when the shifted pencil is invertible.
   * - ``sigma``
     - ``complex``
     - Required
     - ``—``
     - Finite complex spectral shift, in the eigenvalue units.
   * - ``num_modes``
     - ``int``
     - Required
     - ``—``
     - Positive number of eigenpairs requested.
   * - ``tol``
     - ``float``
     - Optional
     - ``1e-10``
     - Nonnegative relative residual target; zero requests the literal limit.
   * - ``ncv``
     - ``int | None``
     - Optional
     - ``None``
     - Arnoldi subspace dimension; None chooses a backend default.
   * - ``max_restarts``
     - ``int``
     - Optional
     - ``12``
     - Nonnegative restart budget.
   * - ``random_seed``
     - ``int``
     - Optional
     - ``0``
     - Seed for the deterministic initial vector.
   * - ``backend``
     - ``PublicBackend``
     - Optional
     - ``'auto'``
     - auto selects the native extension when available; python and cython select explicitly.

Returns: an immutable ArnoldiResult.

``native_backend_available``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    native_backend_available() -> 'bool'

Return whether the optional Cython kernel module can be imported.

Returns: bool indicating whether compiled kernels can be imported.

Returned ArnoldiResult
----------------------

``eigenvalues`` and ``eigenvectors`` contain the selected eigenpairs.
``physical_residuals`` measures the original A x = lambda B x problem;
``projected_residuals`` measures refined Hessenberg extraction.
``converged`` distinguishes tolerance satisfaction from exhausted budgets.
``restart_count``, ``step_count`` and ``backend`` report numerical effort.
The result constructor and orthogonalization kernels are implementation details.

Example
-------

.. code-block:: python

    import numpy as np
    from scipy.sparse import diags, eye
    from periodic_eigensolver import solve_generalized

    result = solve_generalized(diags(np.arange(1., 21.)), eye(20),
                               sigma=3.1, num_modes=2)
    print(result.eigenvalues, result.physical_residuals)

Invalid shapes or options raise ValueError. An explicitly unavailable Cython
backend raises ImportError. Singular shift factorizations raise the numerical
backend error; choose a nonsingular shift. No plotting or persistence is
performed by this library.

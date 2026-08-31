Periodic eigensolver
====================

``periodic_eigensolver`` provides a shared implementation of refined shift-and-invert Arnoldi for generalized sparse pencils. The refined Ritz vectors use the paper-defined small rectangular Hessenberg residual, while convergence is validated against the original ``A x = lambda B x`` pencil.

The optional Cython extension accelerates large complex orthogonalization and residual-norm kernels. If it is unavailable, the same public API uses a portable NumPy/SciPy implementation.

Installation
------------

For development from the repository root, use an editable native build. This places the extension beside the checkout package, so running Python from the repository does not shadow a site-packages copy of the extension:

.. code-block:: console

   python -m pip install -e ./periodic_eigensolver
   python -c "import periodic_eigensolver as p; print(p.native_backend_available())"

The verification command should print ``True``. End users may instead install a published binary wheel normally with ``python -m pip install periodic-eigensolver``.

Building the extension requires a C compiler supported by the active Python: MinGW-w64 or MSVC on Windows, AppleClang on macOS, or GCC/Clang on Linux. The compiler must match the Python distribution's extension ABI (official CPython Windows builds normally use MSVC; use a MinGW-compatible Python or an explicitly configured ``mingw32`` build for MinGW). The extension uses the BLAS implementation exposed by SciPy, so no separate BLAS configuration is needed. The extension is optional: when a compiler is unavailable, installation still provides the NumPy/SciPy implementation.

Source installs may therefore fall back, but a published binary wheel must contain the extension. Build publishable artifacts through the automated release entry point:

.. code-block:: console

   python periodic_eigensolver/scripts/build_release_wheel.py

It builds into a private staging directory, runs the native-extension contract check, and only then moves the wheel into ``periodic_eigensolver/dist``. A failed optional extension build can therefore support an ordinary source install but cannot produce a release artifact. ``scripts/verify_native_wheel.py DIST_WHEEL`` remains available as an independent upload-job check.

Backend contract
----------------

The primary API is:

.. code-block:: python

   from periodic_eigensolver import solve_generalized

   result = solve_generalized(
       A,
       B,
       sigma=shift,
       num_modes=4,
       backend="auto",
   )

Public backend names are ``auto``, ``cython``, and ``python``. ``auto`` selects the Cython extension when it can be imported and otherwise selects Python/NumPy. An explicitly requested but unavailable Cython backend raises ``ImportError``; an installed but broken native binary is never silently hidden by the fallback. The older ``refined_shift_invert_arnoldi`` four-tuple API remains available and continues to accept ``kernel_backend="numpy"`` as a legacy alias for ``python``.

The returned immutable ``ArnoldiResult`` contains eigenvalues, eigenvectors, physical original-pencil residuals, projected smallest singular values, restart and Arnoldi-step counts, convergence status, and the resolved backend. When the restart budget is exhausted it returns the best candidate set found with ``converged=False``.

The sparse shift-and-invert factorization and sparse matrix products remain in SciPy. The native extension owns the repeated complex two-pass Arnoldi orthogonalization and batched final residual norms. Refined extraction follows the paper's ``(k + 1) x k`` Hessenberg SVD. Roundoff-identical repeated roots are refined as an independent right-singular subspace; distinct clustered roots retain their unrestricted refined vectors even when a non-normal pencil makes those vectors nearly parallel. Every returned mode is validated with the original sparse pencil residual.

Controlled performance gate
---------------------------

Performance is intentionally kept out of ordinary CI because BLAS thread counts and shared-runner load make timing assertions noisy. A release runner with the native extension installed can execute the specified gate with:

.. code-block:: console

   python periodic_eigensolver/benchmarks/benchmark_mgs.py --enforce

The script forces the common BLAS thread-control variables to one before importing NumPy. At ``n=100000, ncv=32`` it requires native MGS to be at least 20% faster than the optimized Fortran-order Python fallback and at least 2x faster than the former C-order layout.

Run the complementary LU-inclusive release check with:

.. code-block:: console

   python periodic_eigensolver/benchmarks/benchmark_end_to_end.py --enforce

It uses alternating paired Python/Cython solves on a deterministic complex five-point pencil (``n=16384``, ``ncv=16``, two modes), reports the fraction spent in SuperLU and whether the run is LU-dominated, checks eigenvalue and invariant-subspace agreement, and rejects a median end-to-end Cython/Python ratio above ``1.05``. Both performance commands are intended for an otherwise idle controlled runner, not ordinary CI.

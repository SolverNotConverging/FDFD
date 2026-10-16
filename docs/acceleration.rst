Acceleration
============

CUDA installation is completely optional; CPU solving uses SciPy by default.
Every solver accepts ``device="cpu"``, ``"hybrid"``, ``"gpu"`` or ``"auto"``.
``"cuda"`` remains a compatibility alias for explicit ``"gpu"``.

Eigenvalue backends
-------------------

* ``cpu``: SciPy ARPACK/SuperLU (or the requested CPU refined method).
* ``hybrid``: cuDSS factorization, custom FP64 CUDA Arnoldi kernels and GPU
  original-pencil residuals, with SciPy dense Ritz eigensolving/SVD on CPU.
* ``gpu`` / ``cuda``: the resident CUDA Arnoldi, Ritz eigensolving/SVD and
  backward-error path. Results use the canonical ``info.device="cuda"``.
* ``auto``: try hybrid, then emit a ``RuntimeWarning`` before CPU fallback.
  Auto never attempts the resident GPU eigensolver.

Both existing method selections use refined Arnoldi on accelerated backends.
``info.method`` reports ``"cuda-hybrid-refined"`` or ``"cuda-refined"``;
``info.device`` records ``"hybrid"`` or ``"cuda"`` respectively.
Explicit unavailable hybrid/GPU requests raise ``BackendCapabilityError``.
Numerical failures after selection propagate without switching algorithms.
CPU execution does not probe CUDA.

Hybrid keeps the large basis and sparse numerical work on GPU, reads norm
scalars, transfers small Hessenberg matrices/residuals to CPU and refined
coefficients back to GPU. Only explicit GPU requires CUDA ``geev``/SVD and
uses two status bytes per resident cycle. CPU work also assembles operators
and reconstructs public fields. Vendor libraries may use internal host workspaces.

Direct scattering
-----------------

Use ``device="gpu"`` (or ``"cuda"``) to require cuDSS direct scattering.
``hybrid`` uses the same cuDSS factorization/substitution; there is no Ritz
matrix in a direct solve. Factors are reused until geometry or mesh changes.
Each source is uploaded and its final solution downloaded. ``auto`` selects
the accelerated path when available and warns before CPU fallback.

Installation
------------

Install ``pip install 'fdfd[cuda12]'`` or ``'fdfd[cuda13]'`` for CuPy and
NVIDIA cuDSS. Hybrid readiness checks a real cuDSS solve and compiled custom
kernels with CPU Ritz extraction. CUDA ``geev``/SVD availability is checked
only for explicit gpu/cuda eigenproblems. CUDA 13 needs a Turing or newer GPU.
Set ``CUDSS_LIBRARY`` if automatic discovery cannot find your cuDSS installation.

The default wheel remains ``py3-none-any``. It ships CUDA source compiled
on first use and requires no CUDA toolchain for a CPU-only installation.
Opt-in hardware checks run with:

.. code-block:: console

   FDFD_TEST_CUDA=1 pytest tests/test_cuda.py tests/test_cuda_residency.py

Compiled Arnoldi kernels
------------------------

``method="refined"`` spends much of its time in complex Gram-Schmidt and
residual kernels. A Cython build of them is used automatically when present:

.. code-block:: console

   pip install cython
   FDFD_BUILD_CYTHON=1 pip install -e . --no-build-isolation

This compiles ``fdfd.core._arnoldi_cython`` and needs a C compiler. Without
it the portable Python kernels are used. The benchmarks in
``benchmarks/periodic_eigensolver`` compare the two on an idle machine;
SuperLU factorization often dominates, so the end-to-end gain is modest.

The refined method
------------------

Refined shift-and-invert Arnoldi replaces each Ritz vector by the smallest
right singular vector of :math:`\bar H_k - \mu_i [I_k; 0]`, which is more
robust than ARPACK for clustered or defective spectra (e.g. 3D periodic
leaky modes). Repeated roots share an invariant subspace. Restarts continue
from the best unconverged vectors; ``max_restarts`` bounds the work and
``random_seed`` fixes the start vector. Convergence is always checked against
the original pencil, with tolerance ``max(tol, 1e-12)``.

Measured CUDA versus SciPy performance
--------------------------------------

See the `RTX 4070 Laptop / i9 benchmark <benchmarks/cuda_vs_scipy.rst>`_ for
current eigenvalue-stage and complete-solve timings, numerical checks and the
recorded measurement data.

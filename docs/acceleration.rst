Acceleration
============

Both accelerators are optional and give the same results as the default
CPU path within solver tolerance.

Resident CUDA eigensolving (``device="cuda"``)
----------------------------------------------

Every solver accepts ``device="cpu"``, ``"cuda"`` or ``"auto"``. CUDA
eigenproblems use cuDSS factorization and custom FP64 complex Arnoldi kernels.
Krylov vectors, sparse products, orthogonalization, Ritz eigenvalues and SVD,
normalization, restarts and original-pencil backward errors stay on the GPU.
Both ``method="arpack"`` and ``"refined"`` select this resident algorithm
on CUDA; ``result.info.method`` reports ``"cuda-refined"``.

CPU work assembles the sparse operators, prepares static metadata, orchestrates
CUDA calls and reconstructs public fields after the final eigenpair export.
Each restart downloads only two Boolean control bytes. CUDA libraries may
use internal host workspaces; this is not a claim of zero library transfers.

Scattering uses reusable cuDSS factorizations and GPU direct solves, uploading
each source and downloading its final solution. Factors are reused until the
geometry or mesh changes.

``"auto"`` checks the active device, a real cuDSS solve and, for eigenproblems,
custom-kernel compilation and CUDA Ritz/SVD execution. It emits a
``RuntimeWarning`` if it falls back to CPU. Explicit ``"cuda"`` raises
``BackendCapabilityError`` if the required path is unavailable. CPU execution
does not import CuPy. Runtime numerical failures propagate without CPU retry.

Install ``pip install 'fdfd[cuda12]'`` or ``'fdfd[cuda13]'`` for CuPy and
NVIDIA cuDSS. Resident Ritz extraction also requires CUDA ``geev`` support in
the installed CuPy/cuSOLVER; availability is checked at execution time.
CUDA 13 requires a Turing or newer GPU. If automatic library discovery fails,
set ``CUDSS_LIBRARY`` to ``cudss64_0.dll`` or ``libcudss.so``.

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

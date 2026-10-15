Acceleration
============

Both accelerators are optional and give the same results as the default
CPU path within solver tolerance.

GPU factorization (``device="cuda"``)
-------------------------------------

Every solver accepts ``device="cpu"``, ``"cuda"`` or ``"auto"``. With
``"cuda"`` the shifted sparse operator is factorized with NVIDIA cuDSS and
the shift-and-invert solves run on the GPU; the Krylov iteration (ARPACK or
refined Arnoldi) stays on the CPU and exchanges one vector per step.
Scattering factorizations are cached on the GPU and reused for further
sources until the geometry or mesh changes.

``"auto"`` uses CUDA when CuPy imports, a CUDA device is present and the
cuDSS library loads, and otherwise falls back to the CPU. The choice is made
once per process and recorded in ``result.info.device``. ``"cuda"`` fails
with an error instead of falling back.

Requirements:

* CuPy for your CUDA version: ``pip install -e .[cuda12]`` or ``.[cuda13]``.
* cuDSS 0.8 or later. It is not bundled; install NVIDIA's ``nvidia-cudss``
  wheel, put the library on the system loader path, or set
  ``CUDSS_LIBRARY`` to the full path of ``cudss64_0.dll`` / ``libcudss.so``.

``python -m fdfd info`` shows what was found. The opt-in GPU tests run with:

.. code-block:: console

   FDFD_TEST_CUDA=1 pytest tests/test_cuda.py

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

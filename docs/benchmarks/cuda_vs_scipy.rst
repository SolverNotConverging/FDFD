FDFD CUDA versus SciPy benchmark
================================

Measured on 16 October 2026 using FDFD 2.1.1. This page is the
current RTX 4070 Laptop / i9 reference for the resident CUDA implementation.
It supersedes earlier local CPU/GPU laptop performance reports. Historical
Titan RTX measurements and hybrid-versus-resident controls are separate experiments.

Backend policy update: these recorded measurements explicitly selected the
resident GPU path. They remain reproducible with ``gpu`` (FDFD also accepts
``cuda``). Current ``auto`` selects hybrid GPU Arnoldi/CPU Ritz, then warns
before CPU fallback; the timing snapshots below are unchanged.

Machine and software
--------------------

* NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB; Intel Core i9-13900H.
* Windows 11, Performance power scheme; CPU BLAS/OpenMP limited to one thread.
  Both loaded OpenBLAS libraries reported one thread.
* Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1,
  CuPy 14.2.0, CUDA runtime 13.2, cuDSS 0.8.0.10.
* Numerical source commits: FEM ``29609315da77``;
  FDFD ``ca2d5f442847``.

CUDA installation remains completely optional. The standard installation supports
CPU solving; the optional CUDA extras enable the accelerated path.

Measurement protocol
--------------------

One forward mode is requested at eigensolver tolerance ``1e-9``. CPU uses
SciPy ARPACK ``eigs`` with SuperLU shift-invert; CUDA uses custom complex-FP64
resident refined Arnoldi, cuDSS sparse factorization/solves and CUDA Ritz
eigensolving/SVD. Each backend solves the same mesh and material model.

Each configuration receives one untimed solve per backend, followed by three
measured repeats in alternating CPU/GPU order. Tables report median wall times;
CUDA streams are synchronized at timer boundaries. Every solve performs a fresh
factorization. Warm FEM 3D solves reuse static resident geometry. Runtime probes
and compiler caches are warmed globally. Adaptive refinement is disabled.

Waveguides and FEM periodic 2D retain production Krylov defaults. FDFD periodic
2D and FEM periodic 3D use ``ncv=96`` on both backends. FEM physical filters request
extra eigenpairs; GPU refined Arnoldi and CPU ARPACK perform different extraction
and convergence work. These are measurements of the current public implementations.

The **eigenvalue stage** includes sparse factorization, iteration, dense Ritz work
and final eigenpair export. FDFD also includes its original-pencil backward-error
check. The resident FEM 3D timer includes its polynomial residual check; CPU FEM
physical residual filters run outside this timer.

The **complete solve** measures a premeshed ``solver.solve`` call, including
physical filtering, field reconstruction/normalization, error estimation and
result export. FDFD also rebuilds modal matrices inside each public call. Meshing
is excluded from these warmed timings. GPU times include runtime setup performed
inside the solve and required transfers. Speedup is CPU seconds / CUDA seconds;
values below one favor CPU.

.. image:: ../_static/benchmarks/cuda_vs_scipy_20261016/speedup.png
   :alt: FDFD eigenvalue and complete-solve speedup versus DOF
   :width: 100%

The plot shows PEC reference cases; the tables include every measured material case.

Waveguide 2D
------------

.. list-table:: Median times and speedups
   :header-rows: 1

   * - Case
     - DOF
     - Eigen CPU / CUDA (s)
     - Eigen speedup
     - Complete CPU / CUDA (s)
     - Complete speedup
   * - ``pec``
     - 4,000
     - 0.017 / 0.137
     - 0.12x
     - 0.023 / 0.158
     - 0.15x
   * - ``pec``
     - 16,192
     - 0.077 / 0.191
     - 0.40x
     - 0.092 / 0.221
     - 0.42x
   * - ``pec``
     - 65,152
     - 0.809 / 0.528
     - 1.53x
     - 0.882 / 0.614
     - 1.44x
   * - ``pec``
     - 146,880
     - 4.570 / 1.705
     - 2.68x
     - 4.787 / 1.969
     - 2.43x
   * - ``pec_lossy``
     - 16,192
     - 0.171 / 0.219
     - 0.78x
     - 0.187 / 0.250
     - 0.75x
   * - ``pec_lossy``
     - 65,152
     - 1.155 / 0.550
     - 2.10x
     - 1.223 / 0.637
     - 1.92x

Periodic 2D
-----------

.. list-table:: Median times and speedups
   :header-rows: 1

   * - Case
     - DOF
     - Eigen CPU / CUDA (s)
     - Eigen speedup
     - Complete CPU / CUDA (s)
     - Complete speedup
   * - ``pec``
     - 4,032
     - 0.068 / 0.242
     - 0.28x
     - 0.071 / 0.265
     - 0.27x
   * - ``pec``
     - 16,256
     - 0.359 / 0.240
     - 1.49x
     - 0.364 / 0.259
     - 1.41x
   * - ``pec``
     - 65,280
     - 2.033 / 0.666
     - 3.05x
     - 2.054 / 0.767
     - 2.68x
   * - ``pec``
     - 147,072
     - 4.844 / 1.337
     - 3.62x
     - 4.899 / 1.479
     - 3.31x
   * - ``dielectric_insert``
     - 65,280
     - 2.040 / 0.779
     - 2.62x
     - 2.061 / 0.843
     - 2.45x
   * - ``pec_lossy``
     - 16,256
     - 0.347 / 0.250
     - 1.39x
     - 0.353 / 0.270
     - 1.31x
   * - ``pec_lossy``
     - 65,280
     - 2.036 / 0.656
     - 3.10x
     - 2.056 / 0.757
     - 2.72x

Cases and DOF
-------------

Waveguides use a 1 m by 0.5 m PEC enclosure at 299.792458 MHz. ``pec``
is vacuum; ``pec_lossy`` uses relative permittivity ``2 - 0.02j``.

Periodic 2D uses a 20 mm by 10 mm (x, z) cell at 10 GHz, PEC x boundaries,
and TE polarization. ``pec`` and ``pec_lossy`` use the same fills as above.
``dielectric_insert`` has vacuum background and an insert at x = 6--14 mm,
z = 4--6 mm with relative permittivity ``4 - 0.04j``.

FDFD tables report the actual reduced pencil dimension after constrained
Yee unknowns are removed. These benchmarks cover waveguide 2D and periodic 2D.
Periodic 3D was excluded from this run at the user's request.

Interpretation
--------------

The largest waveguide achieves 2.68x eigenvalue-stage and 2.43x complete-solve
speedup. The largest periodic 2D reference reaches 3.62x and 3.31x respectively.
The nonuniform dielectric-insert cell gives 2.62x eigenvalue-stage and 2.45x
complete-solve speedup. Smaller grids favor CPU: the approximately 4,000-DOF
waveguide and periodic reference both spend more time on CUDA than SciPy.

Numerical checks and recorded data
----------------------------------

All 13 FDFD configurations passed, with 78 measured warmed solves
and 26 first-solve samples. CPU/GPU mode sets are matched by effective-index
distance and must agree within ``1e-6`` relative error. Every returned-mode
residual must be at most ``1e-8``. The measured maximum relative mode difference
is ``4.133e-12``; the maximum residual is ``1.808e-14``.
The joint FEM/FDFD run contains 33 configurations and 198 warmed samples.

* `Summary CSV <../_static/benchmarks/cuda_vs_scipy_20261016/summary.csv>`_
* `All timings, first solves, residuals and effective indices <../_static/benchmarks/cuda_vs_scipy_20261016/results.json>`_
* `Machine, software, source commits and thread-pool record <../_static/benchmarks/cuda_vs_scipy_20261016/environment.json>`_
* `Verification summary <../_static/benchmarks/cuda_vs_scipy_20261016/verification.json>`_
* `SHA-256 checksums <../_static/benchmarks/cuda_vs_scipy_20261016/SHA256SUMS.txt>`_

Reproduction
------------

The shared harness lives in the FEM repository and reads the sibling FDFD
source checkout. Clone FEM and FDFD side by side, then run from FEM using a
CUDA-capable Python environment. The optional ``threadpoolctl`` dependency is
used by the benchmark to verify single-thread CPU execution.

.. code-block:: powershell

   python -m pip install -e ".[cuda13]" -e "../FDFD" threadpoolctl
   python benchmarks/cuda_vs_scipy.py --output outputs/benchmarks/cuda_vs_scipy_reproduction
   python benchmarks/report_cuda_vs_scipy.py outputs/benchmarks/cuda_vs_scipy_reproduction

Use ``cuda12`` instead when matching a CUDA 12 environment with the required
CUDA Ritz support. ``--ids`` selects individual cases listed by
``configurations()``. Completed cases are skipped on resume; use a new output
directory for a fresh measurement. Run GPU jobs sequentially on an idle machine.

`Benchmark harness <https://github.com/SolverNotConverging/Electromagnetics-FEM/blob/main/benchmarks/cuda_vs_scipy.py>`_
and `report generator <https://github.com/SolverNotConverging/Electromagnetics-FEM/blob/main/benchmarks/report_cuda_vs_scipy.py>`_
are committed with these results. Recorded timings describe this laptop,
mode count and solver settings; Titan RTX results remain a separate hardware comparison.

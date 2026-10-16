# fdfd

Finite-difference frequency-domain (FDFD) electromagnetic solvers on Yee
grids: waveguide modes, periodic (Bloch) modes, photonic band structures,
2D scattering, and mode tracking across frequency. Every solver shares one
workflow, one set of materials and shapes, and one result format.

```python
import fdfd

solver = fdfd.ModeSolver2D(frequency=100e9, x_range=(0, 10e-3), y_range=(0, 10e-3))
solver.add_circle(center=(5e-3, 5e-3), radius=3e-3, material=fdfd.Material(name="core", epsilon=6))
solver.mesh(resolution=(50, 50))
result = solver.solve(num_modes=4, device="auto")  # GPU if available, else CPU
print(result.neff)
result.save("modes.h5")  # fdfd.load("modes.h5") reads any result
result.show()
```

| Solver | Problem |
| --- | --- |
| `ModeSolver1D`, `ModeSolver2D` | Guided modes of slabs and waveguide cross-sections, including metal wall loss |
| `PeriodicModeSolver2D`, `PeriodicModeSolver3D` | Complex Bloch modes of structures periodic along z, e.g. leaky-wave antennas |
| `BandStructureSolver2D` | Photonic bands of a cell periodic in x and y |
| `ScatteringSolver2D` | TE/TM plane-wave or line-source scattering |
| `ModeTracker1D`, `ModeTracker2D` | Modes followed through crossings and cutoff over a frequency sweep, with verification |

## Install

Python 3.10 or later. **CUDA is completely optional.** The standard installation
supports all solvers on the CPU and requires no NVIDIA GPU, CUDA toolkit, CuPy or
cuDSS:

```sh
python -m pip install fdfd
```

Only if you want NVIDIA GPU acceleration, install **one** optional CUDA extra:

```sh
python -m pip install "fdfd[cuda12]"  # optional CUDA 12 support
python -m pip install "fdfd[cuda13]"  # optional CUDA 13 support
```

Choose the extra matching your NVIDIA CUDA environment. Both extras install
CuPy and NVIDIA cuDSS for custom GPU-accelerated eigenvalue solving and accelerated
direct solves. The universal `py3-none-any` wheel includes the custom CUDA kernel
source; CuPy compiles it on first GPU use. A CPU installation needs no CUDA
dependencies or compiler. CUDA execution requires a compatible NVIDIA GPU and
driver; see [Acceleration](docs/acceleration.rst) for requirements.

Select `device="hybrid"` for GPU factorization/Arnoldi with CPU Ritz extraction,
`device="gpu"` for resident CUDA Ritz extraction, or `device="cpu"` for SciPy.
`device="auto"` tries hybrid and warns before CPU fallback; resident GPU execution
is selected only explicitly. Unavailable explicit hybrid/GPU requests raise an
error. `device="cuda"` remains a compatibility alias for `gpu`; results retain
`info.device="cuda"` for that path. For scattering direct solves, `gpu` and
`hybrid` both use cuDSS; there is no Ritz extraction in the direct stage.

In our own [hybrid versus resident GPU benchmark](https://github.com/SolverNotConverging/Electromagnetics-FEM/blob/main/benchmarks/resident_comparison/results/2026-10-16/REPORT.rst), the two paths achieved
very similar eigenvalue-stage times for the tested small Ritz systems (`ncv=80`).
The dense Ritz matrix is quick to solve on CPU, so moving that step to GPU gives
little benefit; hybrid was often faster. This is why `auto` prefers hybrid.
The comparison used FEM periodic cells on Titan RTX; results depend on the
problem, subspace size and hardware.

For a source checkout, use `python -m pip install -e .` or
`python -m pip install -e ".[cuda12]"` (alternatively `cuda13`). `uv sync` also
installs the test and lint tools.

## Use

- `examples/` has runnable scripts for every solver; each saves its results to
  an `outputs/` folder next to it.
- Documentation: `docs/` (build with `sphinx-build docs docs/_build/html`).
  Start with *Getting started* and *Conventions* (SI units, `exp(+iωt)`,
  `exp(-iβz)`, Yee staggering).
- `CHANGELOG.md` lists the 2.0 changes and a migration table from 1.x.

- [CUDA versus SciPy benchmarks: RTX 4070 Laptop / i9](docs/benchmarks/cuda_vs_scipy.rst)

## Develop

```sh
uv sync
uv run pytest                 # FDFD_TEST_CUDA=1: GPU tests; FDFD_RUN_EXAMPLES=1 (or =all, incl. slow 3D): examples
uv run ruff check . && uv run ruff format --check .
uv run pre-commit install
```

## License

MIT; see `LICENSE`.

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

Python 3.10 or later. From a checkout:

```sh
pip install -e .
```

or `uv sync`, which also installs the test and lint tools. Optional extras:
`pip install -e .[cuda12]` (or `cuda13`) for GPU factorization with NVIDIA
cuDSS; see `docs/acceleration.rst`.

## Use

- `examples/` has runnable scripts for every solver; each saves its results to
  an `outputs/` folder next to it.
- Documentation: `docs/` (build with `sphinx-build docs docs/_build/html`).
  Start with *Getting started* and *Conventions* (SI units, `exp(+iωt)`,
  `exp(-iβz)`, Yee staggering).
- `CHANGELOG.md` lists the 2.0 changes and a migration table from 1.x.

## Develop

```sh
uv sync
uv run pytest                 # FDFD_TEST_CUDA=1: GPU tests; FDFD_RUN_EXAMPLES=1 (or =all, incl. slow 3D): examples
uv run ruff check . && uv run ruff format --check .
uv run pre-commit install
```

## License

MIT; see `LICENSE`.

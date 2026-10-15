# Benchmarks

Accuracy and performance checks that are too slow for the test suite. Each
script writes its report to an ignored `outputs/` folder next to it.

## Accuracy

Rectangular PEC waveguide, TE10 against the analytical propagation constant:

```sh
python benchmarks/analytical/rectangular_waveguide_modes.py --check
```

Periodic leaky-wave cell against stored first-order FEM results at 18, 20 and
22 GHz (no FEM installation needed):

```sh
python benchmarks/periodic/leaky_wave_fem_reference.py --check
python benchmarks/periodic/leaky_wave_fem_reference.py --case slab --frequencies 20e9 --check
python benchmarks/periodic/leaky_wave_fem_reference.py --frequencies 20e9 --grids 1000x320 2000x640 --check
```

The default grid resolves the 50 µm PEC patch; the finer grid takes longer.
The FEM mesh sizes and the remaining discretization differences are described
in the periodic solver documentation (`docs/solvers/periodic.rst`).

## Refined Arnoldi kernels

Compare the compiled Cython kernels with the Python ones (build them with
`FDFD_BUILD_CYTHON=1`, see `docs/acceleration.rst`):

```sh
python benchmarks/periodic_eigensolver/benchmark_mgs.py
python benchmarks/periodic_eigensolver/benchmark_end_to_end.py
```

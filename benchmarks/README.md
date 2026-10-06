# FDFD benchmarks

Run analytical benchmarks after installing the project:

```sh
python benchmarks/analytical/rectangular_waveguide_modes.py --check
```

Reports are written to ignored `fdfd_waveguide_modes/outputs/benchmarks/analytical/`.
Shared periodic eigensolver performance benchmarks are in `periodic_eigensolver/`.

The periodic antenna benchmark compares with the validated FEM geometry at
18e9, 20e9, and 22e9 Hz using stored FEM results:

```sh
python benchmarks/periodic/leaky_wave_fem_reference.py --check
python benchmarks/periodic/leaky_wave_fem_reference.py --case slab --frequencies 20e9 --check
python benchmarks/periodic/leaky_wave_fem_reference.py --frequencies 20e9 --grids 1000x320 2000x640 --check
```

Its report is saved in `fdfd_periodic_modes/outputs/benchmarks/leaky_wave_fem_reference/`.
The default grid resolves the 50 µm PEC patch. The finer grid takes longer.
The [comparison notes](../fdfd_periodic_modes/docs/fem_comparison.rst) describe
the FEM mesh sizes and the remaining discretization differences.

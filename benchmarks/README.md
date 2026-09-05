# Computational Electromagnetics benchmarks

These scripts use the installed project packages. See the [root README](../README.md)
for environment setup. Each benchmark runs directly from any working directory.

## Compare solvers with analytical solutions

```console
python benchmarks/analytical/rectangular_waveguide_modes.py --check
python benchmarks/analytical/parallel_plate_electrostatics.py --check
python benchmarks/analytical/uniform_periodic_medium.py --check
python benchmarks/analytical/coaxial_waveguide_adaptivity.py --check
```

The default cases are small numerical demonstrations. They save CSV tables and
Matplotlib figures under `outputs/benchmarks/analytical/<benchmark>/` and do not
open windows. `--output PATH` overrides the destination. `--check` turns the stated
accuracy criteria into a nonzero exit on failure; reports are saved before checking.

| Benchmark | Solvers | Analytical comparison | Saved plots |
|---|---|---|---|
| [Rectangular waveguide](analytical/rectangular_waveguide_modes.py) | FDFD and FEM waveguide modes, 2D | Vacuum PEC TE10 effective index and phase constant | Error versus nominal spatial resolution |
| [Parallel-plate electrostatics](analytical/parallel_plate_electrostatics.py) | FEM electrostatics, 1D and 2D | Potential, capacitance, energy, and uniform-space-charge potential | Analytical/FEM potential and interpolation-error convergence |
| [Uniform periodic medium](analytical/uniform_periodic_medium.py) | FEM periodic modes, 2D | Lossless and lossy TEM propagation and attenuation | Phase constant and attenuation versus frequency |
| [Coaxial waveguide adaptivity](analytical/coaxial_waveguide_adaptivity.py) | FEM waveguide modes, 2D | Unit-power TEM electric/magnetic fields, index, and adaptive diagnostics | Field-error convergence and initial/final meshes |

The [checked reference reports](reference_results/README.md) contain tracked PNG
plots, their CSV data, and a manifest of commands, versions, and source/artifact
hashes. Normal runs continue to use ignored `outputs/`. Explicitly refresh the
tracked reports after all analytical checks pass with:

```console
python scripts/update_benchmark_references.py
```

### Rectangular PEC waveguide

For a vacuum-filled rectangle of width `a` and height `b`,

```text
k0 = 2 pi f / c
kc^2 = (m pi / a)^2 + (n pi / b)^2
beta = sqrt(k0^2 - kc^2)
neff = beta / k0
```

The benchmark fixes `a = 22.86 mm`, `b = 10.16 mm`, `f = 10 GHz`, and the TE10
mode (`m=1, n=0`). The exact effective index is approximately `0.7550093383`.
The eigenvalue search targets this mode at every resolution, above its cutoff.

Try more resolutions with `--levels 8 12 16 24`. FDFD uses solid border cells whose
inner faces enclose the exact specified clear dimensions. FEM uses first-order
elements and independent meshes, with adaptation disabled. The meshes are not
nested, so FEM errors need not decrease monotonically between individual levels.
Equal nominal resolution does not mean equal unknown counts or cost: the CSV
records element counts, numerical values, relative errors, and elapsed time.
FEM algebraic residuals are recorded separately from error against the analytical
solution; the FDFD API does not expose an algebraic residual for this operation.

`--check` requires finite solutions, negligible imaginary effective index for this
lossless guide, and less than 1% relative index error at the finest requested level.

### Parallel plates and uniform space charge

With plate separation `L`, relative permittivity `epsilon_r`, and applied voltage `V`,

```text
phi(x) = V x / L
C / area = epsilon_0 epsilon_r / L
energy / area = 0.5 (C / area) V^2
```

In the 2D test, insulating top/bottom boundaries eliminate fringing. Multiplying
the per-area capacitance by the modeled plate height gives capacitance per
out-of-plane length. The CSV labels 1D capacitance as `F/m^2` and 2D capacitance
as `F/m`.

A second problem grounds both plates and adds uniform volume charge `rho`:

```text
phi(x) = rho x (L - x) / (2 epsilon_0 epsilon_r)
```

Piecewise-linear FEM can reproduce the quadratic solution at the nodes while
still having interpolation error between them. This benchmark measures relative
L2 error on a dense physical sampling line. Use `--levels 8 16 32 64` to vary the
requested maximum element size `L / level`; geometry refinement may produce
smaller actual elements. The CSV records actual node counts.

`--check` requires capacitor potential error below `1e-8`, relative capacitance
and energy errors below `1e-7`, and charged-potential L2 error below 1% at the finest
requested level. Small constant-dependent differences remain because the reference
uses SciPy's permittivity constant while the backend retains its original constant.

### Uniform periodic TEM propagation

A homogeneous medium with relative permeability 1 has

```text
neff = sqrt(epsilon_r)
phase constant = Re(k0 neff)
attenuation = -Im(k0 neff)
```

The benchmark uses `epsilon_r = 2.25` and `2.25 - 0.02j` at 8, 10, and 12 GHz.
Negative imaginary permittivity is passive for `exp(+i omega t)`. The constant
periodic envelope is representable exactly here, so errors can approach numerical
roundoff. This validates branch/sign conventions and material handling rather than
demonstrating a general convergence rate.

`--check` requires relative complex-index error below `5e-4`, nonnegative attenuation
within `1e-8 Np/m`, and lossy-attenuation error below 1%.

### Coaxial TEM adaptive refinement

The vacuum coax has inner radius `a = 1 mm`, outer radius `b = 4 mm`, PEC walls,
and frequency `1 GHz`. For forward propagation with `exp(+i omega t - i beta z)`,

```text
neff = 1; beta = k0
eta0 = sqrt(mu0 / epsilon0)
Zc = eta0 ln(b/a) / (2 pi)
V = sqrt(2 Zc)                       # peak phasor for 1 W average power
E = V / ln(b/a) (x, y, 0) / r^2
H = (-Ey, Ex, 0) / eta0
```

First-order edge elements can reproduce the TEM index near roundoff even on a
coarse mesh. Field accuracy is therefore the main comparison: relative full-vector
L2 errors against the analytical E and H fields, integrated with order-four
triangle quadrature. Only the arbitrary global complex phase is aligned; amplitudes
are independently normalized to 1 W. Straight mesh edges approximate the circular
walls. Errors are measured on the numerical annulus, include boundary approximation,
and are accompanied by its relative area error. Area error can cancel between the
two polygonal walls and is not an estimate of field error.

Each run starts from the same 1 mm maximum element size, with boundary-size
refinement disabled, and increases `solve(max_refinements=budget)` from zero to
four. The **current waveguide backend globally remeshes at h/1.5** when its Maxwell
jump estimator exceeds the requested tolerance. This is not a locally marked
adaptive mesh. Independent runs retain the final fields at each budget; timings
include repeated earlier solves and are not incremental refinement costs.

Use `--max-refinements N` and `--adaptive-tolerance VALUE` to change the budget and
stopping threshold. The default reference reduces both E/H errors from **21.47%
to 4.66%**, using 125 to 3,178 triangles. Its estimator decreases from 0.693 to 0.169,
which **does not meet the default 0.05 tolerance**: every reference run ends at its
refinement limit. The CSV records this explicitly. The eigenproblem residual and
the adaptive estimator are reported separately from errors against theory.

`--check` requires finite diagnostics, effective-index error below `1e-6`,
eigenproblem residual below `1e-8`, forward power within `1e-8 W` of 1 W, consistent
adaptive stopping status, and finest E/H errors below 5% and below their initial
values. Passing these physical checks does not imply the adaptive tolerance was met.

![Coaxial field errors and adaptive diagnostics](reference_results/coaxial_waveguide_adaptivity/convergence.png)

![Initial and refined coaxial meshes](reference_results/coaxial_waveguide_adaptivity/meshes.png)

## Native eigensolver performance gates

Run these separately on an otherwise idle machine with the compiled extension:

```console
python benchmarks/periodic_eigensolver/benchmark_mgs.py --enforce
python benchmarks/periodic_eigensolver/benchmark_end_to_end.py --enforce
```

These performance gates require an otherwise idle machine and the compiled
periodic eigensolver extension. Numerical regression tests live under root `tests/`.
Elapsed times in the analytical examples are observations, not controlled performance
comparisons. Store generated benchmark reports under `outputs/`.

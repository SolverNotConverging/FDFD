# Release history

Changes to [Computational Electromagnetics](https://github.com/SolverNotConverging/FDFD),
formerly published as the FDFD solver collection. Historical entries summarize
the [published releases](https://github.com/SolverNotConverging/FDFD/releases)
and their tagged commits. Examples in older release notes use the API of that
release; use the root README and current solver guides for 1.0 syntax.

## 1.0.0 — Computational Electromagnetics — 2026-09-06

The project becomes a collection of eight independently installable Python
solver families, supported by shared libraries and three native applications.
Its simplified workflow is: define a solver, define materials, add geometry,
mesh, solve, and open an interactive results viewer.

- Introduces `cem_common` materials and geometry across all 12 public solver
  classes. Reusable bulk materials, vacuum/air, PEC/PMC, and eight metal SIBC
  presets replace numerical material arguments on geometry methods. Shape
  primitives support Boolean operations, extrusion, translation, and rotation
  where the backend supports them.
- Adds the FEM waveguide-mode, periodic-mode, waveguide-scattering, and
  electrostatic families to the release organization. Waveguide scattering is
  documented as 2.5D full-vector physics on a 2D mesh.
- Establishes keyword-only public configuration, SI units, physical coordinate
  ranges, zero-based mode selection, geometry invalidation, and the common
  `mesh()`, `solve()`, `show()`, `plot()`, `save()`, and `load_result()` workflow.
- Supports FEM adaptive refinement with separate eigenproblem and
  discretization diagnostics. The defaults allow two refinements with a 0.05
  adaptive tolerance; zero refinements requests one fixed-mesh solve.
- Converts FEM waveguide scattering to `exp(+i*omega*t)` with consistent source,
  lead-mode, reconstruction, PML, projection, absorption, and de-embedding signs.
  Lossy and oblique fixed-mesh regressions check conjugation against the previous
  formulation. FDFD numerical kernels retain their established convention.
- Introduces atomic `cem-fem-results` and `cem-fdfd-results` HDF5 archives with
  schema 1.0, convention and field-location metadata. Loaded results can be
  inspected and plotted without solving; supported FEM sweeps retain lazy loading.
- Provides interactive Matplotlib views for waveguide modes, electrostatics,
  and FDFD results, plus separately built FEM Periodic Mode Viewer and FEM
  Waveguide Scattering Viewer applications. The Transmission Line Calculator
  remains a separate native application.
- Centralizes tutorials in `examples/`, user guides and curated API references
  in `doc/`, and Python tests in `tests/`. Root installation instructions use
  the active interpreter and the project environment name `cem`.
- Adds analytical comparisons for rectangular waveguides, parallel-plate
  electrostatics, uniform periodic media, and coaxial FEM adaptive refinement.
  CSV tables and plots are included in `benchmarks/reference_results/`.
- Final review fixes removal of scattering slots through `remove()` and gives
  the waveguide viewer's colorbar enough room beside the component controls.

This is a clean breaking release. Previous imports, material-assignment syntax,
the separate scattering `run()` workflow, and old archive formats have no
compatibility layer. Existing scripts must use the current examples and guides.

Material support remains specific to each solver. In particular, FEM waveguide
scattering supports scalar passive media and PEC sheets; its integrated port
projection requires lossless leads, and PMC/SIBC scattering boundaries are not
implemented. Other backends reject unsupported materials and shapes explicitly.
The PEC-slot source remains a first-order boundary model; see its documented
power-accounting regression rather than assuming exact flux closure.

Release qualification passed 710 Python tests and 115 subtests, 52 native CTest
cases, four analytical benchmarks, HDF5 interoperability, and all 11 wheels
installed outside the checkout. All 34 numerical examples passed and all 38
tutorial/postprocessing scripts imported successfully. The two Python warnings are the expected
coarse-PML adaptive fixtures. The README microstrip code was executed verbatim,
and its GUI screenshot shows a computed result. Binary wheel qualification is for Windows x64
with CPython 3.12; other platforms use source builds and are not claimed as
binary-qualified by this release.

## 0.7 — Real Metal, Consistent Signs — 2026-08-26

[Release notes](https://github.com/SolverNotConverging/FDFD/releases/tag/FDFD_solver_v0.7)

- Replaces thin-layer conductor approximations with true surface-impedance
  boundaries on opaque conductor cells in the 1D and 2D FDFD mode solvers.
- Adds eight metal presets and checks attenuation and phase shifts against
  perturbation theory for rectangular and parallel-plate waveguides.
- Standardizes the then-published FDFD solvers on positive-time phasors,
  negative-imaginary passive constitutive values, outgoing PML signs, and
  consistent periodic propagation mappings.
- Corrects reconstructed field phases and preserves complex band frequencies.
  The release reported 118 passing tests.

## 0.6 — Periodic PEC/PMC constraints — 2026-08-23

[Release notes](https://github.com/SolverNotConverging/FDFD/releases/tag/FDFD_solver_v0.6)

The periodic solvers reduce degrees of freedom according to PEC and PMC
constraints, bringing these boundaries into the periodic eigenproblems.

## 0.5 — Waveguide boundary corrections — 2026-08-08

[Release notes](https://github.com/SolverNotConverging/FDFD/releases/tag/FDFD_solver_v0.5)

Fixes 2D mode-solver PEC/PMC cross-component constraints.

## 0.4 — Staggered material grids — 2026-06-06

[Release](https://github.com/SolverNotConverging/FDFD/releases/tag/FDFD_solver_v0.4)

Tagged changes introduce Yee-grid placement of permittivity and permeability
components in the mode solvers and update geometry naming and examples. The
published release body contains no technical summary; this entry is based on
the tagged commit history.

## 0.3 — Ideal conductor materials — 2026-06-02

[Release notes](https://github.com/SolverNotConverging/FDFD/releases/tag/FDFD_solver_v0.3)

Adds PEC and PMC material support to the mode solvers.

## 0.2, 0.1, and 0.0 — Initial releases — 2026-02-24

The first three published snapshots establish the original solver collection.
Their short release messages do not enumerate technical changes. Consult the
tagged sources when reproducing calculations made with these versions:

- [0.2](https://github.com/SolverNotConverging/FDFD/releases/tag/FDFD_solver_v0.2)
- [0.1](https://github.com/SolverNotConverging/FDFD/releases/tag/FDFD_solver_v0.1)
- [0.0](https://github.com/SolverNotConverging/FDFD/releases/tag/FDFD_solver_v0.0)

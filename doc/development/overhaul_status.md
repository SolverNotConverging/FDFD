# 1.0 implementation status

The accepted overhaul introduces installable solver families, a common positive-time
phasor convention, the FEM `mesh / solve / show` lifecycle, typed HDF5 results,
and a curated user API. This file records implementation and qualification progress.

## Stages

- [x] Capture numerical baseline and full existing test results.
- [x] Reorganize sources, examples, tests, applications, and build metadata.
- [x] Convert waveguide scattering to `exp(+i*omega*t)`.
- [x] Standardize FEM public APIs and state invalidation.
- [x] Implement common persistence and update native readers/viewers.
- [x] Rewrite user documentation and migrate bundled examples/tests.
- [x] Qualify independently installed wheels and native applications on Windows x64.

Untracked research work and existing simulation outputs are preserved. No release
publishing or backward-compatibility layer is part of this change.

## Baseline

The original source snapshot passes **673 tests and 117 subtests** (96.93 s).
The first two attempts encountered temporary-directory setup errors; preparing a
workspace-contained pytest base directory resolved them. The report is in
`outputs/overhaul/baseline3.xml`. Lossy and 17-degree oblique scattering fields,
S-parameters, propagation constants, and power observables are captured in
`outputs/overhaul/scattering_baseline.npz`.

Maintained code now resides in eight solver families, three shared libraries,
and three native application directories. Every Python family has its own
`pyproject.toml`, `src/`, and a short navigation README. Runnable tutorials
live under root `examples/`; guides and curated API references live under root
`doc/`. Solver and library development notes are merged into their guides.
All Python solver and library tests live under root `tests/`, grouped by method
and family; cross-family checks remain directly in `tests/`. Calculators are guarded entry points in `tools/`, performance studies
are in `benchmarks/`, and release commands use the active interpreter. The
project's Conda environment is named `cem`.

The original tracked snapshot and migration diagnostics remain under ignored
`outputs/overhaul/`. All 15 legacy source directories have been removed, including
their obsolete build outputs and caches. Eight remaining simulation result files
were moved into ignored `outputs/legacy/`, preserving their relative paths and
verifying SHA-256 checksums recorded in `preservation_manifest.json` there.
The 3,016 tracked research files from branch `ground_slots_opt` (commit
`06091ae6157da662a05124e05a81224b39472589`) and existing local research artifacts
are preserved under ignored `outputs/ground_slots_opt/`. All imported files were
checksum-verified; there were no conflicts. Research remains outside the release
commit. The original branch was deleted after verifying the import; its snapshot
and import manifest remain under ignored `outputs/overhaul/`.

## Qualification results

Qualification used Windows x64 and CPython 3.12 with the existing developer
numerical environment. Native applications were built with MinGW-w64, Qt 6,
HDF5, Gmsh, Eigen, FTXUI, and VTK support for the periodic viewer.

| Check | Result | Local evidence |
|---|---|---|
| Complete Python suite | 709 tests and 115 subtests passed | `outputs/overhaul/material-api-release-final.xml` |
| Original negative-time scattering equivalence | Lossy and 17-degree oblique fields, beta, S-parameters, and powers passed | `tests/test_scattering_convention.py`, `tests/data/scattering_convention_baseline.npz` |
| Native builds and CTest | All three applications built; 52 tests passed | `outputs/overhaul/native-final-tests.log` |
| Python/native archive interoperability | Periodic 2D/3D, mixed-case periodic sweep, scattering single/sweep; inspectors and offscreen GUIs passed | `outputs/native-qualification-material-api-final/` |
| Installed examples | All 34 numerical examples passed and all 38 scripts imported with viewer launches suppressed | `outputs/example-qualification-material-api-final/` |
| Documentation | 49 RST files, curated signatures, and relative links validated | `scripts/check_documentation.py`, `tests/test_fem_documentation.py`, `tests/test_fdfd_documentation.py` |
| Python distributions | All 11 wheels built from the final sources and installed outside the checkout | `outputs/dist-material-api-final/` |

The wheel check creates a temporary isolated environment outside the checkout,
using preinstalled third-party numerical dependencies. It asserts that all 11
maintained packages load from that environment, exercises every FEM dimension
and archive round trips, and explicitly uses the compiled eigensolver. No
repository-root import adjustments are made. The build excludes cached objects
and verifies that wheel Python sources exactly match the maintained package.

Results use the `cem-fem-results` envelope, schema `1.0`, with physical units,
time/spatial conventions, mesh topology, material and boundary context, fields,
coefficients, and separate numerical/adaptive diagnostics. Sweep cases remain
lazy when loaded; saving is atomic. Old archives and incompatible conventions
are rejected. Loaded results support inspection, plotting, native/Matplotlib
viewing, and saving; they do not restore solvers or Python callbacks.

Two expected coarse-PML warnings occur in the adaptive test fixtures. Numerical
tolerances were not loosened to accommodate the overhaul. Recursive API helper
coverage and obsolete import-facade checks were replaced by the curated API
inventory and the new lifecycle/archive tests.

As an intermediate cleanup check, the Python suite passed 668 tests with
one optional native-extension skip. Restoring the compiled extension from the
qualified 1.0 wheel into the current source package and rerunning its entire suite
passed all 32 eigensolver tests, including the nine native cases absent or skipped
in the first run. Reports are `outputs/overhaul/post-cleanup-python.xml` and
`outputs/overhaul/post-cleanup-native-python.xml`.

## Artifacts and release commands

Wheels are in `outputs/dist/`; native binaries are in
`outputs/build/apps/<application>/`. Use the following from an installed `cem`
environment to repeat qualification:

```console
python -m pytest
python scripts/build_wheels.py
python scripts/qualify_wheels.py
python scripts/check_documentation.py
python scripts/qualify_examples.py
cmake --build outputs/build --parallel
ctest --test-dir outputs/build --output-on-failure
python scripts/qualify_native.py
```

Other operating systems, Python ABIs, full-length FDFD dispersion studies, and
controlled performance gates were not exercised in this Windows qualification.
Publishing remains a separate operation; no release was published.

## Central tutorials and user guides

The tutorial collection now lives in `examples/<method>/<family>/`, with consistent
physical-problem and mesh-dimension filenames. Mixed-dimension examples were split
into individual scripts, yielding 18 FEM tutorials and 19 FDFD scripts (including
four postprocessors), plus one shared eigensolver matrix example. All scripts have guarded entry points and import installed
packages. Saved results use ignored `outputs/examples/<method>/<family>/<example>/`
paths independently of the invoking directory. Existing dispersion sample data
was preserved in the matching output directory.

All Python package guides and API references live under root `doc/`. Each solver
guide starts with a runnable first example, expected results, and the core workflow.
Shared installation and project quick-start instructions live in root `README.md`.
The former development notes are merged below the solver introduction;
the separate solver/library development directories were removed. Root `README.md`
links every guide directly. Short package READMEs remain for wheel metadata.

Validation after this change:

- All 38 scripts imported in isolated Python with the installed packages.
- All 18 FEM tutorials completed with viewer windows suppressed. One mesh-snapshot
  diagnostic access was corrected and its tutorial rerun successfully; logs are
  under `outputs/root-example-qualification/`.
- The clarified FDFD dielectric-cylinder example produced a finite, nonzero TM
  field on its 120-by-120 grid. The 1D dispersion postprocessor loaded the preserved
  sample data and saved its plot in the expected ignored output directory.
- The shared eigensolver example returned eigenvalues 3 and 4 with small residuals.
- The 16 targeted documentation and example-dependent numerical tests passed.
- All 49 maintained RST documents and local links, including root README links,
  passed `scripts/check_documentation.py`.
- All 11 wheels rebuilt and passed installation, native eigensolver, numerical,
  and archive checks outside the checkout. Evidence is in
  `outputs/overhaul/root-layout-wheels.log` and
  `outputs/overhaul/root-layout-installed-wheels.log`.

The final example qualification also ran every maintained FDFD numerical example,
including the dispersion and band-structure studies selected by
`CEM_EXAMPLE_QUALIFICATION`.

## Material-first public API

All 12 solver classes now take keyword-only configuration and share physical
range, material, geometry, lifecycle, result, and error vocabulary. Users define
immutable `cem_common.Material`, boundary, or SIBC values first and pass those
objects to `add_geometry()` or dimensional convenience methods. Shared presets
include vacuum, air, PEC, PMC, and eight good-conductor metals. Each solver
rejects unsupported tensor, ideal-boundary, or SIBC forms with an actionable
capability error.

The shared geometry library provides 1D, 2D, and 3D primitives, Boolean
combinations, extrusion, translation, and rotation. Geometry edits use stable
handles and invalidate mesh/result state. The former direct PEC/PMC, object,
region, `run()`, and GUI-helper solver workflows are absent from public solver
classes. Curated inventories verify every documented FEM and FDFD export,
signature, argument, guide, and API reference.

## Root tests and analytical benchmark tutorials

Python test discovery now targets only root `tests/`. The final complete suite
passed **709 tests and 115 subtests**, with the same two coarse-PML warnings, in
96.97 seconds. The report is `outputs/overhaul/material-api-release-final.xml`.

Four runnable comparisons under `benchmarks/analytical/` passed their documented
`--check` criteria and save working CSV tables and figures under ignored
`outputs/benchmarks/analytical/`. Reviewed CSV/PNG references are committed under
`benchmarks/reference_results/`:

- Rectangular PEC TE10 modes compare FDFD and FEM with the analytical index
  0.7550093383. At the default finest level, relative index errors were about
  0.121% (FDFD) and 0.00759% (FEM); the independent meshes have different element
  counts, and the FEM errors are not assumed to decrease monotonically.
- Parallel-plate electrostatics compares potential, capacitance, and energy in
  1D and 2D, then measures interpolation error for the quadratic space-charge
  potential. Its relative L2 error fell by a factor of four on each refinement,
  reaching approximately 0.0244% at the finest default level.
- Uniform periodic TEM modes compare lossless and passive lossy propagation and
  attenuation over 8–12 GHz. Complex effective-index errors were below 4e-15.
- Coaxial TEM adaptivity compares the FEM field and capacitance against the
  logarithmic analytical solution. The field error fell from about 21.5% to
  4.66% over the stored adaptive history; the committed figure also shows the
  corresponding mesh sequence and estimator behavior.

These are educational numerical benchmarks rather than controlled performance
comparisons. The root benchmark index documents analytical formulas, assumptions,
units, output locations, and explicit accuracy checks.

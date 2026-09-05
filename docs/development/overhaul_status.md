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
`pyproject.toml`, `src/`, examples, tests, README, and curated API reference.
FDFD-specific tests live with their family; cross-family checks remain under
`tests/`. Calculators are guarded entry points in `tools/`, performance studies
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
| Complete Python suite | 677 tests and 115 subtests passed | `outputs/overhaul/release-python.xml` |
| Original negative-time scattering equivalence | Lossy and 17-degree oblique fields, beta, S-parameters, and powers passed | `tests/test_scattering_convention.py`, `tests/data/scattering_convention_baseline.npz` |
| Native builds and CTest | All three applications built; 52 tests passed | `outputs/overhaul/native-final-tests.log` |
| Python/native archive interoperability | Periodic 2D/3D, mixed-case periodic sweep, scattering single/sweep; inspectors and offscreen GUIs passed | `outputs/overhaul/native-interop2.log` |
| FEM examples | All 15 ran with their original numerical settings and viewer launches suppressed | `outputs/overhaul/examples-final.log` |
| FDFD examples | Maintained imports and statically resolved signatures checked; numerical regressions passed | `outputs/overhaul/fdfd-layout.log` |
| Documentation | 32 RST files and relative links validated | `scripts/check_documentation.py` |
| Python distributions | All 11 wheels built from fresh sources and installed outside the checkout | `outputs/overhaul/wheels-clean.log`, `outputs/overhaul/wheel-qualified-final.log` |

The wheel check creates a temporary virtual environment outside the checkout,
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

After removing the legacy directories, the Python suite passed 668 tests with
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

# Changelog

## 2.0.0

A breaking release that turns seven loosely coupled packages into one `fdfd`
package with one vocabulary. The physics is unchanged except where noted
under *Behaviour changes*; `tests/test_regression.py` reproduces results
recorded with 1.1.2.

### Layout and installation

- One package: `src/fdfd/{core,waveguide,periodic,bands,scattering,tracking}`.
  Install with `pip install -e .`; examples no longer modify `sys.path`.
- Everything public is importable from `fdfd` (`fdfd.ModeSolver2D`,
  `fdfd.Material`, `fdfd.materials`, `fdfd.shapes`, `fdfd.load`, ...).
- The version is defined once, in `pyproject.toml`.
- Examples moved to `examples/<family>/`; outputs go to an ignored
  `outputs/` folder next to each script.
- Removed: `requirements.txt`, `setup_cython.py`, `MANIFEST.in`, the custom
  API-reference scripts, wheel tooling, benchmark manifest, `tools/` (now
  `fdfd.suggest_resolution` and `fdfd.suggest_pml_strength`), and the
  internal legacy solver APIs and Tk viewers.
- Docs are a single Sphinx tree in `docs/` built from docstrings.

### Migration

| 1.x | 2.0 |
| --- | --- |
| `from fdfd_waveguide_modes import ModeSolver2D` (and the other `fdfd_*` packages) | `import fdfd; fdfd.ModeSolver2D` |
| `from fdfd_common import Material, materials, shapes` | `from fdfd import Material, materials, shapes` |
| `x_range=12e-3` (scalar extent) | `x_range=(0.0, 12e-3)` |
| `solve(backend="cpu" \| "auto" \| "cuda")` | `solve(device="cpu" \| "auto" \| "cuda")`; `"auto"` now really uses CUDA when available |
| `neff_guess=`, `eigenvalue_guess=` | `guess=` (effective index; Hz for band structures) |
| `eigensolver_tolerance=` | `tol=` |
| `eigensolver="eigs"` / `"refined"` | `method="arpack"` / `"refined"` |
| `PeriodicModeSolver2D(polarization=...)` | `solve(polarization="TE" \| "TM" \| "both")` |
| `ScatteringSolver2D(polarization=...)` | `solve(polarization="TE" \| "TM")` |
| `BandStructureSolver2D.solve(polarizations=("TE", "TM"))` | `solve(polarization="both")` |
| `add_source(...)` + `set_source_region(inset=...)` | `set_source(..., inset=...)` |
| `solve(reuse_factorization=...)` | removed; factorizations are always reused |
| `solver.make_bloch_path(points=..., num_points=...)` | `fdfd.bloch_path(points, num_points=...)` |
| `add_pml(thickness, direction="x+", sigma_max=..., order=...)` | `add_pml(thickness=..., sides="x+", strength=..., order=...)` |
| `ModeTracker*.solve(tracking_config=...)` | `solve(config=...)` |
| `TrackingConfig(neff_guess=..., eigensolver_tolerance=...)` | `TrackingConfig(guess=..., tol=...)` |
| `track_modes(..., backend=...)` | `track_modes(..., device=...)` |
| `ModeSet`, `PeriodicModeSet` | `fdfd.ModeResult` |
| `BandStructureResult` | `fdfd.BandResult` (`.eigenvalues` removed; use `.frequencies`) |
| `ScatteringResult.neff`, `.beta`, `.attenuation_constant` | removed; `.polarization` added |
| `result.mesh_data`, `solver.mesh_data` | `result.grid`, `solver.grid` |
| `result.solve_info["residuals"]` etc. | `result.info.residuals` (a `SolveInfo` with the same fields everywhere) |
| `result.solve_info["backend"]` | `result.info.device` |
| `result.metadata["k0"]` | `result.k0` |
| `fdfd_*.load_result(path)`, `load_sweep(path)` | `fdfd.load(path)` |
| `fdfd_periodic_modes.plot_dispersion` (unconnected) | `fdfd.plot_dispersion(..., connect=False)` |
| `TrackedSweep.plot(component=..., quantity=...)`, `.show(component=..., ...)` | `plot(component=..., quantity=...)`, `show(block=...)` |
| `CEMError` | `fdfd.FDFDError` |
| `periodic_eigensolver.solve_generalized(..., backend=...)` | `fdfd.core.eigs.solve(A, B, method="refined", ...)` (`kernels=` / `device=` in `fdfd.core._refined`) |
| `shapes.Segment`, `materials.SpatialMaterial`, `MeshSnapshot` | removed (no solver supported them) |
| metal presets by name (`preset="Cu"`) | `materials.copper` etc. |

Result files from 1.x cannot be loaded; re-run the solve.

### Behaviour changes

- **PML.** One PML for every solver: the dimensionless stretch
  `s = 1 - i*strength*(depth/thickness)**order`, sampled at cell centres
  with the physical thickness (not rounded to cells). Waveguide and scattering
  PMLs previously used a conductivity in S/m (strength
  `sigma/(eps0*omega)`) rounded to whole cells; periodic solvers already used
  this form. Guided modes are unaffected; leaky and scattered fields change
  slightly. One-sided PMLs (`"x+"`) now work in every solver that has a PML.
- **Subpixel averaging.** `mesh(subpixels=8)` everywhere. 1D waveguides used
  100; band structures and scattering sampled cell centres only and now
  average dielectrics too (`subpixels=1` restores point sampling).
- **Guess.** An explicit `guess` is honoured as the shift. 1.x silently raised
  waveguide guesses below the largest material index. `guess=None` is
  automatic for every modal solver (periodic solvers used `1.0`).
- **Eigensolver default.** `method="arpack"` for every solver (3D periodic
  used refined Arnoldi); pass `method="refined"` for difficult 3D cells.
- **Polarization default.** 2D periodic solves `"both"` (was `"TE"`).
- **Band structure at Gamma.** The default shift is now just below the
  spectrum. The 1.x shift at zero sat on the static mode, so the other Gamma
  bands lost up to ~1e-4 relative accuracy and varied between runs.
- **Residuals.** `info.residuals` is the normwise backward error for every
  eigenproblem (meaningful for zero eigenvalues); `info.converged` checks it.
- **Constants.** CODATA `epsilon_0`/`mu_0` throughout (the waveguide family
  used truncated values; ~1e-10 relative effect).
- **Strict ranges.** Domain and geometry ranges must be `(min, max)` pairs.
- **CUDA discovery.** cuDSS is no longer searched for inside the repository's
  `build/` folder; set `CUDSS_LIBRARY` or install `nvidia-cudss`.

### Fixed

- HDF5 mappings with more than ten entries were read back out of order.
- The band-structure Gamma point could miss or misplace degenerate bands
  (see above).

### Internal

- Shared core modules: Yee operators built by Kronecker products (replacing
  per-family Python loops), one conductor-footprint and material-averaging
  rule, one PML, one eigensolver entry point (`fdfd.core.eigs.solve`), one
  result hierarchy and one persistence header.
- Library code shrank from 12,100 to 8,700 lines (legacy standalone APIs, Tk
  viewers, per-component constraint paths and duplicated helpers removed),
  while every public method gained a NumPy-style docstring.
- ruff lint/format configuration, pre-commit hook and GitHub Actions CI.

## 1.1.2

Last release with the `fdfd_*` package layout.

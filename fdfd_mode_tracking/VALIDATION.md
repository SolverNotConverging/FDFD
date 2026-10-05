# Implemented validation

Updated 2026-09-21. Validation concerns frequency-domain tracked modes and
exported physical port fields. Actual FDTD execution is explicitly out of scope.

Run the FDFD waveguide, mode-tracking and curated documentation tests to validate
this implementation, then run `scripts/check_documentation.py` for RST and link
checks. The cutoff, open-guide, anisotropic-crossing and degenerate-guide
experiments below describe the exercised numerical cases.

## Automated numerical coverage

The tests under `tests/fdfd/mode_tracking/` cover:

| Scenario | Checks |
| --- | --- |
| Physical PEC plates and rectangular guide below cutoff | Analytical beta/decay, finite common norm, negligible real power, bound eligibility |
| TE and TM cutoff continuation | Persistent branch, lambda sign transition, adaptive cutoff bracket, explicit excluded interval |
| Exact discrete 2D cutoff | Preserve and track the bound identity from the high-frequency seed, skip singular division, show a cutoff star without a false non-bound x, and block singular profile export |
| Physical Maxwell phase | TE/TM E/H ratios, common scaling, unchanged raw fields |
| Source direction reversal | Reverse beta and appropriate E/H components together; signed axial power |
| Passive material loss | Complex beta retained; a verified bound lossy mode remains eligible |
| Open dielectric slab | Independent mesh plus two padding checks accept confined guidance |
| Finite-box modes | Smooth numerical modes fail independent domain/edge checks |
| Claimed enclosure without physical walls | Reject certification |
| Declared PML, including zero conductivity | Reject before solving; reject archives missing no-PML provenance |
| Crossing, avoided crossing, random phase and missing candidates | One-to-one identity matching and unmatched states |
| Degenerate square guide and complex unitary rotations | Track the subspace, preserve physical excitation, require explicit subspace export |
| Rank-deficient candidate span | Rank-revealing orthonormalization identifies lost rank |
| Exhausted solve budget | Store unresolved events; missing-frequency export fails |
| HDF5 round trip | Preserve complex fields, residual order, tracks and excluded intervals; load legacy 1.0 sweeps and save conventional-only 1.1 configuration |
| Material-first sweep API | 1D/2D geometry helpers, fixed-grid meshing, solve lifecycle, invalidation after edits |
| Automatic complete-set tracking | Fixed num_modes count, scalar/diagonal/lossy material search guesses, later branch discovery, reacquisition after candidate-window gaps, arbitrary reference frequency, empty initial seed set and HDF5 round trip |
| Interactive GUI | Every candidate gets a field panel; invalid x, degeneracy diamond and cutoff star markers; programmatic frequency selection |

The existing waveguide solver suite also runs against the residual, singular-shift
retry and reconstruction changes. Documentation tests inspect the maintained
public API and RST syntax.

## Runnable experiments

`fdfd_mode_tracking/examples/parallel_plate_cutoff.py` creates a verified sweep
on both sides of the TE1 cutoff, writes `tracked_modes.h5`, and exports a
port-profile table containing complex beta and real/reactive power.

`tracked_parallel_plate_1d.py` exercises the material-first lifecycle and GUI
through cutoff. `tracked_dielectric_waveguide_2d.py` exercises the equivalent 2D
API and open-guide confinement checks. `anisotropic_pec_mode_crossing_1d.py` follows
the independently polarized TE1 and TM1 branches through a true crossing in a
diagonal-anisotropic dielectric. `degenerate_square_waveguide_2d.py` retains the
two-dimensional TE10/TE01 eigenspace and marks both members as degenerate. These
examples completed under the Agg backend; a rendered 1D viewer frame was visually
checked for layout, all-mode coverage, tracked dispersion and status-marker
legibility.

The square-guide example was also rerun over the user's 9–20 GHz band with the
simplified `solve(num_modes=4)` API: every sample returned exactly four modes,
all usable candidates received track IDs, and both degenerate pairs continued
through cutoff. The resulting viewer was rendered and visually inspected.
The same 9–20 GHz sweep with twelve requested modes completed with twelve
tracks and fixed returned counts. Automatic 2D sweeps use a larger ARPACK Krylov
space to retain the fourfold higher-order eigenspace; the returned profile count
does not grow. A regression checks the analytical multiplicities 2, 2, 2, 4, 2.
Anisotropic and open-dielectric examples were rerun with automatic material
shifts and fixed counts; all numerically valid returned modes had track IDs.

Generated results live in `outputs/fdfd_mode_tracking/examples/`; they are
not committed source artifacts.

## Acceptance conventions

Numerical thresholds remain configurable. Defaults are:

- eigenpair and reconstructed field-equation residual <= 1e-8;
- mesh/domain beta drift <= 1e-3 relative to max(abs(beta), k0);
- verification overlap >= .999;
- artificial-edge norm fraction <= 1e-3;
- tracking excludes abs(neff) <= 1e-6 from field export.

A finite residual is not proof of a physical open mode. Confinement decisions
require the additional evidence described in the guide. Near cutoff, grid
sensitivity can make an otherwise trackable mode unresolved for export.

Power is diagnostic. No assertion requires a bound evanescent mode to carry
positive real power; coefficients use a common E/H norm rather than incident
watts. The tests compare complex fields and decay, not a fictitious nonzero
single-mode power normalization.

## Remaining qualification work

- Exact beta=0 reconstruction is intentionally deferred.
- Confinement evidence assumes the factory preserves physical geometry during
  verification; the interface cannot prove this for an arbitrary callback.
- Automatic physical-wall detection currently covers exterior conductor edges,
  not arbitrary interior closed cavities.
- Full independent divergence/interface diagnostics and exceptional-point
  conditioning analysis are not implemented.
- Broader geometry families and lossy SIBC tracking experiments remain useful
  for qualifying the conventional tracker beyond the current fixtures.
- Continuous-spectrum interpolation and current/time-waveform synthesis are
  downstream work. Exports retain unresolved intervals to prevent accidental
  interpolation through cutoff.

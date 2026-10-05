# Bound port mode tracking: implementation status

Updated 2026-09-21. The implemented package is `fdfd_mode_tracking`, in this
folder's `src/` tree. See the [public guide](docs/guide.rst)
and [API reference](docs/API_REFERENCE.rst).

## Accepted decisions

- Port eigenproblems and all verification solves use no PML.
- Injection eligibility requires numerical validity and transverse confinement.
  Both propagating and bound evanescent modes are supported; real power is
  reported independently and is not an eligibility gate.
- Use a common combined E/H field normalization across the band, with an explicit
  complex excitation amplitude at the recorded reference plane.
- Preserve branch identity through longitudinal cutoff when confinement persists.
  Bracket exact cutoff; defer singular beta=0 reconstruction.
- Return frequency-domain tracked modes suitable for downstream injection.
  Actual FDTD execution, current insertion and time-domain validation are outside
  this project scope, as clarified by the user.
- Matching uses the conventional overlap/prediction and subspace algorithms only.

## Implemented data flow

```text
scene factory(frequency, verification request)
  -> enforce no PML, solve candidates and measured residuals
  -> convert physical E/H, common normalization, complex power
  -> independent mesh verification; open-domain padding verification
  -> pair/subspace matching, assignment, phase/basis transport
  -> adaptive samples, candidate expansion, cutoff brackets, reverse audit
  -> TrackedSweep with all raw candidates and decision evidence
  -> sampled PortMode exports or explicit subspace superposition terms
```

The factory is responsible for preserving physical geometry and evaluating
frequency-dependent materials. Grid bounds and spacing are checked. Base samples
must share a physical Yee grid; verification grids may differ.

## Solver changes

Both kernels now expose measured reduced eigenpair residuals. They also measure
reconstructed discrete field-equation residuals: the 1D primary TE/TM equation,
and the 2D reduced first-order E/H equations. These checks complement each other;
they are not a complete independent six-equation/divergence verification.

Public results preserve residual order through the 1D TE/TM merge/selection.
Metadata includes eta0, no-PML declarations, actual exterior conductor coverage,
source eigenpair indices and reconstruction validity.

The 2D reconstruction checks small neff before dividing by it. Singular fields
are marked invalid and filled with NaN in the raw result; the eigenpair remains
available. Tracking uses a larger configurable exclusion threshold and cannot
export those fields. An exactly singular shift-invert factorization retries a
slightly offset search shift without changing the physical operator.

Testing also exposed and fixed the 1D default candidate-sort key reading a list
while Python's sort temporarily presents that list as empty.

## Field, direction and export contract

Phasors use `exp(+i*omega*t-i*beta*z)`. Native magnetic fields satisfy
`H_num=-i*eta0*H_physical`; the adapter applies `H_physical=i*H_num/eta0`.

The normalization is

`N² = integral(|E|² + eta0²|H|²) / area`,

or the length average for 1D. Every normalized profile has `N=1 V/m`.
Quadrature compares each component at its own Yee locations; physical power uses
cell-centred cross products. Power is complex, with units W in 2D or W/m in 1D.
The common spectral coefficient scales every E/H component.

For +z, an evanescent root `beta=-i*alpha` decays away from the port. A -z export
reverses beta and Ez/Hx/Hy together. Arbitrary rotated source planes are not yet
an interface option.

Exports contain physical E/H, coordinates, beta, amplitude and phase convention,
reference plane, orientation, numerical/confinement evidence and excluded
spectral intervals. They are discrete profiles, not a continuously interpolated
source. Valid endpoints can straddle an unresolved cutoff interval, but export
metadata explicitly sets `continuous_band_certified=False`. A downstream
consumer must resolve or explicitly exclude that interval.

## Confinement and identity

A declared physical enclosure must have actual exterior PEC/PMC/SIBC coverage,
and mesh verification must pass. Arbitrary interior enclosures are not
automatically recognized by this conservative exterior-edge check.

Open ports require one refined mesh and two expanded domains at fixed exterior
spacing. Beta and field/subspace stability plus low artificial-edge participation
provide confinement evidence. Suspected radiation/box modes and unresolved
candidates remain visible but ineligible. Lossy bound modes are supported.

Tracking uses lambda=-neff², phase-invariant overlaps, a secant predictor,
one-to-one assignment with explicit unmatched states, global assignment margins,
extra candidate requests, midpoint refinement and bounded work. Reverse
assignment disagreements mark intervals unresolved. Broadly separated branches
may require separate seeded sweeps/search guesses.

A physically enclosed exact-cutoff candidate remains a valid identity seed at
the default low-frequency reference and is joined to its reconstructable
higher-frequency samples by polarization plus lambda prediction. It therefore
is not omitted merely because its reference-sample fields are singular. Singular
exact-cutoff field reconstruction is still blocked from export. An open or
otherwise unconfined cutoff candidate remains marked with a red x.

Degenerate clusters use rank-revealing orthonormalization, principal angles and
unitary alignment. Seed sets expand to include a whole cluster. Smooth basis
transforms and raw eigenpairs are both retained. Ordinary individual-mode export
refuses a cluster identity; explicit coefficients export raw eigenmode terms with
their separate beta values. This avoids presenting a near-degenerate mixture as
one eigenmode. Exceptional-point continuation is not certified.

## Files and use

- Implementation: `src/fdfd_mode_tracking/` (contracts, metrics, adapter,
  assignment, sweep, export, persistence, material-first API, interactive
  visualization).
- Tests: [mode tracking tests](../tests/fdfd/mode_tracking/test_tracking.py)
  and [CPW example tests](../tests/fdfd/mode_tracking/test_coplanar_example.py).
- Examples: [cutoff sweep](examples/parallel_plate_cutoff.py),
  [anisotropic crossing](examples/anisotropic_pec_mode_crossing_1d.py),
  and [degenerate square guide](examples/degenerate_square_waveguide_2d.py).
- Generated sweeps and reports are under ignored `outputs/`.

The package is registered in the existing single-distribution build, pytest
source discovery and curated public API documentation. See
[VALIDATION.md](VALIDATION.md) for current coverage and remaining limitations.

## Material-first sweep and viewer

`ModeTracker1D` and `ModeTracker2D` now mirror the normal FDFD lifecycle:
construct with `frequencies`, add material geometry, mesh once, solve the sweep,
then inspect or save the returned `TrackedSweep`. The wrapper generates the
required refined and padded verification scenes from the stored physical scene.
The lower-level scene-factory API remains available for more specialized geometry.

The high-level solve takes one `num_modes` count and automatically tracks every
usable returned eigenpair. `extra_modes` and high-level `seed_modes` have been
removed. No retry increases the returned count. The default search guess is the
largest material square-root index (by magnitude for complex media), including
the background and all bulk regions; diagonal anisotropy uses the maximum over
principal epsilon/mu pair products. Explicit `neff_guess` remains supported.
Automatic 2D sweeps use a larger Krylov space to recover repeated eigenvalues
without adding returned candidates; otherwise partial degenerate eigenspaces
can spuriously change orientation between samples and fragment tracks.

Discovery runs at every accepted frequency, rather than only at the reference.
Global track arrays are padded with -1 where a branch is absent, and historical
field comparisons reacquire returning branches. New degenerate groups retain
their basis transforms. Discovery origins and actual search shifts are archived.
Confidently tracked non-bound samples have dashed connections with x markers.
Missing/numerically invalid samples and unresolved identity intervals break lines. The
factory API retains explicit seeded tracking and enables automatic behavior
with `seed_modes=None`.

The Matplotlib viewer draws continuous tracked identities and every candidate
returned by every main/adaptive frequency solve. Its paired dispersion axes show
phase and decay/loss indices. Clicking either axis or moving the frequency slider
updates a field grid containing all candidates. Component and quantity controls
apply to the whole grid. Invalid candidates receive red x overlays in dispersion
and field plots; degenerate candidates use diamonds and cutoff-bracket candidates
use stars. Versioned sweep archives can reopen the same viewer without solving.

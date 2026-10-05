Tracked bound port modes
=========================

``fdfd_mode_tracking`` follows propagating and bound evanescent modes returned by
the FDFD waveguide solvers. It supplies physical frequency-domain E/H fields for
port injection; it neither runs nor validates an FDTD simulation. All candidate
and verification eigenproblems must have no PML, including zero-conductivity PML.

Quick start
-----------

The material-first API follows the normal FDFD lifecycle: construct a tracker,
add geometry, mesh, solve, and inspect the result. ``num_modes`` is the number of
returned modes at every requested or adaptive frequency. All usable returned
modes are tracked automatically; there is no separate extra-candidate count or
seed selection in this API. The count stays fixed during refinement.

The solver displays a tqdm progress bar by default. Each completed frequency
evaluation advances it once; adaptive frequencies increase the total. The
current frequency and primary/mesh/padding stage appear alongside the counter.
Verification solves do not count the same frequency again. A completed
evaluation is not necessarily injection eligible. Use
``tracker.solve(num_modes=4, progress=False)`` (or ``track_modes(...,
progress=False)``) to suppress progress output.

By default, the search uses the largest material ``sqrt(epsilon_r*mu_r)``
(ranked by magnitude for complex materials), including the background and bulk
geometry. For diagonal anisotropy the maximum over principal epsilon/mu pairs
provides a conservative search bound. PEC/PMC and surface impedances do not
contribute a bulk index. ``neff_guess`` remains an optional explicit override.

.. code-block:: python

   from fdfd_common import Material
   from fdfd_mode_tracking import ModeTracker2D, PortSpec

   tracker = ModeTracker2D(
       frequencies=[5e9, 6e9, 8e9],
       x_range=(-5e-3, 5e-3), y_range=(-5e-3, 5e-3),
       port=PortSpec(boundary="open", name="input port"),
   )
   tracker.add_circle(center=(0., 0.), radius=1e-3,
                      material=Material(name="core", epsilon=4.))
   tracker.mesh(resolution=(40, 40))
   sweep = tracker.solve(num_modes=4)
   sweep.save("fdfd_mode_tracking/outputs/tracked_modes.h5")
   sweep.show(component="E")

The interactive viewer plots every candidate returned by every primary and
adaptive solve. Continuous curves identify tracked branches. A frequency slider
and clicks on either dispersion panel select the solved frequency; the lower grid
then shows every candidate field at that frequency. Component and quantity
selectors update all panels. Invalid candidates receive a red ``x`` overlay,
degenerate candidates use diamonds, and cutoff-bracket candidates use stars.
The two dispersion panels show ``Re(neff)`` and ``-Im(neff)`` so valid evanescent
modes remain visible below cutoff. ``E`` and ``H`` display cell-centred vector
magnitudes; Cartesian components support magnitude, real, imaginary, and phase.

Each global track ID has a distinct, deterministic color shared by both
dispersion panels, its candidate markers, and its 1D field curves. Colors do not
repeat after ten or twenty tracks, or change with frequency or eligibility.
Untracked candidates remain gray; 2D field maps retain their scalar colormap.

``plot()`` creates the same interactive figure without calling ``pyplot.show``.
Loaded sweeps retain everything needed by the viewer. Branches first returned at
a later frequency receive new track IDs. An absent mode has candidate index -1;
matching against earlier samples can recover its identity when it returns.
The total number of track IDs can therefore exceed ``num_modes`` over a sweep.
Tracking identity is independent of injection eligibility. Numerically valid
unbound or confinement-unresolved profiles remain connected by dashed lines
with their existing track colors and red x markers. An edge is solid only when
both endpoints are injection eligible; otherwise a confidently tracked edge is
dashed, including transitions into or out of eligibility. Lines break at absent
or numerically invalid samples and across unresolved identity intervals.
Reliable exact-cutoff eigenpairs retain spectral connections (dashed at the
non-exportable sample); bound cutoff stars still have no red x. Cutoff brackets
alone do not break identity lines. Unresolved identity is never filled by simply
sorting eigenvalues. Degenerate identities continue as subspaces. These display
rules do not relax confinement verification or export restrictions.

Advanced factory API
--------------------

``track_modes`` remains available for parameterized geometries. Its factory
receives frequency in hertz and a ``VerificationSpec`` and returns a fresh,
meshed ``ModeSolver1D`` or ``ModeSolver2D``. During verification,
``mesh_factor=2`` halves cell spacing at fixed bounds; ``padding_fraction=.25``
or ``.5`` adds exterior cells at fixed spacing. Padding is requested only for
open ports. Frequency-dependent materials must be evaluated by the factory.

The factory API supports ``seed_modes=None`` for the same automatic discovery
behavior. Its explicit seed tuples (default ``(0,)``) retain targeted tracking
for existing callers. ``reference_frequency`` optionally selects one of the requested frequencies;
otherwise the lowest frequency seeds the sweep. A seed that is transversely
bound but exactly at longitudinal cutoff is retained by lambda and polarization
and continued into higher-frequency samples with reconstructable fields. Seed
indices refer to the reference solve's returned array only. An entire degenerate
cluster is included when any member is seeded.
``sweep.metadata['seed_modes']`` gives the initial track map. Automatic sweeps
record ``branch_discovered`` events for subsequent IDs and ``num_tracks`` in
metadata; each sample's ``candidate_indices`` maps all global IDs to that solve.
Base samples must share exactly the same grid. For widely separated target
branches, use separate seeded sweeps/search guesses to ensure candidate coverage.

Confinement and eligibility
----------------------------

``PortSpec(boundary="enclosed")`` declares that the exterior conductors are
physical walls. The adapter checks that actual PEC/PMC/SIBC cells cover every
outer edge and verifies the modes on a refined mesh. This conservative enclosure
test does not automatically recognize an arbitrary interior closed cavity.

``boundary="open"`` requests mesh refinement and two padding variants. A bound
candidate must have stable beta and fields, and small norm in the outer ten
percent of each domain. The checks establish numerical evidence within configured
tolerances, not a mathematical proof for every possible material/geometry.
The factory is responsible for keeping physical geometry unchanged. The tracker
checks grid bounds/spacing; it cannot infer whether a callback changed a material.

``CandidateSet`` separates ``numerical_valid``, ``confinement``, and
``propagation``. ``eligible`` requires numerical validity and ``confinement ==
'bound'``. Evanescent modes are eligible under the same confinement criteria.
Lossy bound modes can have complex beta and nonzero real power below a nominal
lossless cutoff; power or beta's imaginary part alone is never an artifact test.
Unverified and radiation/box-suspect candidates remain stored with a red x and
cannot be exported. At exact longitudinal cutoff, a physical enclosure can still
prove that a branch is transversely bound. The GUI then uses the cutoff star
without a red x and spectral lambda matching carries its identity through the
sample. Singular E/H reconstruction remains separately non-exportable at that
exact sample. An open-boundary cutoff without independent confinement evidence
can retain spectral branch identity, but remains unresolved and receives a red x.

Defaults are eigenpair/field residual at most ``1e-8``, verification beta drift
at most ``1e-3`` relative to ``max(abs(beta), k0)``, verification overlap at least
``0.999``, and artificial-edge norm fraction at most ``1e-3``. These are conservative
starting settings, configurable through ``TrackingConfig``. Strong near-cutoff
mesh sensitivity can make confinement unresolved even when branch tracking works.

Fields, power and cutoff
------------------------

Raw ``ModeSet`` fields remain unchanged. Stored magnetic fields are converted
with ``Hphysical = i*Hnum/eta0``. Every candidate uses the same positive field norm:
the area/length average of ``abs(E)**2 + eta0**2*abs(H)**2`` equals ``1 (V/m)**2``.
Component-native Yee quadrature supplies half weights on boundary nodes.
Power uses cell-centred cross products and is reported as a complex number,
in W for a two-dimensional section and W/m for an invariant-width section.
Its imaginary part is reactive power; amplitudes of evanescent profiles are not
incident watts. ``export(amplitude=...)`` multiplies every component by a common
complex spectral coefficient at the port reference plane.

Phasors use ``exp(+i*omega*t-i*beta*z)``. For ``direction=1``, ``beta=-i*alpha``
decays toward +z. For ``direction=-1`` the export reverses beta and the signs of
Ez, Hx and Hy, keeping Maxwell's relative field phases consistent. Axes remain
x/y with propagation along z; arbitrary rotated port planes require a separate
coordinate adapter.

The tracker predicts ``lambda=-neff**2`` and refines sign-changing lossless
cutoff intervals. ``cutoff_neff=1e-6`` is the default reconstruction exclusion.
The underlying 2D waveguide API uses ``1e-8`` unless called by the tracker.
Singular fields are NaN in raw 2D results and carry
``solve_info['reconstruction_valid'] == False``; normalized tracking arrays use
zero placeholders with explicit invalid status. They are never exported.

``sweep.unresolved_intervals`` and cutoff events record interval uncertainty.
Export returns discrete sampled profiles only; it never interpolates. Valid
endpoints can be exported on either side of cutoff, but their metadata marks
``continuous_band_certified=False`` and retains the excluded interval. A consumer
must not interpolate through that interval or silently synthesize it as zero.
An actual sample inside an unresolved interval is rejected. Exact beta=0 field
reconstruction and continuous-spectrum synthesis are outside this implementation.

Assignment and degeneracy
-------------------------

Matching combines phase-invariant complex overlap with an eigenvalue predictor,
one-to-one assignment, explicit unmatched choices and global assignment margins.
Ambiguity requests extra candidates or midpoint solves. Default limits are 200
eigensolves, depth 8, and minimum relative frequency step ``1e-5``. Verification
solves count toward the budget. Exhaustion preserves diagnostic events and blocks
uncertified export. A reverse pair audit marks inconsistent intervals unresolved.

Degenerate clusters use rank-revealing orthonormalization, principal angles and
complex Procrustes transport. Ordinary single-mode export refuses cluster members
because the tracked identity is the subspace. Supply explicit smooth-basis
coefficients instead:

.. code-block:: python

   from fdfd_mode_tracking import export_subspace

   terms = export_subspace(sweep, frequency=5e9, cluster_index=0,
                           coefficients=[1.0, 0.0])

The result contains raw-eigenmode terms with separate beta values and stored basis
transforms. This preserves a chosen excitation without calling a near-degenerate
mixture a single eigenmode. Rank loss and merge/split ambiguities are conservative;
exceptional-point continuation is not certified.

For a superposition, compute total power from the summed complex E/H fields.
Adding the individual terms' powers omits interference; paired evanescent fields
can transfer real power even when each separate lossless term has zero real power.

Persistence
-----------

``load_sweep(path)`` restores the versioned ``cem-fdfd-tracking`` HDF5 record.
It includes raw candidates, coordinates, normalization, residuals, physical scene
context, boundary/PML provenance, phase and cluster transforms, all adaptive
samples and unresolved events. Existing waveguide archives without explicit
no-PML provenance are unsuitable for certified tracking/export.

The `mathematical reference <mathematics.rst>`_ gives the implemented equations,
assignment costs, normalization, subspace transport, cutoff rules, confinement
tests, including current limitations.

New sweep archives use schema 1.1 and store only conventional tracking controls.
Schema 1.0 sweeps remain readable: the retired scoring-weight field is ignored,
while historical events and computed results are retained without recomputation.

See the `runnable examples <../examples/README.rst>`_.

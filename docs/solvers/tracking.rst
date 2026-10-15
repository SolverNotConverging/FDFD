Mode tracking
=============

:class:`~fdfd.ModeTracker1D` and :class:`~fdfd.ModeTracker2D` solve a
waveguide cross-section at every requested frequency and follow each branch
by field overlap, through crossings, degeneracies and cutoff. Every base solve
is checked by an independent refined-mesh solve (and, for open ports, by
padded domains), and only verified, bound modes can be exported as port
profiles. No PML is used anywhere: confinement is established by evidence.

Example
-------

.. code-block:: python

   import numpy as np
   import fdfd
   from fdfd import materials
   from fdfd.core.constants import C0

   width, cells = 22.86e-3, 96
   dx = width / cells
   tracker = fdfd.ModeTracker1D(frequencies=C0 / (2 * width) * np.array([0.8, 0.9, 1.1, 1.2]),
                                x_range=(-dx, width + dx), port=fdfd.PortSpec(boundary="enclosed"))
   tracker.add_layer(x_range=(-dx, 0.0), material=materials.PEC, name="left plate")
   tracker.add_layer(x_range=(width, width + dx), material=materials.PEC, name="right plate")
   tracker.mesh(resolution=cells + 2, subpixels=1)
   sweep = tracker.solve(num_modes=4, polarization="TE")
   sweep.show()                      # dispersion of every track plus all candidate fields
   ports = sweep.export(track=0)     # certified, power-normalized port profiles

``PortSpec(boundary="enclosed")`` declares that the outer conductors are
physical walls; ``"open"`` means the domain edges are artificial and must be
shown irrelevant by padding. :class:`~fdfd.TrackingConfig` holds the
thresholds below. :func:`fdfd.tracking.track_modes` is the lower-level entry
point: it takes a factory ``make_solver(frequency, VerificationSpec)`` instead
of a tracker (see ``examples/tracking/1d_tracked_parallel_plate.py``).

The algorithm answers three separate questions:

* **Identity:** which candidate continues a previously observed eigenbranch?
* **Numerical validity:** are its eigenpair and reconstructed fields reliable?
* **Confinement:** is it transversely localized or enclosed by physical walls?

A track can exist without being eligible for injection. Neither zero real
power nor longitudinal evanescence makes a mode spurious. The exported data
are frequency-domain profiles; continuous-spectrum synthesis and time-domain
validation are outside this package.

.. contents:: Mathematics
   :local:
   :depth: 2

1. Conventions and the eigenproblem
-----------------------------------

With reference plane :math:`z_0`, the physical field is

.. math::

   \mathbf E(\mathbf r,t)=\operatorname{Re}\left\{
      a(f)\,\mathbf e(x,y;f)\,e^{i\omega t-i\beta(f)(z-z_0)}\right\},
   \qquad n=\beta/k_0,

and likewise for H. The waveguide solvers store
:math:`\mathbf H_{\rm num}=-i\eta_0\mathbf H` (see :doc:`waveguide`); the
tracker converts to physical H with :math:`\mathbf H=(i/\eta_0)\mathbf H_{\rm num}`.
The reduced eigenproblem is :math:`A(f)u_j=\lambda_j u_j` with
:math:`\lambda_j=-n_j^2`. Tracking predicts :math:`\lambda`, not the sorted
index or :math:`\beta`. The operator may be complex and non-Hermitian; the
overlaps below use a positive field-space inner product, not a biorthogonal
left/right product.

The automatic search shift is :math:`\sigma=-n_{\rm guess}^2` with
:math:`n_{\rm guess}=1.01\max|\sqrt{\varepsilon_p\mu_q}|` over the principal
axes of every bulk material (conductors excluded). 2D automatic sweeps enlarge
the Krylov space to :math:`\min(N,\max(6k+1,64))` to keep repeated
eigenvalues. Finite candidate coverage is never a completeness proof.

2. Normalization and complex power
----------------------------------

With section measure :math:`M` (area in 2D, length in 1D), the profile norm is

.. math::

   N_j^2=\frac{1}{M}\int_\Omega
      \left(\sum_{\alpha}|E_{\alpha,j}|^2+\eta_0^2\sum_{\alpha}|H_{\alpha,j}|^2\right)d\Omega,

evaluated with each component's native Yee quadrature (node samples have half
weight at the ends). Dividing every component by :math:`N_j` gives the profile
with RMS norm 1 V/m; the user's coefficient multiplies that profile at the
reference plane. The norm is not an energy density, stays positive with lossy
materials, and averages over the whole domain, so enlarging the domain changes
the normalized amplitude; verification re-normalizes on the reference domain.

Complex axial power is computed separately on cell-centred fields,

.. math::

   \mathcal P_j=\frac12\int_\Omega(E_{x,j}H_{y,j}^*-E_{y,j}H_{x,j}^*)\,d\Omega,

in W (2D) or W/m (1D). A lossless evanescent profile can have zero real power
and a finite norm.

3. Numerical validity
---------------------

``result.info.residuals`` is the normwise backward error of each eigenpair,
:math:`\|Au-\lambda u\|_2/((\|A\|+|\lambda|)\|u\|_2)`. For a reconstructed
equation :math:`\sum_\ell t_\ell=0` the field residual is
:math:`\|\sum_\ell t_\ell\|_\infty/\sum_\ell\|t_\ell\|_\infty`; in 2D this is
the larger residual of :math:`Ph_t-in\,e_t=0` and :math:`Qe_t-in\,h_t=0`, in
1D the TE/TM Ampère/Faraday rows on unconstrained samples.

A candidate is numerically valid when its norm is finite and nonzero, both
residuals are at most ``residual_tolerance``, and :math:`|n|>` ``cutoff_neff``.
A small eigenpair error does not guarantee a small field error:
reconstruction differentiates and, in 2D, divides by :math:`n`.

4. Branch matching
------------------

For normalized old and new vectors,
:math:`O_{ij}=|v_i^\dagger w_j|^2\in[0,1]`, independent of each vector's phase.
With a secant predictor
:math:`\widehat\lambda_i(f_{\rm new})=\lambda_i(f_{\rm cur})+
\frac{f_{\rm new}-f_{\rm cur}}{f_{\rm cur}-f_{\rm prev}}[\lambda_i(f_{\rm cur})-\lambda_i(f_{\rm prev})]`
the pair cost is

.. math::

   C_{ij}=1-O_{ij}+0.1\min(d_{ij}^2,4),\qquad
   d_{ij}=\frac{|\lambda_j^{\rm new}-\widehat\lambda_i|}{\max(1,|\lambda_i^{\rm cur}|)}.

Invalid candidates, pairs below ``overlap_min`` and (in 1D) TE/TM mismatches
are forbidden. Confinement is not required for identity, so box modes can be
followed for diagnosis without becoming eligible.

The assignment adds one dummy column of cost ``unmatched_cost`` per old track
and solves the linear assignment problem
(`SciPy linear_sum_assignment <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html>`_),
so no candidate serves two tracks. A proposed pair is accepted only if
:math:`C_{ij}<` ``unmatched_cost`` and its global margin
:math:`J_{-ij}-J_*` (the optimum with that pair forbidden, minus the
optimum) is at least ``assignment_margin``. Accepted fields are multiplied by
:math:`e^{-i\arg(v_i^\dagger w_j)}` so the aligned overlap is real and
nonnegative; seeds and new branches make their largest entry real.

5. Degenerate subspaces
-----------------------

``cluster_gap`` defines a spectral neighbourhood. Basis rotation requires the
stricter ``degeneracy_tolerance`` (default 1e-8): every pair in a group must
satisfy the relative gap bound. This complete-link rule prevents a chain of
small gaps from merging resolved branches. Known 1D TE/TM sectors are kept
separate, including seeds and newly discovered branches.

At continuation steps, a group must also remain degenerate under the secant
prediction. Thus an accidental crossing does not erase established branch
variations. Persistent pairs may be matched through a larger crossing group
when projection onto the old span identifies an unambiguous subset of raw
candidates. ``subspace_margin`` controls that subset separation. A fully
mixed larger eigenspace without identifiable subsets remains unresolved and
triggers adaptive sampling.

Seed bases first resolve longitudinal E/H nullspaces, then mirror sectors
whose reflections preserve the resolved span. Remaining multiplicities retain
a subspace identity. These smooth-basis labels are stored as ``characters``
in the cluster records, separately from raw candidate evidence. The field
viewer applies these transformations before plotting components or magnitudes
and displays the transported E/H and parity labels. Individual modes use their
tracked phase; untracked candidates retain their raw profiles.

Each accepted span is orthonormalized by polar decomposition and compared
using the smallest squared principal overlap. Competing proposals sharing
any member need a cost margin of at least ``assignment_margin``. Procrustes
transport stores the raw-mode-to-smooth-basis transformation. Reverse audits
permit permutations within each accepted subspace, never across unrelated
groups. Individual export refuses cluster members; use
:func:`fdfd.tracking.export_subspace` with explicit coefficients.

Candidate evidence includes ``character``: TE (H-type), TM (E-type), TEM,
hybrid or unresolved, longitudinal E/H fractions, and reflection parity for
each transverse axis. Reflection treats E as a polar vector and H as an axial
vector; parity 0 means indefinite. Definite contradictory E/H or parity
sectors cannot match as individual branches. This preserves distinct symmetry
sectors through crossings without imposing a label on a mixed degenerate
basis. Parity is measured field evidence, not a proof of geometric symmetry
or symmetry protection. Hybrid modes may change character continuously.

Mesh and padding variants use the same strict grouping and sector rules.
Subspace verification requires every participating variant member to be
numerically valid and satisfy the drift and edge checks. Character evidence,
group transformations and the assignment policy are retained in saved sweeps.

6. Continuation and cutoff
--------------------------

The root choice gives positive phase index, or negative imaginary index for a
purely evanescent mode, so :math:`\beta=-i\alpha` decays as
:math:`e^{-\alpha(z-z_0)}`. Candidates are labelled ``cutoff_unresolved``
(:math:`|n|\le` ``cutoff_neff``), ``evanescent``, ``propagating`` or
``complex``; these are propagation labels, not confinement certificates.

Near cutoff :math:`\lambda` varies smoothly while :math:`\beta` is steep
(:math:`\delta\beta\simeq-k_0^2\,\delta\lambda/2\beta`), hence the
:math:`\lambda` predictor. An interval is a cutoff crossing when both ends
are nearly real and :math:`\operatorname{Re}\lambda` changes sign; it is
bisected, as are intervals with unmatched tracks, until ``max_depth``,
``max_solves`` or ``min_relative_step`` stops it. The final bracket is stored
with its width as ``uncertainty_hz`` (not a statistical interval). At exact
cutoff the 2D fields are NaN; an assignment touching a cutoff candidate with a
reliable eigenpair uses :math:`|\lambda_j-\widehat\lambda_i|/\max(1,|\widehat\lambda_i|)`
without field overlap, and such identities are never exported.

Automatic tracking seeds every usable candidate, gives later unassigned
candidates new track IDs (``-1`` marks absence), recovers absent tracks
against older samples, and finally matches adjacent samples in reverse;
disagreement marks an unresolved interval.

7. Confinement verification
---------------------------

Each sample is re-solved on a grid with twice the cells at fixed bounds and,
for open ports, on domains padded by 25% and 50% of the cells at the same
spacing. Variant fields are interpolated onto the reference lattices and
re-normalized. For base mode :math:`i` the partner maximizes
:math:`O_{ij}-\min(d^\beta_{ij},1)` with
:math:`d^\beta_{ij}=|\beta'_j-\beta_i|/\max(|\beta_i|,k_0)`. A variant passes
when its partner is numerically valid, the drift is at most
``verification_beta_tolerance``, the overlap at least
``verification_overlap``, and either the conductor masks close every outer
edge (enclosed ports) or the outer-band energy fraction

.. math::

   q_{\rm edge}=\frac{\sum_{p\in B}I_p}{\sum_{p\in\Omega}I_p},\qquad
   I_p=\sum_\alpha|E_{\alpha,p}|^2+\eta_0^2\sum_\alpha|H_{\alpha,p}|^2,

over the outer 10% of cells is at most ``edge_fraction_max`` (open ports).
The confinement state is ``bound`` when all checks pass,
``radiation_or_box_suspect`` for a complete failing open check, and
``unresolved`` otherwise. Injection eligibility is numerical validity and
``bound``; there is no real-power threshold. Mesh accuracy and confinement are
not fully separated: a physical enclosure becomes ``unresolved`` if a
non-cutoff candidate fails its mesh check, so inspect overlap, drift,
residuals and edge fraction separately.

8. Export
---------

:meth:`~fdfd.TrackedSweep.export` returns sampled, eligible profiles only and
rejects missing identities, unsolved frequencies, unresolved intervals,
failed reverse audits and cluster members; nothing is interpolated. For a port
facing :math:`-z`, reflection gives :math:`\beta'=-\beta`,
:math:`(E_x,E_y,E_z)\to(E_x,E_y,-E_z)` and
:math:`(H_x,H_y,H_z)\to(-H_x,-H_y,H_z)`; exported power is
:math:`d|a|^2\mathcal P`. Superpositions must sum fields before computing
power, since cross terms do not vanish.

9. Defaults
-----------

.. list-table:: ``TrackingConfig``
   :header-rows: 1
   :widths: 40 18 42

   * - Parameter
     - Default
     - Meaning
   * - ``tol``
     - 0
     - Eigensolver tolerance (0 = machine precision)
   * - ``residual_tolerance``
     - 1e-8
     - Eigenpair and field residual ceiling
   * - ``cutoff_neff``
     - 1e-6
     - Near-zero-index reconstruction/export exclusion
   * - ``overlap_min``
     - 0.9
     - Minimum squared matching overlap
   * - ``assignment_margin``
     - 0.02
     - Minimum global alternative-cost gap
   * - ``unmatched_cost``
     - 0.65
     - Dummy assignment cost
   * - ``cluster_gap``
     - 1e-5
     - Relative eigenvalue-neighbourhood scale
   * - ``degeneracy_tolerance``
     - 1e-8
     - Maximum pairwise relative gap for basis rotation
   * - ``subspace_margin``
     - 0.02
     - Minimum projection gap selecting a subgroup
   * - ``verification_beta_tolerance``
     - 1e-3
     - Drift normalized by max(abs(beta), k0)
   * - ``verification_overlap``
     - 0.999
     - Minimum squared verification overlap
   * - ``edge_fraction_max``
     - 1e-3
     - Maximum outer-band norm fraction
   * - ``max_depth``, ``max_solves``
     - 8, 200
     - Refinement depth and total eigensolve budget
   * - ``min_relative_step``
     - 1e-5
     - Relative interval refinement floor

The tracker sets the candidate counts from ``num_modes``. Verification solves
count against the budget.

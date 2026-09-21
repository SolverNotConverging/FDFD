Mathematics of FDFD mode tracking
========================================

This is a mathematical reference for the current ``fdfd_mode_tracking``
implementation, not a claim that every computed branch is a guided mode.
See the `usage guide <guide.rst>`_ and the
`examples <../../../../examples/fdfd/mode_tracking/README.rst>`_.

The algorithm answers three different questions:

* **Identity:** which candidate continues a previously observed eigenbranch?
* **Numerical validity:** are its eigenpair and reconstructed fields reliable?
* **Confinement:** is it transversely localized or enclosed by physical walls?

A track can exist without being eligible for injection. Neither zero real
power nor longitudinal evanescence makes a mode spurious. No primary or
verification solve uses PML. The package exports frequency-domain profiles;
continuous-spectrum current synthesis and FDTD validation are not implemented.

.. contents:: In this document
   :local:
   :depth: 2

1. Conventions and the eigenproblem
------------------------------------------

Let the reference plane be :math:`z_0`, frequency be :math:`f`, and
:math:`\omega=2\pi f`. The physical field convention is

.. math::

   \mathbf E(\mathbf r,t)=\operatorname{Re}\left\{
      a(f)\mathbf e(x,y;f)e^{i\omega t-i\beta(f)(z-z_0)}\right\},
   \qquad k_0=\omega\sqrt{\epsilon_0\mu_0},\qquad n=\beta/k_0.

The same convention applies to H. With :math:`\eta_0=\sqrt{\mu_0/\epsilon_0}`,
the stored magnetic fields obey

.. math::

   \mathbf H_{\rm num}=-i\eta_0\mathbf H_{\rm phys},\qquad
   \mathbf H_{\rm phys}=\frac{i}{\eta_0}\mathbf H_{\rm num}.

With material loss included in constitutive tensors, source-free phasor
Maxwell equations are

.. math::

   \nabla\times\mathbf E=-i\omega\boldsymbol\mu\mathbf H,\qquad
   \nabla\times\mathbf H=i\omega\boldsymbol\epsilon\mathbf E.

The dimensionless reduced eigenproblem is

.. math::

   A(f)u_j=\lambda_j u_j,\qquad
   \lambda_j=-n_j^2,\qquad \beta_j=k_0n_j.

Tracking predicts :math:`\lambda`, not the sorted eigenvalue index or a direct
linear extrapolation of beta. The operator may be complex and non-Hermitian.
The overlap used below is a positive field-space inner product, not a
biorthogonal left/right eigenvector product.

2D reconstruction contract
~~~~~~~~~~~~~~~~~~~~~~~~~~

The staggered derivatives include :math:`1/(k_0\Delta x)` and
:math:`1/(k_0\Delta y)`. The kernel constructs transverse blocks P and Q:

.. math::

   P h_t=i n e_t,\qquad Q e_t=i n h_t,\qquad
   P Q e_t=-n^2e_t.

Here :math:`h_t` contains numerical magnetic fields. PEC electric and PMC
magnetic constrained degrees of freedom are removed. Surface impedances
modify the boundary equations. In code, P is restricted to free magnetic
columns, Q to free magnetic rows, and PQ to free electric rows and columns.
Zero constrained components are restored after solving.

Reconstruction uses

.. math::

   h_t=\frac{Qe_t}{i n},\qquad
   E_z=\epsilon_{zz}^{-1}
       (D_{H_y\to E_z}H_y-D_{H_x\to E_z}H_x),

.. math::

   H_{z,\rm num}=\mu_{zz}^{-1}
       (D_{E_y\to H_z}E_y-D_{E_x\to H_z}E_x).

The H components in the expression for :math:`E_z` are numerical fields too.
Material inverses are taken only on unconstrained degrees of freedom.
Division by n causes the generic 2D reconstruction to become singular at cutoff.
Material rasterization and the entries of P and Q belong to the
`waveguide formulation <../waveguide_modes/guide.rst>`_; the tracker does not
replace that discretization.

1D specialization
~~~~~~~~~~~~~~~~~

For variation in x only, let :math:`D_{e h}` map electric to magnetic samples
and :math:`D_{h e}` map magnetic to electric samples. Material symbols below
are sampled diagonal matrices. The TE and TM operators are

.. math::

   A_{\rm TE}=-\mu_{xx}(D_{h e}\mu_{zz}^{-1}D_{e h}+\epsilon_{yy}),\qquad
   A_{\rm TM}=-\epsilon_{xx}(D_{e h}\epsilon_{zz}^{-1}D_{h e}+\mu_{yy}).

The primary unknowns are :math:`E_y` and :math:`H_{y,\rm num}`, respectively.
Reconstruction uses

.. math::

   H_{x,\rm num}=i n\mu_{xx}^{-1}E_y,\quad
   H_{z,\rm num}=\mu_{zz}^{-1}D_{e h}E_y,\quad
   E_x=i n\epsilon_{xx}^{-1}H_{y,\rm num},\quad
   E_z=\epsilon_{zz}^{-1}D_{h e}H_{y,\rm num}.

These formulas need not divide by n, but the tracker currently applies the
same cutoff exclusion to 1D and 2D export.

Search shift and candidate count
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The automatic material search collects

.. math::

   \mathcal N=
   \left\{\sqrt{\epsilon_{r,p}^{(m)}\mu_{r,q}^{(m)}}\right\}_{m,p,q},
   \qquad n_{\rm guess}=\underset{n\in\mathcal N}{\arg\max}|n|,
   \qquad \sigma=-n_{\rm guess}^2.

Background and bulk materials are included; PEC, PMC and surface impedances
are excluded. For diagonal anisotropy all principal epsilon/mu pairs are
considered, including crossed pairs. This is a search heuristic, not proof
of a maximum guided index for arbitrary anisotropic or lossy media.

Complex shift-invert transforms the eigenproblem to

.. math::

   (A-\sigma I)^{-1}u_j=(\lambda_j-\sigma)^{-1}u_j.

This targets candidates near the shift; see
`SciPy eigs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`_.
The material-first API returns ``num_modes`` candidates at each frequency;
it does not automatically increase this count. The 2D automatic solve uses
Krylov dimension :math:`\min(N,\max(6k+1,64))` for candidate count k.
An exactly singular shift is retried at

.. math::

   \sigma'=\sigma+10^{-7}\max(1,|\sigma|)(1+0.37i),

without changing A. The explicitly seeded factory workflow can expand the
candidate count up to ``max_candidates`` after an unmatched step.
Finite candidate coverage is never a completeness proof.

2. Positive normalization and complex power
-------------------------------------------

Let :math:`\Omega` be the computational section, with measure
:math:`M=|\Omega|`: area in 2D or length in 1D. The common norm is

.. math::

   N_j^2=\frac{1}{M}\int_\Omega
      \left(\sum_{\alpha=x,y,z}|E_{\alpha,j}|^2+
            \eta_0^2\sum_{\alpha=x,y,z}|H_{\alpha,j}|^2\right)d\Omega.

All H fields in this section are physical. This is a positive profile norm,
not electromagnetic energy density or incident power. It does not use
epsilon/mu energy weights and remains positive with complex materials.
The same definition is used throughout the frequency band.

For component c, let :math:`w_{c,p}` be its native Yee quadrature weight.
Cell-centred samples have coordinate weight :math:`\Delta x`; node samples
have this weight internally and half weight at the endpoints. In 2D, multiply
the coordinate weights. Set :math:`\gamma_c=1` for E components and
:math:`\gamma_c=\eta_0` for H components. Then

.. math::

   N_j^2=\sum_{c,p}\frac{w_{c,p}}{M}|\gamma_c F_{c,p,j}|^2,\qquad
   v_j=\operatorname{stack}_{c,p}\left[
       \sqrt{\frac{w_{c,p}}{M}}\gamma_c\frac{F_{c,p,j}}{N_j}\right],
   \qquad v_j^\dagger v_j=1.

Numerically dividing every E/H component by :math:`N_j` produces the profile
with RMS norm :math:`1\,{\rm V/m}`. The user's complex coefficient multiplies
that profile at the recorded reference plane; it is not a square root of watts.
A nonfinite or effectively zero norm invalidates the profile. Normalized
invalid arrays can contain zero placeholders; these are not physical modes
and cannot be exported.

The norm averages over the computational domain, not just the core. Enlarging
the domain can therefore change normalized amplitude even for the same
physical field shape. Verification re-normalizes on the reference domain.

Complex axial power is calculated separately:

.. math::

   \mathcal P_j=\frac12\int_\Omega
       (E_{x,j}H_{y,j}^*-E_{y,j}H_{x,j}^*)\,d\Omega.

Node-centred components are first averaged onto cell centres, successively
along each staggered coordinate. The sum uses uniform cell area or length.
This is different from the component-native norm quadrature. Units are W
in 2D and W/m in 1D; real and reactive parts are recorded separately.
Scaling all fields by a multiplies power by :math:`|a|^2`.
A lossless evanescent profile can have zero real power and a finite norm.

3. Numerical validity and residuals
-----------------------------------

The measured infinity-norm eigenpair backward error is

.. math::

   r_{\rm eig}=
   \frac{\|Au-\lambda u\|_\infty}
        {(\|A\|_\infty+|\lambda|)\|u\|_\infty}.

The matrix norm is the maximum absolute row sum. A zero denominator gives
an infinite eigenpair residual.

For a reconstructed equation :math:`\sum_{\ell=1}^L t_\ell=0`, the code uses

.. math::

   r_{\rm eq}=\frac{\|\sum_\ell t_\ell\|_\infty}
                    {\sum_\ell\|t_\ell\|_\infty}.

An all-zero equation has residual zero. In 2D, the field residual is the
maximum residual of :math:`Ph_t-i n e_t=0` on free electric rows and
:math:`Qe_t-i n h_t=0` on free magnetic rows. In 1D the checked equations are

.. math::

   D_{h e}H_z+\epsilon_{yy}E_y+i n H_x=0\quad({\rm TE}),\qquad
   D_{e h}E_z+\mu_{yy}H_y+i n E_x=0\quad({\rm TM}),

using numerical H and unconstrained primary-field rows. These checks are not
an independent full continuum Maxwell validation.

Numerical validity requires a finite nonzero norm, finite eigenpair and field
residuals no greater than ``residual_tolerance``, and
:math:`|n|>{\tt cutoff\_neff}`. The eigensolver stopping tolerance is a separate
control. A tiny eigenpair error does not ensure a small field error:
reconstruction includes differentiation, cancellation and, in 2D, division by n.

4. Ordinary branch matching
---------------------------

Overlap and prediction
~~~~~~~~~~~~~~~~~~~~~~

For normalized old and new vectors on the same grid,

.. math::

   O_{ij}=|v_i^\dagger w_j|^2,\qquad 0\le O_{ij}\le1.

Ordinary assignment clips roundoff to this interval. Arbitrary unit complex
phase changes leave the overlap unchanged. At a crossing, eigenvalue order
can exchange without field identities exchanging.

With a previous and current sample available for the same track, use

.. math::

   \widehat\lambda_i(f_{\rm new})=\lambda_i(f_{\rm cur})+
   \frac{f_{\rm new}-f_{\rm cur}}{f_{\rm cur}-f_{\rm prev}}
   [\lambda_i(f_{\rm cur})-\lambda_i(f_{\rm prev})].

Otherwise predict the current value. The pair error and baseline cost are

.. math::

   d_{ij}=\frac{|\lambda_j^{\rm new}-\widehat\lambda_i|}
                     {\max(1,|\lambda_i^{\rm cur}|)},\qquad
   C_{ij}=1-O_{ij}+0.1\min(d_{ij}^2,4).

Numerically unusable candidates, polarization-incompatible pairs, and ordinary
pairs below ``overlap_min`` are excluded. The 1D TE/TM labels constrain ordinary
matching. Confinement is intentionally not required for identity matching:
numerical box branches may be followed for diagnosis without becoming eligible.

Joint assignment and global margins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For m old tracks, append m dummy columns with cost
:math:`u={\tt unmatched\_cost}`. Forbidden real pairs receive cost
:math:`10^6`. Solve

.. math::

   J_*=\min_X\sum_{ij}\widetilde C_{ij}X_{ij},\qquad
   X_{ij}\in\{0,1\},\quad
   \sum_jX_{ij}=1,\quad \sum_iX_{ij}\le1.

Each old track chooses a real candidate or an unmatched column; no real
candidate can serve two tracks. This uses
`SciPy linear_sum_assignment <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html>`_.

For each proposed real pair, forbid it and solve again, yielding
:math:`J_{-ij}`. The global assignment margin is

.. math::

   m_{ij}=J_{-ij}-J_*.

Accept the pair only when :math:`C_{ij}<u` and
:math:`m_{ij}\ge{\tt assignment\_margin}`. This is not simply the difference
between a row's two best costs, and is not a calibrated confidence probability.
Rejected pairs remain unmatched.

Phase transport
~~~~~~~~~~~~~~~

For an accepted pair, multiply every new field component by

.. math::

   p_j=\exp[-i\arg(v_i^\dagger w_j)].

The aligned overlap becomes real and nonnegative. At a seed or a new branch,
the largest-magnitude vector entry is made real and nonnegative. This fixes
a computational gauge, not a physical excitation amplitude. Phases are stored.

5. Degenerate subspaces
-----------------------

Clusters are connected components of the eigenvalue-gap graph:

.. math::

   i\sim j\quad\hbox{if}\quad
   |\lambda_i-\lambda_j|\le
   g\max(1,|\lambda_i|,|\lambda_j|),\qquad g={\tt cluster\_gap}.

Clustering is transitive: the endpoints of a cluster need not directly satisfy
the pairwise bound. Numerical proximity is not proof of exact degeneracy.

Orthonormalization and principal angles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a matrix V of normalized raw candidate vectors, take the thin SVD
:math:`V=U\Sigma W^\dagger`. Retain singular values greater than
:math:`10^{-10}\sigma_{\max}`. At full column rank use polar orthonormalization:

.. math::

   T=W\Sigma^{-1}W^\dagger,\qquad Q=VT=UW^\dagger,\qquad Q^\dagger Q=I.

At reduced rank r, return :math:`T=W_r\Sigma_r^{-1}` and :math:`Q=U_r`.
Cluster matching rejects deficient rank and unequal cluster sizes.

For old and new orthonormal bases,

.. math::

   Q_o^\dagger Q_n=L S R^\dagger,\qquad
   s_\ell=\cos\theta_\ell,\qquad O_{\rm sub}=\min_\ell s_\ell^2.

The worst principal overlap checks the entire span, not one good vector.
Equal-rank cluster matching uses

.. math::

   d_{\rm sub}=\frac{|\overline\lambda_o-\overline\lambda_n|}
                        {\max(1,|\overline\lambda_o|)},\qquad
   C_{\rm sub}=1-O_{\rm sub}+0.1\min(d_{\rm sub}^2,4).

It uses centroid drift rather than the scalar secant predictor.
The unmatched and global-margin rules also apply. Matched cluster members
are reserved before ordinary scalar assignment.

Procrustes transport and excitation coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The unitary alignment is

.. math::

   R_{\rm align}=R L^\dagger,\qquad
   Q_{\rm aligned}=Q_nR_{\rm align},\qquad
   T_{\rm stored}=T_nR_{\rm align}.

It minimizes :math:`\|Q_o-Q_nR_{\rm align}\|_F` over unitary rotations.
For user coefficients c in the smooth basis, raw-mode amplitudes are

.. math::

   a=T_{\rm stored}c,\qquad
   \mathbf F(z)=\sum_j a_j\mathbf F_j(z_0)e^{-i\beta_j(z-z_0)}.

Each term retains its own beta. Rotating distinct eigenvalues gives an explicit
superposition, not a new eigenmode. Exact degeneracy permits arbitrary basis
rotation inside the common eigenspace. Individual export refuses cluster
members; use ``export_subspace`` and explicit coefficients.
Rank-deficient exceptional points and general non-Hermitian completeness
are not certified by this Euclidean subspace procedure.

6. Adaptive continuation and cutoff
-----------------------------------

Root choice and propagation labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The backend takes a square root of :math:`-\lambda`, choosing positive real
phase index, or negative imaginary index for a purely evanescent root.
Its root roundoff threshold is :math:`10^{-12}\max(1,|n|)`. In particular,

.. math::

   \beta=-i\alpha,\quad \alpha>0
   \quad\Longrightarrow\quad
   e^{-i\beta(z-z_0)}=e^{-\alpha(z-z_0)}.

The tracker labels :math:`|n|\le{\tt cutoff\_neff}` as ``cutoff_unresolved``.
Otherwise, set :math:`t=10^{-8}\max(1,|n|)`:

* ``evanescent`` means :math:`|\operatorname{Re}n|\le t` and
  :math:`\operatorname{Im}n<-t`;
* ``propagating`` means :math:`|\operatorname{Im}n|\le t`;
* ``complex`` is the remaining case.

These are propagation labels, not confinement or passivity certificates.
Material/surface loss can make beta complex without making a mode radiative.

Why lambda is preferable near cutoff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a simple uniform lossless vacuum-filled guide with transverse eigenvalue
:math:`k_c^2`, the separated eigenproblem gives

.. math::

   \beta^2=k_0^2-k_c^2,\qquad \lambda=-1+k_c^2/k_0^2.

The beta root is steep near cutoff, while lambda crosses zero smoothly.
At fixed frequency and away from exactly zero beta,

.. math::

   \delta\beta\simeq\frac{\delta(\beta^2)}{2\beta}
                  =-\frac{k_0^2}{2\beta}\delta\lambda.

This explains the lambda predictor and the sensitivity of the current
beta-based verification test. A general lossy guide need not have a real
cutoff frequency. Loss of transverse confinement is a different event
from a zero of beta.

Midpoint refinement and uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A matched interval is flagged as a lossless cutoff crossing when both endpoints
satisfy the near-real test and their real parts have opposite signs:

.. math::

   |\operatorname{Im}\lambda|<10^{-8}\max(1,|\lambda|),\qquad
   \operatorname{Re}\lambda_a\operatorname{Re}\lambda_b<0.

An unmatched existing track also triggers refinement. Insert
:math:`f_m=(f_a+f_b)/2` while depth, solve budget and relative width

.. math::

   h_f=\frac{|f_b-f_a|}{\max(f_a,f_b)}

permit it. Cutoff detection is deferred when a step has unmatched tracks.
At the stopping limit, store the unresolved interval and its full width
:math:`\Delta f=f_{\rm hi}-f_{\rm lo}` as ``uncertainty_hz``.
This is not a statistical confidence interval or a discretization-error bound.
No fitted exact root is reported. A zero at an endpoint does not meet the
strict sign-product test.

Exact-cutoff identity
~~~~~~~~~~~~~~~~~~~~~~~~~~

The 2D kernel avoids division by small n and stores NaNs for unresolved raw
fields. The adapter's zero placeholders remain explicitly invalid.

An assignment touching a cutoff candidate with reliable eigenpair residual
may use polarization and the special cost

.. math::

   C^{\rm cutoff}_{ij}=
      \frac{|\lambda_j-\widehat\lambda_i|}
                   {\max(1,|\widehat\lambda_i|)}

without field overlap. Dummy unmatched choices and global margins still apply.
Adjacent valid fields supply phase when possible. Spectral identity through
cutoff does not certify the singular field for export.

Discovery, recovery and reverse audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatic tracking seeds every usable returned candidate, including reliable
cutoff identities, regardless of confinement. Later unassigned usable
candidates receive new IDs. All samples share the global track columns;
``-1`` means absent. The total track count can exceed ``num_modes``.

If a track is absent in both the preceding and new sample, older anchors may
recover it while already assigned candidates remain reserved. This is matching
against history, not interpolation of missing fields.

Final adjacent pairs are matched in reverse without a secant predictor.
Ordinary candidate identities must agree. When clusters are involved, the
current audit compares sets of common candidate IDs rather than basis vectors.
Disagreement marks an unresolved interval. This detects some identity errors,
but is not independent physical verification.

7. Transverse-confinement verification
--------------------------------------

Independent meshes and resampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every sample requests twice the cells per coordinate at fixed bounds.
Open ports also request two padding fractions, :math:`p=0.25` and
:math:`p=0.5`. On a base axis with N cells and spacing h, add
:math:`q=\operatorname{round}(pN)` cells at each end:

.. math::

   [a,b]\longrightarrow[a-qh,b+qh],\qquad N'=N+2q.

Both padding variants are relative to the base domain, not successive
enlargements. Physical geometry and materials must be unchanged. The built-in
factory reuses the geometry; a custom factory must honor this contract.
Verification checks bounds and spacing, not material identity.

Variant fields are linearly interpolated onto each reference component lattice
and re-normalized on that reference domain. Near staggered exterior offsets,
coordinates are clipped to the variant interpolation lattice; excursions larger
than one variant spacing are rejected. This clamping is a comparison
convenience, not physical field continuation.

Partner selection and beta drift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For base i and variant j, define

.. math::

   d^\beta_{ij}=\frac{|\beta'_j-\beta_i|}{\max(|\beta_i|,k_0)},\qquad
   j_*=\underset{j}{\arg\max}
       [O_{ij}-\min(d^\beta_{ij},1)].

Verification selects the best partner independently for each base candidate;
it is not joint frequency assignment. For equal-sized, close eigenvalue
neighborhoods, it replaces scalar overlap with the worst principal overlap.
These neighborhoods are distances to the selected eigenvalue, not the
transitive graph clusters used in frequency matching.

A variant passes only if its selected candidate is numerically valid,
beta drift is at most ``verification_beta_tolerance``, overlap is at least
``verification_overlap``, and the boundary test below passes.
When :math:`|\beta|<k_0`, beta drift is normalized by :math:`k_0`;
it is not relative error in beta.

Artificial-edge participation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using cell-centred physical fields, define

.. math::

   I_p=\sum_\alpha|E_{\alpha,p}|^2+
                   \eta_0^2\sum_\alpha|H_{\alpha,p}|^2.

Let B be the union of the outermost
:math:`\max(1,\lceil0.1N_d\rceil)` cells at both ends of every axis.
Corner cells are counted once. The edge fraction is

.. math::

   q_{\rm edge}=\frac{\sum_{p\in B}I_p}{\sum_{p\in\Omega}I_p}.

Uniform cell measure cancels. A zero total gives fraction one.
For open ports, both the base and every selected variant must have edge
fraction no greater than ``edge_fraction_max``. Visual localization,
longitudinal decay, or a small imaginary beta alone is insufficient.

States and eligibility
~~~~~~~~~~~~~~~~~~~~~~

An enclosed port requires actual conductor masks covering every exterior
edge in the base and refined domains. PEC, PMC and surface-impedance cells
count; the user's enclosure declaration supplies physical intent.
The test does not automatically recognize arbitrary interior enclosures.

For non-cutoff candidates the current state is:

* ``bound`` if all required comparisons pass;
* ``radiation_or_box_suspect`` for complete but failing open verification;
* ``unresolved`` for missing verification or failing enclosed verification.

A confirmed physical enclosure at exact cutoff has a special residual-and-wall
check that can retain ``bound`` without reconstructed fields. An open cutoff
does not receive this exception. Injection eligibility is

.. math::

   {\rm eligible}_j={\rm numerical\_valid}_j
                  \ \land\ ({\rm confinement}_j={\tt bound}).

There is no real-power threshold. Verification is evidence within tolerances,
not a general mathematical proof. A no-PML artificial box still has eigenmodes;
small eigensolver residuals cannot alone distinguish them from open-guide modes.

Current limitation near cutoff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mesh convergence and confinement are not fully separated in the stored
decision. A physical PEC enclosure becomes ``unresolved`` if a non-cutoff
candidate fails its mesh check. The viewer labels numerically valid but
ineligible candidates ``NON-BOUND``, including unresolved cases; numerically
invalid candidates get ``SPURIOUS/INVALID``. Neither label proves leakage
or a spurious eigenbranch.

Display connectivity is separate from these labels: an assigned branch with
numerically valid endpoints remains connected even when confinement fails.
Segments are solid only if both endpoints are injection eligible; otherwise
they are dashed, with existing x markers retained. Missing or numerically
invalid samples and unresolved identity intervals break connections. Reliable
exact-cutoff identities are the spectral-only exception to the field-validity
requirement: they can be connected with dashed segments but remain non-exportable.
A cutoff bracket alone is not an identity gap. These are visualization rules,
not new physical certification criteria.

For example, beta drift 0.005154 fails a tolerance of 0.005 even with overlap
0.999987 and a confirmed enclosure. This indicates an unmet mesh-accuracy
requirement, not disappearance of physical confinement. Inspect overlap,
beta drift, field/eigenpair residuals and edge fraction separately.
A cutoff-aware lambda verification test and a separate mesh-accuracy state
would improve this distinction, but are not the current rule.

8. Export, orientation and superposition
---------------------------------------------

Export returns sampled, eligible profiles only. It rejects missing identities,
unsolved frequencies, samples strictly inside unresolved intervals, and both
endpoints of failed reverse-audit intervals. Ordinary export also rejects
cluster members. Valid samples on opposite sides of a cutoff bracket can be
exported as discrete data; the excluded interval is not interpolated or
silently filled with zero.

For negative source direction, reflection about the reference plane gives

.. math::

   \beta'=-\beta,\qquad
   (E_x',E_y',E_z')=(E_x,E_y,-E_z),\qquad
   (H_x',H_y',H_z')=(-H_x,-H_y,H_z).

An evanescent root then decays toward negative z. Flipping beta alone would
break the relative field equations. For direction d and coefficient a,
exported complex power is :math:`d|a|^2\mathcal P`. No additional longitudinal
phase is applied at export: amplitude is specified at the stored reference plane.

For multiple terms, sum fields before calculating power:

.. math::

   \mathcal P_{\rm total}=\frac12\sum_{j,k}a_j a_k^*
      \int_\Omega(\mathbf E_j\times\mathbf H_k^*)\cdot\hat{\mathbf z}\,d\Omega.

Adding individual term powers drops the cross terms. Zero real power for
each isolated lossless evanescent term therefore does not imply zero real
power for every evanescent superposition.

Archives retain raw eigenpairs, physical fields, coordinates, scales, power,
phase/subspace transforms, reference plane, orientation, evidence and unresolved
intervals. They do not represent a fitted continuous dispersion model or
certification for a particular FDTD discretization.

9. Defaults and source map
---------------------------

These are ``TrackingConfig`` defaults; examples may override them.
The material-first API sets both candidate counts from ``num_modes``.

.. list-table:: Numerical controls
   :header-rows: 1
   :widths: 40 18 42

   * - Parameter
     - Default
     - Meaning
   * - ``eigensolver_tolerance``
     - 1e-10
     - Iterative eigenvalue stopping tolerance
   * - ``residual_tolerance``
     - 1e-8
     - Eigenpair and field residual ceiling
   * - ``cutoff_neff``
     - 1e-6
     - Near-zero-index reconstruction/export exclusion
   * - ``overlap_min``
     - 0.8
     - Minimum squared matching overlap
   * - ``assignment_margin``
     - 0.02
     - Minimum global alternative-cost gap
   * - ``unmatched_cost``
     - 0.65
     - Dummy assignment cost
   * - ``cluster_gap``
     - 1e-5
     - Relative eigenvalue-neighborhood scale
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
     - Refinement depth and total eigenproblem budget
   * - ``min_relative_step``
     - 1e-5
     - Relative interval refinement floor

Verification solves count against the budget. Missing checks and unresolved
intervals remain explicit; a larger budget does not itself improve a fixed
grid's accuracy.

Authoritative implementation locations:

* `metrics.py <../../../../solvers/fdfd/mode_tracking/src/fdfd_mode_tracking/metrics.py>`_:
  quadrature, normalization, power, interpolation and subspace algebra.
* `assignment.py <../../../../solvers/fdfd/mode_tracking/src/fdfd_mode_tracking/assignment.py>`_:
  costs, clusters, assignment and margins.
* `sweep.py <../../../../solvers/fdfd/mode_tracking/src/fdfd_mode_tracking/sweep.py>`_:
  predictors, adaptive sampling, discovery, cutoff identity and reverse audit.
* `adapter.py <../../../../solvers/fdfd/mode_tracking/src/fdfd_mode_tracking/adapter.py>`_:
  search index, numerical validity and confinement.
* `diagnostics.py <../../../../solvers/fdfd/waveguide_modes/src/fdfd_waveguide_modes/diagnostics.py>`_:
  residuals, boundary provenance and eigenpair search.
* `solver_1d.py <../../../../solvers/fdfd/waveguide_modes/src/fdfd_waveguide_modes/solver_1d.py>`_
  and `solver_2d.py <../../../../solvers/fdfd/waveguide_modes/src/fdfd_waveguide_modes/solver_2d.py>`_:
  reduced Maxwell operators and reconstruction.
* `export.py <../../../../solvers/fdfd/mode_tracking/src/fdfd_mode_tracking/export.py>`_:
  orientation, eligibility enforcement and superposition terms.
* `contracts.py <../../../../solvers/fdfd/mode_tracking/src/fdfd_mode_tracking/contracts.py>`_
  and `visualization.py <../../../../solvers/fdfd/mode_tracking/src/fdfd_mode_tracking/visualization.py>`_:
  defaults, status storage and display semantics.

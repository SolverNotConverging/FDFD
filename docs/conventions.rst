Conventions
===========

These apply to every solver; the solver pages do not repeat them.

Units
   SI throughout: metres, hertz, seconds; fields in V/m and A/m (except where
   noted below); conductivity in S/m; impedance in ohms. Material values are
   relative to vacuum. Constants are the SciPy (CODATA) values in
   :mod:`fdfd.core.constants`.

Time dependence
   Phasors use :math:`e^{+i\omega t}` (``fdfd.TIME_CONVENTION``). Passive
   loss therefore has a *negative* imaginary part,
   :math:`\varepsilon_r = \varepsilon' - i\varepsilon''`, and a passive surface
   impedance has :math:`\operatorname{Re} Z_s \ge 0`.

Propagation
   Modes travel along +z as :math:`e^{-i\beta z}` with
   :math:`\beta = n_{\rm eff} k_0` and :math:`k_0 = \omega / c`. A passive
   forward mode has :math:`\operatorname{Re} n_{\rm eff} \ge 0` and
   :math:`\operatorname{Im} n_{\rm eff} \le 0`; ``attenuation_constant`` is
   :math:`-\operatorname{Im}\beta` in Np/m. Periodic solvers return the Bloch
   factor :math:`e^{-\gamma z}` with :math:`n_{\rm eff} = -i\gamma/k_0` and
   store the periodic envelope.

Band structures
   Bloch fields vary as :math:`e^{-i\mathbf k\cdot\mathbf r}` and the
   eigenvalue is :math:`(\omega/c)^2`; passive loss gives
   :math:`\operatorname{Im} f \ge 0` (decay in time).

Yee grid
   Along each axis, *nodes* are the :math:`N+1` cell faces and *cells* the
   :math:`N` cell centres. :math:`E_c` lies on cell centres along axis
   :math:`c` and on nodes along the other axes; :math:`H_c` is the dual.
   On a periodic axis node :math:`N` is node 0, so both lattices have
   :math:`N` samples. Each field array keeps its native samples; use
   ``result.field_coordinates[name]`` for its coordinates. Materials are
   averaged from cell centres to each component's location.

Materials on the grid
   Dielectrics are averaged by the fraction of each cell they cover,
   sampled at ``subpixels**dimension`` points per cell. Conductors (PEC, PMC,
   surface impedance) take whole cells whose centre lies inside the shape and
   constrain the field samples on and inside that footprint.

Mode numbers
   Modes and bands are numbered from 1 in ``plot(mode=...)`` and viewers;
   arrays such as ``neff`` are ordinary zero-based NumPy arrays.

Field normalization
   Eigenvectors are returned with their solver normalization and a phase
   that makes each mode as real as possible; there is no power normalization.
   Waveguide results store :math:`H` as :math:`-i\eta_0 H` (same units as
   :math:`E`), recorded in ``metadata["field_normalization"]``. Periodic and
   scattering results store SI :math:`E` and :math:`H`. The tracker exports
   power-normalized port modes (:doc:`solvers/tracking`).

Accuracy information
   ``result.info.residuals`` is the normwise backward error of each returned
   eigenpair, :math:`\|Av-\lambda Bv\| / ((\|A\|+|\lambda|\|B\|)\|v\|)`, or
   the relative residual of the linear system for scattering.
   ``result.info.converged`` checks it against ``max(1e-8, 100*tol)``. A small
   residual means the discrete problem was solved; mesh refinement measures
   discretization accuracy.

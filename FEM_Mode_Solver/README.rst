FEM Mode Solver
===============

``FEM_Mode_Solver`` is a standalone finite-element package for fixed-frequency
waveguide modes.  Geometry and boundary objects are placed in physical metres
first; a conforming line or triangular mesh is created only when
``discretize()`` is called.  The original Yee-grid solvers remain separate and
can therefore be used as regression references while the FEM implementation
develops independently.

The package contains:

* ``ModeSolver1D`` for TE/TM modes of an x-stratified guide;
* ``ModeSolver2D`` for full-vector modes of an x-y cross-section;
* immutable, backend-neutral ``Mode``, ``ModeSet``, and ``SampledFields``
  results;
* static and interactive Matplotlib visualization with component, quantity,
  mode, material, and mesh controls.

Installation
------------

From the repository root:

.. code-block:: bash

   python -m pip install -e ./FEM_Mode_Solver

The 2D mesher uses Gmsh.  On Windows, use an environment in which the Gmsh
Python package can locate its native libraries.  The 1D line mesher does not
start Gmsh.

Convention and units
--------------------

The public phasor convention matches the existing FDFD mode solvers:

.. math::

   F(x,y,z,t)=\operatorname{Re}\{\widetilde F(x,y)
   \exp(+j\omega t-j\beta z)\}.

Consequently, passive material loss normally has
``Im(epsilon_r) <= 0`` and ``Im(mu_r) <= 0``.  A passive forward mode has
``Im(beta) <= 0`` and dimensional attenuation
``alpha = -Im(beta)`` in inverse metres.  Geometry is always specified in SI
metres and frequency in hertz.

Geometry-first lifecycle
------------------------

Every solver follows the same state transition:

.. code-block:: text

   construct -> add geometry/boundaries -> discretize -> solve -> inspect
                      ^                        |
                      +-- any geometry edit --+

``add_*`` calls store exact physical objects; they do not rasterize them.
Changing geometry after ``discretize()`` invalidates both the mesh and the last
solution.  Call ``discretize()`` again before the next solve.  Calling
``discretize()`` also clears a previous solution.  Solvers intentionally raise
``NotDiscretizedError`` or ``StaleDiscretizationError`` rather than silently
solving on an obsolete mesh.

One-dimensional example
-----------------------

.. code-block:: python

   from FEM_Mode_Solver import ModeSolver1D

   frequency = 193.414489e12
   solver = ModeSolver1D(
       frequency,
       x_range=(-3.0e-6, 3.0e-6),
       num_modes=4,
       background_epsilon=1.44**2,
   )
   core = solver.add_layer(
       epsilon=3.45**2,
       mu=1.0,
       x_range=(-0.25e-6, 0.25e-6),
       name="core",
   )

   # Interfaces, including both core faces, become exact mesh nodes.
   mesh = solver.discretize(max_element_size=60e-9)
   modes = solver.solve(neff_guess=3.2)

   for number, mode in enumerate(modes, start=1):
       print(number, mode.polarization, mode.neff, mode.alpha)

   solver.visualize(
       mode=1,
       components=("Ey", "Hz"),
       quantity="real",
       mesh=True,
       show=True,
   )

``num_modes`` is the total returned count after the TE and TM spectra are
merged.  ``add_pec()`` or ``add_pmc()`` without an interval changes both outer
walls; with an interval it creates an opaque internal region.  A physical PML
can be placed before discretization, for example:

.. code-block:: python

   solver.add_pml(0.75e-6, n=3, sigma_max=5.0, direction="all")

Here the first argument is a physical thickness, not a cell count.
Separate calls may configure different exterior sides (for example ``x-`` and
``x+`` with different thicknesses); repeated coverage of the same side is
rejected instead of silently multiplying stretch factors.

Two-dimensional example
-----------------------

.. code-block:: python

   from FEM_Mode_Solver import ModeSolver2D

   solver = ModeSolver2D(
       frequency=193.414489e12,
       x_range=(-3.0e-6, 3.0e-6),
       y_range=(-2.0e-6, 2.0e-6),
       num_modes=4,
       background_epsilon=1.44**2,
       boundary="pec",
   )
   solver.add_rectangle(
       epsilon=3.45**2,
       mu=1.0,
       x_range=(-0.25e-6, 0.25e-6),
       y_range=(-0.11e-6, 0.11e-6),
       name="core",
   )

   mesh = solver.discretize(
       max_element_size=120e-9,
       quadrature_order=4,
   )
   modes = solver.solve(
       neff_guess=3.2,
       residual_tolerance=1e-8,
       divergence_tolerance=1e-6,
   )
   modes[0].fields.component("Ex")       # read-only complex samples
   modes[0].fields.vector_magnitude("E")

   solver.visualize(
       mode=1,
       component="E",
       quantity="magnitude",
       mesh=True,
       show=True,
   )

``ModeSolver2D`` uses the same physical ``neff_guess`` constructor name as the
1D solver.  ``guess=`` remains accepted as a keyword-only transition alias, but
the two names cannot be supplied together.

The 2D transverse field uses first-order Nedelec ``H(curl)`` elements and the
longitudinal field uses continuous P1 elements.  The fixed-frequency
propagation problem is assembled as the analytic quadratic pencil

.. math::

   \left(A_0+n_\mathrm{eff}A_1+n_\mathrm{eff}^2A_2\right)u=0.

Material and geometry API
-------------------------

The common convenience methods retain the familiar solver names but now take
physical coordinates:

.. code-block:: python

   # 1D
   add_layer(epsilon, mu, x_range, name=None)
   add_pec(x_range=None, components=None, name=None)
   add_pmc(x_range=None, components=None, name=None)
   add_impedance_surface(Zs, x_range=..., name=None)
   add_pml(pml_width, n=3, sigma_max=5, direction="all")

   # 2D
   add_rectangle(epsilon, mu, x_range, y_range, name=None)
   add_circle(epsilon, mu, center, r1, r2=None, name=None)
   add_polygon(epsilon, mu, points, name=None)
   add_triangle(epsilon, mu, p1, p2, p3, name=None)
   add_pec(x_range=None, y_range=None, components=None, name=None)
   add_pmc(x_range=None, y_range=None, components=None, name=None)
   add_impedance_surface(Zs=None, preset=None, x_range=..., y_range=..., name=None)
   add_mesh_refinement(shape, max_element_size, transition_width=0, name=None)
   add_pml(pml_width, n=3, sigma_max=5, direction="all")

Scalar or three-entry diagonal ``(xx, yy, zz)`` relative ``epsilon`` and
``mu`` values are accepted.  Later material objects take precedence in an
overlap.  Geometry methods return handles that can be passed to
``solver.remove(handle)`` before rediscretizing.

Discretization
--------------

For 1D:

.. code-block:: python

   mesh = solver.discretize(
       max_element_size=None,
       resolution=160,
       wavelength_elements=10,
       material_aware=True,
       element_order=1,
       quadrature_order=4,
   )

For 2D:

.. code-block:: python

   mesh = solver.discretize(
       max_element_size=None,
       wavelength_elements=10,
       material_aware=True,
       interface_refinement=None,
       boundary_refinement=0.5,
       quadrature_order=4,
   )

Material-aware sizing reduces the local element size where the conservative
``sqrt(max(abs(epsilon)) * max(abs(mu)))`` estimate is larger and enforces
``wavelength_elements`` across the local material wavelength.  In 2D,
``boundary_refinement=0.5`` is the default and halves the nearby size target.
Optional interface refinement and explicit geometry-only sizing regions can be
combined, for example:

.. code-block:: python

   from FEM_Mode_Solver import Rectangle

   solver.add_mesh_refinement(
       Rectangle((-2e-3, 2e-3), (1e-3, 2e-3)),
       max_element_size=0.1e-3,
       transition_width=0.5e-3,
       name="strip_edges",
   )
   mesh = solver.discretize(material_aware=True)
   finer_mesh = solver.refine(factor=2.0)

``refine()`` remeshes the continuous scene at the requested density increase
and invalidates any previous modes.  Provide physically converged settings for
production calculations; conformity and automatic sizing do not replace mesh
and PML convergence studies.

``discretize()`` returns a ``FEMMesh1D`` or ``FEMMesh2D`` wrapper.  The same
wrapper is available as ``solver.mesh`` and ``solver.mesh_data`` after
discretization; use ``solver.native_mesh`` for the underlying scikit-fem mesh.

Surface impedance and copper microstrip
---------------------------------------

The 2D solver can exclude a finite conductor body from the volume mesh and
apply an isotropic surface-impedance boundary condition to its exposed facets:

.. code-block:: python

   ground = solver.add_impedance_surface(
       preset="Cu",
       x_range=(-6e-3, 6e-3),
       y_range=(-35e-6, 0.0),
       name="copper_ground",
   )

Metal presets use the good-conductor approximation at the solver frequency;
an explicit complex ``Zs`` in ohms may be supplied instead.  The sign follows
the package's ``exp(+j*omega*t)`` convention.  The conductor rectangle is an
opaque region, not a lossy dielectric volume.

``examples/microstrip_sibc.py`` constructs a compact 10 GHz microstrip with a
low-loss substrate, 35 micrometre copper strip and ground, material-aware and
boundary-aware sizing, and a local strip-edge refinement.  Run it from an
installed/editable package with:

.. code-block:: bash

   python -m FEM_Mode_Solver.examples.microstrip_sibc

The example is intended as an API and convergence-study starting point; its
mesh parameters should be tightened for reported engineering results.

FEM transmission-line calculator
--------------------------------

The package also includes a two-potential quasi-TEM FEM calculator for
coaxial, microstrip, stripline, and coplanar-waveguide cross-sections.  The CPW
template exposes one mode only: the centre signal conductor against the tied
left and right grounds, labelled ``CPW odd (signal to tied grounds)`` in the
GUI so that the terminal definition is unambiguous.

The lifecycle follows the mode solvers:

.. code-block:: python

   from FEM_Mode_Solver import TransmissionLineCalculator

   calculator = TransmissionLineCalculator.microstrip(
       frequency=10.0e9,
       trace_width=3.0e-3,
       substrate_height=1.524e-3,
       conductor_thickness=35.0e-6,
       epsilon_r=3.55,
       loss_tangent=0.0027,
       domain_padding_factor=1.0,
   )
   calculator.discretize(
       max_element_size=0.30e-3,
       material_aware=True,
       boundary_refinement=0.4,
   )
   result = calculator.solve()

   print(result.neff)
   print(result.characteristic_impedance)
   print(result.wave_impedance)
   print(result.capacitance_per_length)
   print(result.inductance_per_length)

   # Per-element |Et| and |Ht| colour maps with uniform direction arrows.
   calculator.visualize_with_gui()

The field colour in each native triangle carries magnitude; the white arrows
are spatially distributed, phase-resolved unit vectors and therefore encode
direction only.  A power-law colour normalization keeps weaker fringe fields
visible without clipping the conductor-edge maximum.

``refine(factor=2)`` remeshes the complete line, including material jumps,
signal edges, and PEC walls, and invalidates the old result before the next
``solve()``.

Microstrip, stripline, and CPW use a remote zero-potential wall to truncate
their otherwise open or laterally unbounded cross-sections.  Set
``domain_padding_factor`` above one to move that wall outward, then compare
successive solutions to quantify domain-truncation error independently of
mesh refinement.

For a unit signal voltage, the dielectric FEM potential gives

.. math::

   \nabla_t\!\cdot(\epsilon\nabla_t\phi)=0,
   \qquad \mathbf E_t=-\nabla_t\phi,
   \qquad C'=\int_A \epsilon |\mathbf E_t|^2\,dA.

A second solve with every dielectric replaced by vacuum gives ``C0'`` and the
unit-current magnetic dual.  For the supported nonmagnetic quasi-TEM lines,

.. math::

   L'=\frac{1}{c_0^2 C'_0},\qquad
   n_\mathrm{eff}=\sqrt{\frac{C'}{C'_0}},\qquad
   Z_c=\sqrt{\frac{L'}{C'}}.

``Zc`` is the circuit characteristic impedance.  It is intentionally kept
separate from wave impedance.  The reported scalar wave impedance is the
area least-squares transverse field ratio

.. math::

   Z_w =
   \frac{\int_A (E_x H_y^* - E_y H_x^*)\,dA}
        {\int_A (|H_x|^2+|H_y|^2)\,dA},

and ``result.local_wave_impedance`` contains the corresponding pointwise
``Et/Ht`` ratio, with zero-field samples masked.

Launch the complete selector/parameter/results GUI with either:

.. code-block:: bash

   fem-transmission-line

or:

.. code-block:: bash

   python -m FEM_Mode_Solver.examples.transmission_line_calculator

The calculation is a frequency-tagged quasi-static FEM extraction, appropriate
for TEM and quasi-TEM operation.  Open lines are approximated by the finite PEC
truncation wall described above.  The calculator does not model radiation,
conductor skin loss, higher-order dispersion, or magnetic materials; use the
full-vector 2D mode solver when those effects are required.

Results
-------

``solve()`` returns a ``ModeSet`` and also stores it in ``solver.solution``.
``ModeSet`` is an immutable Python sequence:

.. code-block:: python

   first = modes[0]        # normal zero-based sequence access
   first = modes.mode(1)   # explicit user-facing one-based access
   print(modes.neff)       # read-only vector
   print(modes.beta)
   te_modes = modes.by_polarization("TE")

Each ``Mode`` provides:

* ``neff`` and dimensional ``beta``;
* ``alpha = -Im(beta)``;
* optional ``polarization``, power, normalization, eigensolver residual, and
  divergence residual;
* one immutable ``SampledFields`` object.

``SampledFields`` exposes ``coordinates``, ``x``, optional ``y``, component
names, optional material samples, and optional native mesh points/cells.  Use
``component('Ex')``, ``quantity('Ex', 'phase')``, or
``vector_magnitude('H')``.  Arrays are defensive copies with writes disabled.
Native FEM coefficient vectors and solver diagnostics remain in result
metadata when provided by the backend.

For transition code, solvers also publish familiar post-solve views such as
``solver.neff``, ``solver.Ex``, and polarization-specific 1D arrays.  These are
compatibility conveniences; new code should retain the returned ``ModeSet``.

Visualization
-------------

Both solver methods and module-level functions use the same interface:

.. code-block:: python

   from FEM_Mode_Solver.visualization import visualize, visualize_with_gui

   figure, axes = visualize(
       modes,
       mode=1,                       # one-based
       components=("Ex", "Ey", "Ez"),
       quantity="real",             # real, imag, magnitude, phase
       material=True,
       mesh_overlay=True,
       normalize=False,
       show=False,
   )
   figure.savefig("mode.png", dpi=180)

   viewer = visualize_with_gui(
       modes,
       mode=1,
       component="Ex",
       quantity="real",
       mesh=True,
       show=True,
   )

The interactive viewer is implemented with Matplotlib widgets and offers mode,
component, quantity, mesh, material, and normalization controls.  Keep the
returned ``viewer`` alive while a non-blocking window is open.  Legacy flags
such as ``ex=True``, ``hz=True``, ``eabs=True``, and ``habs=True`` are accepted
by ``visualize``.

Solver integration protocol
---------------------------

A solver or external backend can use the shared viewer without importing a
concrete solver class.  It only needs to:

1. return a ``ModeSet`` from ``solve()``;
2. retain that object as ``solver.solution`` (``None`` before a successful
   solve);
3. attach one ``SampledFields`` object to each ``Mode``;
4. optionally implement thin wrappers:

   .. code-block:: python

      def visualize(self, mode=1, **kwargs):
          from .visualization import visualize
          return visualize(self.solution, mode=mode, **kwargs)

      def visualize_with_gui(self, **kwargs):
          from .visualization import visualize_with_gui
          return visualize_with_gui(self.solution, **kwargs)

For 1D, coordinates may have shape ``(N, 1)`` and line cells ``(M, 2)``.  For
2D, coordinates may be structured axes, curvilinear arrays, or ``(N, 2)``
points; triangular mesh connectivity has shape ``(M, 3)``.  Component mappings
are preferred.  A packed array is also accepted when its final axis follows
``metadata['component_order']``.

Current limits
--------------

* First-order line and Nedelec/P1 triangular elements are the validated
  element families.
* Component-selective PEC/PMC volume masks from the Yee solver do not have a
  direct conforming FEM equivalent and are rejected rather than approximated.
* The 2D backend supports isotropic surface-impedance facets.  The scalar 1D
  backend still raises ``BackendCapabilityError`` for impedance regions.
* PML and lossy calculations are non-Hermitian; always inspect eigen-residual,
  divergence-residual, mesh-refinement, and PML-refinement behavior.
* Mode ordering is local to one solve.  Frequency sweeps must track branches
  by field overlap rather than assuming a fixed array index across crossings.

Migration from the Yee solvers
------------------------------

The familiar ``add_rectangle``, ``add_circle``, ``add_triangle``, ``add_pec``,
``add_pmc``, ``add_pml``, ``solve``, and visualization names are retained.
The important intentional changes are:

* ``Nx``/``Ny`` and integer index ranges are replaced by physical geometry and
  a later ``discretize()`` call;
* ``pml_width`` is in metres;
* direct mutation of ``cell_eps_r_*``/``cell_mu_r_*`` arrays is replaced by
  material-region objects;
* ``guess=-neff**2``/``sigma`` is replaced by the physical ``neff_guess``;
* ``solve()`` returns an immutable result instead of requiring users to read
  mutable solver state.

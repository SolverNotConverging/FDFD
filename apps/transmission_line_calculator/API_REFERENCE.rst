Transmission Line Calculator API reference
==========================================

See `README.rst <README.rst>`_ for introduction, build instructions, and tutorials.
This is the public C++20 API in ``native/model.hpp`` and ``native/solver.hpp``;
Qt/FTXUI widget implementation classes are internal application details.
Include ``solver.hpp`` and link against the CMake target ``tl::core``.

Public lengths are metres and frequency is Hz. The convention is
``exp(+j omega t - j beta z)``; a passive forward mode has nonpositive
``Im(beta)``. Relative permittivity uses a negative imaginary loss term.
Field vectors are transverse x/y components. All result vectors refer to the
final solved mesh, including when adaptation changes the initial mesh.

Required means the caller must supply an argument. Optional aggregate members
have the defaults shown below. All tutorial/example calls set
``parameters.maxRefinements = 0``; library and GUI defaults remain 2.

.. contents:: API index
   :local:
   :depth: 1

``tl::defaultParameters``
-------------------------

Return geometry-specific audited defaults. Use this when changing line type;
changing only ``Parameters::type`` retains the previous geometry's values.

.. code-block:: cpp

   [[nodiscard]] Parameters defaultParameters(LineType type);

.. list-table:: Input arguments
   :header-rows: 1

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``type``
     - Required
     - ``tl::LineType``
     - Select Coaxial, Microstrip, Stripline, or CoplanarWaveguide.

Returns a ``Parameters`` value. It does not mesh or solve anything.

``tl::solve``
-------------

Generate conforming P1 triangles, solve dielectric and vacuum electrostatic
problems, reconstruct E/H, and extract the forward quasi-TEM RLGC mode.

.. code-block:: cpp

   [[nodiscard]] Result solve(const Parameters& parameters);

.. list-table:: Input arguments
   :header-rows: 1

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``parameters``
     - Required
     - ``const tl::Parameters&``
     - Geometry, material, mesh, and adaptive controls described below. The input is not mutated.

Returns ``Result``. Invalid input throws ``std::invalid_argument``; meshing or
numerical failure throws ``std::runtime_error``. There is one initial solve
and at most ``maxRefinements`` mesh updates. Refinement raises density by 1.5
and stops when the larger normalized dielectric/vacuum normal-flux jump
residual is at or below ``adaptiveTolerance``. A spent budget returns the last
result with ``adaptiveConverged=false``. This estimator is distinct from the
algebraic residuals and is not a certified error bound. No file is written.

``tl::LineType``
----------------

Enum selecting the physical cross-section.

.. list-table:: Input arguments
   :header-rows: 1

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This is an enum, not a callable. Select one of the values below.

* ``Coaxial``: concentric signal conductor and outer cylindrical conductor.
* ``Microstrip``: trace above a dielectric substrate and bottom ground.
* ``Stripline``: centred trace between two ground planes.
* ``CoplanarWaveguide``: centre trace separated by gaps from lateral grounds.

``tl::Parameters``
------------------

Aggregate input record. ``Parameters{}`` uses microstrip defaults;
``defaultParameters(type)`` selects defaults for other templates. All numeric
fields are validated, including fields unused by the selected geometry. Keep
unused dimensions at their valid template defaults.

.. list-table:: Input arguments / aggregate members
   :header-rows: 1
   :widths: 20 15 20 45

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``type``
     - Optional
     - ``LineType``
     - Default Microstrip; selects the geometry template.
   * - ``frequencyHz``
     - Optional
     - ``double``
     - Positive ordinary frequency; default 10e9 Hz.
   * - ``maxElementSize``
     - Optional
     - ``double``
     - Positive initial maximum edge target; default 1e-3 m. Local conductor/material grading may produce smaller triangles.
   * - ``refinementFactor``
     - Optional
     - ``double``
     - Positive initial density multiplier; default 1. The base target is maxElementSize/refinementFactor.
   * - ``maxRefinements``
     - Optional
     - ``int``
     - Nonnegative adaptive update budget; default 2. Set 0 for one fixed-mesh solve, as all examples do.
   * - ``adaptiveTolerance``
     - Optional
     - ``double``
     - Positive finite discretization-residual threshold; default 0.05. Independent of LU residual validation.
   * - ``innerRadius``
     - Optional
     - ``double``
     - Coax signal radius; default 0.50e-3 m. Must be positive and smaller than outerRadius.
   * - ``outerRadius``
     - Optional
     - ``double``
     - Inner radius of the coax outer conductor; default 1.67e-3 m.
   * - ``outerConductorThickness``
     - Optional
     - ``double``
     - Positive coax outer-metal thickness; default 0.15e-3 m.
   * - ``traceWidth``
     - Optional
     - ``double``
     - Positive microstrip/stripline signal width; default 3.00e-3 m in Parameters{}.
   * - ``substrateHeight``
     - Optional
     - ``double``
     - Positive microstrip/CPW substrate thickness; default 1.524e-3 m.
   * - ``conductorThickness``
     - Optional
     - ``double``
     - Positive planar-metal thickness; default 35e-6 m. Stripline thickness must be less than groundSpacing.
   * - ``groundSpacing``
     - Optional
     - ``double``
     - Positive stripline ground-to-ground separation; default 1.524e-3 m in Parameters{}.
   * - ``centerWidth``
     - Optional
     - ``double``
     - Positive CPW centre-conductor width; default 0.60e-3 m.
   * - ``gap``
     - Optional
     - ``double``
     - Positive CPW signal-to-ground gap on each side; default 0.25e-3 m.
   * - ``groundWidth``
     - Optional
     - ``double``
     - Positive width of each finite CPW side ground; default 1.50e-3 m.
   * - ``epsilonR``
     - Optional
     - ``double``
     - Positive real dielectric relative permittivity; default 3.55.
   * - ``lossTangent``
     - Optional
     - ``double``
     - Nonnegative dielectric loss tangent; default 2.7e-3. Zero gives lossless dielectric.
   * - ``domainPaddingFactor``
     - Optional
     - ``double``
     - Positive remote-wall padding multiplier for non-coax geometries; default 3. Truncation error requires a separate padding study.
   * - ``metalConductivity``
     - Optional
     - ``std::optional<double>``
     - Default nullopt for ideal conductors. A finite positive value in S/m enables a first-order surface-impedance conductor-loss correction.

``tl::Vec2`` and ``tl::FieldVector``
--------------------------------------

Two-component aggregate records with zero-initialized components.

.. list-table:: Input arguments / aggregate members
   :header-rows: 1

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Optional
     - ``double`` / ``std::complex<double>``
     - Vec2 physical x coordinate in metres, or FieldVector complex x field component; default zero.
   * - ``y``
     - Optional
     - ``double`` / ``std::complex<double>``
     - Vec2 physical y coordinate in metres, or FieldVector complex y field component; default zero.

``tl::Triangle``
----------------

One affine P1 element in the native mesh.

.. list-table:: Input arguments / aggregate members
   :header-rows: 1

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``nodes``
     - Optional at construction
     - ``std::array<int, 3>``
     - Three zero-based indices into Mesh::nodes; default zeros are a placeholder, not a valid triangle.
   * - ``relativePermittivity``
     - Optional
     - ``std::complex<double>``
     - Cell dielectric value; default 1+0j.

``tl::Mesh``
------------

Conforming physical-coordinate mesh, normally returned by solve.

.. list-table:: Input arguments / aggregate members
   :header-rows: 1

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``nodes``
     - Optional
     - ``std::vector<Vec2>``
     - Physical coordinates in metres; default empty.
   * - ``triangles``
     - Optional
     - ``std::vector<Triangle>``
     - Connectivity and per-cell dielectric values; default empty.

``tl::FieldSample``
-------------------

One element's reconstructed fields and integration data.

.. list-table:: Input arguments / aggregate members
   :header-rows: 1

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``position``
     - Optional
     - ``Vec2``
     - Triangle sample centre in metres; default zero.
   * - ``electric``
     - Optional
     - ``FieldVector``
     - Complex E in V/m at the prescribed voltage; default zero.
   * - ``magnetic``
     - Optional
     - ``FieldVector``
     - Reconstructed complex H in A/m; default zero.
   * - ``area``
     - Optional
     - ``double``
     - Triangle integration area in m^2; default zero.
   * - ``relativePermittivity``
     - Optional
     - ``std::complex<double>``
     - Material value at the sample; default 1+0j.

``tl::Result``
--------------

Aggregate output returned by solve. Its members can be supplied in aggregate
construction, but normally should be read from a solved result. Arrays default
empty, scalar/complex values default zero unless noted, and records use their
own defaults. A manually default-constructed Result is not a solved state.

.. list-table:: Input arguments / output aggregate members
   :header-rows: 1
   :widths: 24 14 25 37

   * - Argument/member
     - Required / optional
     - Expected type
     - Explanation
   * - ``parameters``
     - Optional; solver output
     - ``Parameters``
     - Original requested configuration, rather than the internally increased adaptive density.
   * - ``mesh``
     - Optional; solver output
     - ``Mesh``
     - Final solved physical mesh.
   * - ``samples``
     - Optional; solver output
     - ``std::vector<FieldSample>``
     - One reconstructed sample per final triangle, in triangle order.
   * - ``electricPotential``, ``vacuumPotential``
     - Optional; solver output
     - ``std::vector<std::complex<double>>``
     - Dielectric and vacuum nodal potentials in volts, in mesh-node order.
   * - ``neff``
     - Optional; solver output
     - ``std::complex<double>``
     - Dimensionless forward effective index beta/k0.
   * - ``characteristicImpedance``
     - Optional; solver output
     - ``std::complex<double>``
     - Circuit characteristic impedance in ohms.
   * - ``waveImpedance``
     - Optional; solver output
     - ``std::complex<double>``
     - Integrated field wave impedance in ohms; distinct from circuit impedance.
   * - ``capacitancePerLength``
     - Optional; solver output
     - ``std::complex<double>``
     - Complex dielectric capacitance in F/m, including dielectric loss.
   * - ``beta``
     - Optional; solver output
     - ``std::complex<double>``
     - Forward propagation constant in rad/m.
   * - ``voltage``
     - Optional; solver output
     - ``std::complex<double>``
     - Excitation voltage; default 1+0j V.
   * - ``current``
     - Optional; solver output
     - ``std::complex<double>``
     - Modal line current in amperes.
   * - ``power``
     - Optional; solver output
     - ``std::complex<double>``
     - Integrated complex longitudinal power in watts.
   * - ``vacuumCapacitancePerLength``
     - Optional; solver output
     - ``double``
     - Vacuum dual capacitance in F/m.
   * - ``inductancePerLength``, ``externalInductancePerLength``
     - Optional; solver output
     - ``double``
     - Total and external inductance in H/m; finite-metal surface reactance contributes to the total.
   * - ``resistancePerLength``
     - Optional; solver output
     - ``double``
     - Conductor series resistance in ohm/m.
   * - ``conductancePerLength``
     - Optional; solver output
     - ``double``
     - Dielectric shunt conductance in S/m.
   * - ``surfaceResistance``
     - Optional; solver output
     - ``double``
     - Good-conductor surface resistance in ohms; zero for ideal metal.
   * - ``conductorGeometryFactorPerLength``
     - Optional; solver output
     - ``double``
     - Surface-field geometry factor in 1/m used for conductor loss.
   * - ``materialResidual``, ``vacuumResidual``
     - Optional; solver output
     - ``double``
     - Relative algebraic residuals for the two electrostatic solves.
   * - ``meshMilliseconds``, ``solveMilliseconds``
     - Optional; solver output
     - ``double``
     - Total meshing and solving times across the adaptive run, in ms.
   * - ``assemblyMilliseconds``, ``factorizationMilliseconds``
     - Optional; solver output
     - ``double``
     - Accumulated assembly and numeric-factorization times, in ms.
   * - ``adaptiveHistory``
     - Optional; solver output
     - ``std::vector<std::array<double, 2>>``
     - Per-pass pairs {element count, normalized flux-jump residual}, including the initial solve.
   * - ``adaptiveConverged``
     - Optional; solver output
     - ``bool``
     - True only when the estimator meets adaptiveTolerance; default false.

CMake integration
-----------------

The public target ``tl::core`` exports the native include directory and its
Eigen/Gmsh link dependencies. The calculator project still requires Qt and
FTXUI at configuration time because it also builds both front ends.

.. list-table:: Configuration inputs
   :header-rows: 1

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``TL_BUILD_EXAMPLES``
     - Optional
     - CMake boolean
     - Default OFF. ON builds the fixed-mesh ``tl-line-comparison`` example.
   * - ``BUILD_TESTING``
     - Optional
     - CMake boolean
     - CTest switch controlling the regression/smoke test targets.

``examples/line_comparison.cpp`` shows all four templates with
``maxRefinements = 0``, prints neff and characteristic impedance, and opens
electric/magnetic field plots in a Qt tab for each geometry. Pass
``--smoke-test`` to render every tab and exit automatically for GUI validation.

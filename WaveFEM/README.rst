WaveFEM
#######

WaveFEM is a compact full-vector finite-element solver for 2.5D frequency-domain scattering from finite perturbations of straight waveguides. It lives in its own directory because the older finite-difference code in the parent repository uses a different time-harmonic convention.

The `complete API reference <API_REFERENCE.rst>`_ documents every top-level class, method, property, parameter, return value, unit, exception category, and the lower-level research interfaces for geometry, meshing, FEM assembly, equivalent sources, monitor sampling, and modal projection.

The current MVP provides:

* a mixed first-order Nedelec + continuous P1 Maxwell discretization;

* fixed-frequency full-vector lead modes with angle-based oblique incidence;

* permittivity contrast, released-background-PEC slots, and finite actual-only PEC plates;

* transformation-optics PMLs in ``x`` and ``z``;

* electromagnetic power-Gram modal projection;

* S-parameters, sampled E/H fields, and structured power diagnostics;

* single-run and frequency-sweep HDF5 result persistence;

* a separate ``WaveFEMViewer`` desktop application for persisted HDF5 results;

* Gmsh geometry and conforming triangular meshes.

The public API uses SI units. Ordinary ``frequency`` in hertz is the preferred spectral input throughout the high-level API. ``wavelength`` and angular ``omega`` remain accepted compatibility inputs, but new code and all examples use ``frequency``.

What 2.5D means
---------------

The mesh covers the physical ``x-z`` plane. The ``y`` direction is invariant apart from a prescribed Fourier/Bloch factor:

.. math::

   \mathbf E(x,y,z,t)=\operatorname{Re}\!\left\{
   \mathbf E(x,z)e^{i k_y y+i\beta z-i\omega t}\right\}.

Thus ``d/dy -> i*ky``. With the project convention ``exp(-i*omega*t)``,

.. math::

   \nabla_{k_y}\times\mathbf E=
   \begin{bmatrix}
   i k_yE_z-\partial_zE_y\\
   \partial_zE_x-\partial_xE_z\\
   \partial_xE_y-i k_yE_x
   \end{bmatrix},
   \qquad
   \nabla\times\mathbf E=i\omega\mu\mathbf H.

WaveFEM centralizes this operator in ``wavefem.operators``; the scattering and mode solvers use the same signs and component ordering. At ``ky=0`` it reduces to the ordinary full-vector two-dimensional curl structure.

The high-level scattering API specifies ``angle`` in degrees from +z toward +y. After ``solve_modes`` identifies the lead families, ``set_incident_mode`` ties the requested angle to the selected family and resolves

.. math::

   k_y = k_0 n_{\mathrm{eff,total}}\sin(\mathrm{angle}).

Thus ``angle=0`` gives ``ky=0``. The lower-level FEM and mode objects continue to expose the resolved ``ky`` used by the 2.5D operators.

Install in ``RF_Engineering_env``
---------------------------------

From the ``WaveFEM`` directory:

.. code-block:: powershell

   conda env update --name RF_Engineering_env --file environment.yml
   conda run --name RF_Engineering_env python -m pip install --editable . --no-deps
   conda run --name RF_Engineering_env python -m pytest

If ``conda`` is not on ``PATH`` in this workspace, its executable is ``C:\Users\Traveler\anaconda3\Scripts\conda.exe``. Using ``conda run`` on Windows also ensures the Gmsh DLL directory is active.

Minimal end-to-end solve
------------------------

This deliberately small closed-transverse example is quick enough for a smoke test. The propagation ends are still open through z-PMLs.

.. code-block:: python

   import wavefem as wf

   sim = wf.Scattering2D(
       frequency=193.414489e12,
       angle=0.0,
       x_span=(0.0, 1.0e-6),
       z_span=(-3.0e-6, 3.0e-6),
       background_eps=1.0,
       transverse_boundary="pec",
   )

   sim.add_rectangle(
       x=(0.0, 1.0e-6),
       z=(-0.30e-6, 0.30e-6),
       eps=1.002,
       name="weak_insert",
   )
   sim.add_pml(z=0.8e-6)
   mesh = sim.mesh(wavelength_elements=8)
   print("maximum requested edge =", mesh.info.requested_maximum_edge)

   modes = sim.solve_modes(num_modes=1, neff_guess=1.0)
   sim.set_incident_mode(modes[0])
   result = sim.run(h5_path="weak_index_perturbation.h5")

   print("beta =", modes[0].beta)
   print("S11 =", result.S11)
   print("S21 =", result.S21)
   print("power error =", result.power_balance_error)
   print("saved to =", result.h5_path)
   print(result.check())

   result.visualize("Ey", quantity="abs")

``run()`` is the recommended terminal step because it solves and persists the complete result. Use ``solve(h5_path="result.h5")`` when a path should be optional or computed dynamically; bare ``solve()`` intentionally leaves the result in memory. An existing result can be written later with ``result.save_h5("result.h5")``.

``visualize()`` displays the Matplotlib window before returning. Pass ``show=False`` only when embedding the returned axes in an existing application. See ``examples/weak_index_perturbation.py``, ``examples/slab_mode.py``, and ``examples/oblique_ky.py`` for runnable variants. The frequency-sweep workflow is shown in ``examples/frequency_sweep.py``; a grounded-slab leaky-wave antenna with a finite radiating slot is in ``examples/grounded_slab_slot.py``.

Grounded-slab PEC slot and finite top plates
--------------------------------------------

A zero-thickness ground plane inside the solve domain is a z-invariant background PEC sheet. Cut a finite opening only from the actual device:

.. code-block:: python

   sim = wf.Scattering2D(
       frequency=20.0e9,
       angle=0.0,
       x_span=(-12e-3, 20e-3),
       z_span=(-30e-3, 30e-3),
   )
   sim.add_rectangle(
       x=(0.0, 4e-3), z="all", eps=4.0,
       background=True, name="dielectric_slab",
   )
   ground = sim.add_pec(
       x=0.0, z="all", background=True, name="ground_plane",
   )
   sim.add_slot(pec=ground, z=(-1e-3, 1e-3), name="ground_slot")
   sim.add_pec(
       x=4e-3, z=(-4e-3, -1.5e-3), background=False,
       name="left_top_plate",
   )
   sim.add_pec(
       x=4e-3, z=(1.5e-3, 4e-3), background=False,
       name="right_top_plate",
   )
   sim.add_pml(x=4e-3, z=6e-3)
   sim.set_monitors(left=-12e-3, right=12e-3)
   sim.mesh(max_element_size=1e-3)
   modes = sim.solve_modes(num_modes=1, neff_guess=1.8, num_elements=96)
   sim.set_incident_mode(modes[0])
   result = sim.run(h5_path="grounded_slab_slot.h5")

The mode port contains the complete PEC sheet. In the 2D actual solve the slot facets are released, and their two one-sided incident magnetic reactions drive the scattered field. This is a boundary source: a valid slot solve has ``result.solve_info["source_active_fraction"] == 0`` when no material changes, but ``released_pec_facet_count > 0`` and a nonzero scattered field. In the HDF5 scene, the yellow ground line is split around the aperture.

Finite actual-only PEC plates are compact boundary insertions and are absent from the lead-mode background. Their Nedelec/P1 tangential scattered-field trace is imposed as ``-E_inc``, so ``E_total,t=0`` on each plate. The mesh exposes these facets separately as ``mesh.inserted_pec_facets``, and integrated results record ``inserted_pec_facet_count`` and ``prescribed_pec_dof_count``. Background grounds must remain z-invariant; actual-only plates must lie strictly inside the non-PML interior.

Run the example from the ``WaveFEM`` directory:

.. code-block:: powershell

   conda run --name RF_Engineering_env python examples/grounded_slab_slot.py

The example has no command-line parser. It writes ``grounded_slab_slot.h5``, opens that result in ``wavefem-viewer``, and displays the field in Matplotlib. Run ``examples/frequency_sweep.py`` for a multi-frequency archive and response plot.

Domain layout
-------------

Monitor lines are mesh-conforming and must lie in straight, non-PML lead sections:

.. code-block:: text

    outer wall                                                        outer wall
        |                                                                 |
        | z-PML | input lead | monitor | perturbation | monitor | output | z-PML |
        |                                                                 |
                            <---- non-PML control volume ---->

An open transverse structure additionally uses left/right x-PMLs. The background straight-guide profile continues into the z-PML, and the physical actual/background material contrast remains zero there; the PML is not an equivalent scattering source.

Material- and PEC-aware meshing
-------------------------------

``sim.mesh()`` uses local physical-wavelength sizing by default. High-index dielectrics receive smaller triangles than the exterior, while a smooth distance field further refines the actual PEC sheets and slot rims:

.. code-block:: python

   sim.mesh(
       wavelength_elements=10,
       dielectric_refinement_factor=0.5,
       pec_refinement_factor=0.5,
       pec_refinement_distance=1.5e-3,
   )

The dielectric factor scales non-exterior regions before applying their local wavelength ratio. The PEC factor is relative to the smallest material target; omit the distance to use an automatic transition width of three local target lengths. Material boundaries, PEC lines, slot endpoints, monitors, and PML interfaces remain mesh-conforming even when ``refine_interfaces=False`` disables the local size fields. See ``API_REFERENCE.rst`` for the complete sizing rules and low-level ``generate_mesh()`` controls.

Numerical formulation
---------------------

WaveFEM solves

.. math::

   \nabla_{k_y}\times\mu_r^{-1}(\nabla_{k_y}\times\mathbf E)
   -k_0^2\epsilon_r\mathbf E=0.

The transverse pair ``(Ex, Ez)`` uses an H(curl)-conforming Nedelec space and ``Ey`` uses continuous P1. Test functions are complex-conjugated explicitly. The SI mesh is internally rescaled by ``1/k0`` for conditioning; material and field callbacks continue to receive physical metres.

For an incident background mode, ``E = E_inc + E_sc``, and the MVP source is

.. math::

   L_{\mathrm{actual}}\mathbf E_{\mathrm{sc}}
   =k_0^2(\epsilon_{r,\mathrm{actual}}-\epsilon_{r,\mathrm{background}})
   \mathbf E_{\mathrm{inc}}.

``result.E_scattered`` and ``result.E_total`` are both retained. Permeability perturbations are rejected explicitly until the corresponding curl-contrast source is implemented.

For a finite slot cut from a background PEC sheet, the material contrast is zero and the volume term above vanishes. WaveFEM instead releases the actual slot facets and assembles the doubled planar-screen reaction from the two one-sided incident magnetic traces. The remaining sheet facets retain homogeneous scattered-field PEC conditions because the background mode already has zero tangential electric field there.

For a finite PEC plate inserted only in the actual device, WaveFEM constrains the scattered trace nonhomogeneously,

.. math::

   \mathbf E_{t,\mathrm{sc}}=-\mathbf E_{t,\mathrm{inc}},

which enforces zero total tangential electric field. Released and inserted facet sets are disjoint and may be present in the same solve.

The one-dimensional lead solve uses the full-vector quadratic eigenproblem

.. math::

   (A_0+(\beta/k_0)A_1+(\beta/k_0)^2A_2)u=0,

including nonzero ``ky`` and diagonal tensors induced by a transverse PML. Propagating modes are normalized to one watt per metre in the invariant direction using the time-averaged z-directed Poynting flux.

At each monitor, forward and backward modal amplitudes are recovered from an electromagnetic power/reciprocity Gram system using both E and H. Its condition number and projection residual are included in result diagnostics.

S-parameters and reference planes
---------------------------------

``set_incident_mode(..., reference_plane=z0)`` fixes the modal phase through ``exp(i*beta*(z-z0))``. Projection traces use that same plane, so a uniform guide returns de-embedded ``S21`` close to ``1+0j`` rather than a monitor-spacing phase.

Move the left and right planes without rerunning the FEM solve:

.. code-block:: python

   shifted = result.deembed(left=-1.0e-6, right=1.0e-6)
   print(shifted.S11, shifted.S21)

``port_betas`` always stores the positive-z root for a modal family; reflected direction is represented by the de-embedding formula rather than a negative stored beta.

For multimode results:

.. code-block:: python

   result.S("left", out_mode=1, in_mode=0)
   result.S("right", out_mode=1, in_mode=0)

HDF5 results, frequency sweeps, and viewer
------------------------------------------

Every persisted file uses a versioned WaveFEM schema. A single-run file contains the frequency and resolved ``ky`` (with the requested angle in solve diagnostics), sampled incident/scattered/total E and H, all indexed modal S-parameters, power terms, lead-mode E/H samples, mesh metadata, solve diagnostics, and a full-domain material/overlay scene. The scene stores the conforming dielectric mesh, outer PEC boundary, wave-port monitor lines, and PML interfaces. A sweep file contains the same complete record for every frequency point; it is not limited to summary curves.

Run a strictly increasing ordinary-frequency sweep in hertz with:

.. code-block:: python

   import numpy as np
   import wavefem as wf

   frequencies_hz = np.linspace(190.0e12, 196.0e12, 13)
   sweep = sim.sweep_frequencies(
       frequencies_hz,
       h5_path="frequency_sweep.h5",
       mesh_options={"wavelength_elements": 10},
       mode_options={"num_modes": 2, "neff_guess": 2.4},
       incident_mode=0,
   )

   print(sweep.S11)                 # complex array, one value per frequency
   print(sweep.S21)
   print(sweep.reflection)          # total reflected-power ratios
   print(sweep.transmission)
   print(sweep.radiated_power)      # W/m at each frequency
   print(sweep.absorbed_power)
   print(sweep.power_balance)
   print(sweep.power_balance_error)

Each sweep point receives its own mesh, mode solve, launch, and scattering solve, and the original ``sim`` object is not mutated. The default destination is ``wavefem_sweep.h5``; set ``h5_path=None`` only for an explicitly in-memory sweep. Callback-defined material devices must supply a ``mode_factory(frequency_hz)`` that returns compatible modes for each point.

Load either kind of file without rebuilding the simulation:

.. code-block:: python

   saved = wf.load_h5("frequency_sweep.h5")
   print(saved.kind)                 # "single" or "sweep"
   print(saved.frequencies_hz)
   point = saved.results[0]
   print(point.s_parameters)
   print(point.E_total.shape, point.H_total.shape)
   print(point.modes[0].E.shape, point.modes[0].H.shape)
   print(point.scene.x_span, point.scene.z_span)  # None only for legacy files

The viewer is a separate native C++20/Qt 6 sibling project with its own source code, HDF5 dependency, executable, deployment scripts, and documentation. Build or install it from ``../WaveFEMViewer``. Python discovers both standalone and repository-root CMake ``build*`` directories first, then ``PATH`` and the default installation, so a solved result can launch it directly:

.. code-block:: python

   result.visualize("Ey")           # Matplotlib field figure
   result.visualize_with_gui()      # every stored mode in the native viewer
   sweep.visualize()                # Matplotlib S11/S21 figure
   sweep.visualize_with_gui()       # every frequency and stored mode in the native viewer
   wf.launch_viewer("results")      # opens a directory and its in-window H5 selector

Set ``WAVEFEM_VIEWER_EXECUTABLE`` to override discovery. The equivalent direct Windows launch is:

.. code-block:: powershell

   & "$env:LOCALAPPDATA\WaveFEMViewer\bin\wavefem-viewer.exe" frequency_sweep.h5

For a MinGW build-tree executable, Python reads its ``CMakeCache.txt`` and adds the matching compiler/Qt runtime and Qt plugin directory to the child process. This makes IDE launches work even when MSYS2 is absent from the IDE's ``PATH``. If the native process still exits before opening a window, ``launch_viewer`` raises an actionable exception instead of failing silently.

See the `WaveFEMViewer README <../WaveFEMViewer/README.rst>`_ for complete installation, uninstallation, command-line, file-picker, and GUI usage instructions. Its lazy native reader indexes a sweep first and loads only the selected frequency's large field arrays. The viewer reads HDF5 only; no FEM solve is started. Its 2D plots put ``z`` on the horizontal axis and ``x`` on the vertical axis and render the stored dielectric, PEC, PMC, wave-port, and PML scene styles.

The installed inspector remains headless by default. Add ``--gui`` to open a file or directory; with no path it opens the current directory. The runnable ``examples/inspect_h5.py`` convenience script opens the current directory in the GUI when it is invoked without arguments:

.. code-block:: bash

   wavefem-inspect-h5 result.h5
   wavefem-inspect-h5 --gui
   python examples/inspect_h5.py
   python examples/inspect_h5.py --gui results

Power and fields
----------------

For passive problems the reported balance is

.. math::

   P_{\mathrm{in}}\simeq P_R+P_T+P_{\mathrm{rad}}+P_{\mathrm{abs}}.

``result.power_balance_error`` is a dimensionless residual. An independent Poynting-flux residual over the non-PML monitor box is kept in ``result.solve_info``. Use ``result.check()`` rather than treating a single S-parameter as sufficient evidence of numerical quality.

Sampled fields are available without rerunning:

.. code-block:: python

   ey = result.field("Ey", quantity="complex")
   e_abs = result.field("E", quantity="norm")
   h_abs = result.field("H", quantity="norm")
   result.plot_field("Ey", quantity="real", part="total")
   result.visualize("Ey", quantity="real", part="total")  # plots and shows
   result.visualize("Ey", show=False)  # plots without calling pyplot.show()

Lower-level meshes, matrices, modes, incident fields, projectors, and mixed FEM systems remain importable from their respective ``wavefem`` modules.

For a callback-defined device, provide the lead modes explicitly instead of asking WaveFEM to infer discontinuous cross-section layers from samples:

.. code-block:: python

   import numpy as np

   sim = wf.Scattering2D.from_material_function(
       frequency=193.414489e12,
       angle=0.0,
       domain=((0.0, 1.0e-6), (-3.0e-6, 3.0e-6)),
       eps_r=lambda x, z: np.where(np.abs(z) <= 0.3e-6, 1.002, 1.0)
       + 0.0 * np.asarray(x),
       eps_background=lambda x: np.ones_like(x, dtype=np.complex128),
       transverse_boundary="pec",
   )
   sim.add_pml(z=0.8e-6)
   sim.set_monitors(left=-1.0e-6, right=1.0e-6)
   sim.mesh(wavelength_elements=8)

   cross_section = wf.CrossSection(
       (0.0, 1.0e-6), background=wf.Material(), boundary="pec"
   )
   modes = wf.ModeSolver(cross_section, frequency=193.414489e12).solve(
       num_modes=1, neff_guess=1.0
   )
   sim.set_modes(modes)
   sim.set_incident_mode(0)
   result = sim.run(h5_path="callback_result.h5")

The injected ``ModeSet`` is checked against the simulation frequency, ``ky``, and transverse span before it can be launched. Callback-defined simulations currently require ``angle=0`` or a directly prescribed compatibility ``ky``; nonzero angle resolution needs an integrated geometry-backed lead. The caller must also ensure a lossless z-invariant callback background, positive-z modal roots, and compact contrast bracketed by the explicit monitors; see the callback section of the `API reference <API_REFERENCE.rst>`_ for the complete contract.

Validation and current limits
-----------------------------

Automated tests cover the modified curl, Hermiticity where applicable, manufactured mixed-space convergence, full-vector mode convergence and power normalization, nonzero-``ky`` symmetry, PML tensor algebra, transverse mode-PML convergence, source support/sign, synthetic and physical modal projection, zero scattering in a uniform guide, weak-contrast field/amplitude/power scaling, physical z-PML and mesh refinement, open-guide zero-scattering, reciprocity, compact loss, inserted-PEC essential traces, combined oblique slot/plate scattering, and independent power accounting.

Current explicit limits are:

* scalar isotropic physical materials; PML tensors are internal diagonals;

* triangular meshes with first-order Nedelec/P1 fields;

* passive reciprocal linear media and one incident mode per solve; compact material loss is supported, while active media and lossy uniform leads are rejected by the integrated power-accounting path;

* permittivity perturbations only (``mu_actual`` must equal ``mu_background``);

* ideal actual-only PEC insertions are finite constant-x plates; arbitrary curved PEC objects and finite-conductivity sheets are not implemented;

* left incidence in the integrated ``Scattering2D.solve()`` path;

* sparse direct linear solves;

* explicit PEC transverse truncation, or symmetric transverse PMLs terminated by PEC; PMC scattering truncation is not yet implemented;

* automatic lead-mode construction requires geometry-backed z-invariant background layers; callback devices use an explicitly injected ``ModeSet``;

* sweep points solve and order their modes independently; automatic cross-frequency branch tracking through modal crossings or cutoffs is not yet implemented, so multimode sweeps should verify saved modal profiles and use sufficiently close frequency steps and a physically informed ``neff_guess``;

* open-transverse integrated port results retain only bound modes above the exterior light line; standalone ``ModeSolver`` results may also contain PML-discretized radiation candidates for research use.

Always establish mesh and PML convergence for a new physical structure. A single mesh is a computation, not a validation.

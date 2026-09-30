FEM Waveguide Scattering Solver
===============================

.. contents:: On this page
   :local:
   :depth: 2

First example
-------------

With the packages installed as described in the `project setup <../../../../README.md>`_,
run this example from the repository root::

    python examples/fem/waveguide_scattering/uniform_waveguide_2d.py

The uniform guide should have effective index near 1, reflection near 0, and transmission near 1. The FEM Waveguide Scattering Viewer opens the result.

This example uses the `native scattering viewer <../../../../apps/fem_waveguide_scattering_viewer/README.rst>`_ when calling ``show()``.
For numerical runs without windows, omit ``show()`` or the plotting call;
FEM results also offer ``plot()`` for static figures.

Open the `first example <../../../../examples/fem/waveguide_scattering/uniform_waveguide_2d.py>`_ to change
geometry and controls. The `example index <../../../../examples/fem/waveguide_scattering/README.rst>`_
provides a learning order and more physical problems. Scripts run from any working
directory once the packages are installed in the same Python environment.

Scripts that save results write to
``outputs/examples/fem/waveguide_scattering/<example>/`` relative to the checkout.
The first example may only display results; see its code for explicit save calls.

Working with the solver
-----------------------

1. Construct the solver with keyword arguments and physical lengths in metres.
2. Add geometry, material properties, and boundary or excitation conditions.
3. Call ``mesh(...)`` to choose spatial and element resolution.
4. Call ``solve(...)`` to obtain a typed result; ``max_refinements=0`` uses one fixed mesh.
5. Inspect diagnostics, then call ``show()``, ``plot()``, or ``save(path)`` explicitly.

Before the scattering solve, use ``solve_modes()`` and ``set_incident_mode(0)``
to configure the first lead mode. The mesh is 2D; the fields are 2.5D full-vector.

Part of **FDFD**, version 1.0.0.

A **2.5D full-vector** scattered-field solver on a two-dimensional x/z mesh.
The invariant-direction factor is ``exp(-i*ky*y)``.

Workflow
--------

.. code-block:: python

    from cem_common import materials
    from fem_waveguide_scattering import WaveguideScatteringSolver2D, load_result

    solver = WaveguideScatteringSolver2D(frequency=10e9, x_range=.04,
        z_range=(-.1, .1), boundary=materials.PEC)
    solver.add_pml(thickness=.03, direction="z")
    solver.mesh(max_element_size=.005)
    solver.solve_modes(num_modes=1, neff_guess=.9, max_refinements=0)
    solver.set_incident_mode(0)
    result = solver.solve(max_refinements=0)
    result.save("outputs/scattering.h5")
    loaded = load_result("outputs/scattering.h5")
    print(loaded.S11, loaded.S21)
    loaded.show()

``solve()`` automatically meshes when needed and never saves or opens a window.
The adaptive defaults are two refinements and a relative tolerance of 0.05.
The example uses a fixed mesh for reproducibility. Geometry edits invalidate
``mesh_data`` and ``result``; an automatic rebuild reuses explicit mesh settings.

``show()`` uses the native viewer included in the complete Windows wheel (or
built separately for source installations). Launch failures report
the executable discovery setting; saving and loading work without the viewer.

``slot = solver.add_slot(geometry=sheet, z_range=(z0, z1))`` cuts an opening
in a background PEC sheet. Use ``solver.remove(geometry=slot)`` to close it.
Remove dependent slots before removing their parent sheet.

Electromagnetic fields use ``exp(+i*omega*t)`` and guided propagation
``exp(-i*beta*z)``. Passive constitutive values have nonpositive imaginary
parts; forward attenuation is ``-Im(beta)``.

Persistence and API
-------------------

Results use the ``cem-fem-results`` HDF5 envelope, schema ``1.0``. Old archives
are rejected. Loading supports inspection, plotting, and saving, not solver
restart or Python callback restoration. Mode indices are zero-based.

See `API_REFERENCE.rst <API_REFERENCE.rst>`_ for supported configuration,
results, units, defaults, and exceptions. Run bundled examples from the repository
root with ``python examples/fem/waveguide_scattering/<example>.py``.

See the `family example index <../../../../examples/fem/waveguide_scattering/README.rst>`_
for learning order, output locations, and viewer requirements.

Matched ports and closed-contour far fields
-----------------------------------------------

Configure ``solver.set_matched_ports()`` to replace the two longitudinal PEC
terminations with modal electric-to-magnetic maps. Omit longitudinal z PML in
this configuration; transverse x PML is still required for an open guide.
Every retained mode is matched. Unrepresented boundary traces use a local
normal-incidence impedance approximation, so port distance and mode count
remain convergence controls. Longitudinal PML remains an alternative, including
for the far-field calculation.

Call ``solver.set_nf2ff_contour(...)`` before meshing. This constructs a
**closed rectangle with all four sides**, including the sections that cross
the continuing waveguide. The contour must enclose every actual/background
perturbation and lie before the PML. Its default x bounds are the physical
x/PML interfaces and its default z bounds are the two modal monitors.

For example, after configuring a slab or grounded-slot geometry::

    import numpy as np

    solver.set_matched_ports()
    solver.set_nf2ff_contour(x_range=(-0.012, 0.012),
                            z_range=(-0.015, 0.015))
    solver.mesh(max_element_size=0.001, element_order=2)
    solver.solve_modes(num_modes=1, neff_guess=1.8,
                       num_elements=384, max_refinements=0)
    solver.set_incident_mode(0)
    result = solver.solve(max_refinements=0)
    theta = (np.arange(360) + 0.5) * 2 * np.pi / 360
    far = result.far_field(theta)
    print(far.integrated_power())  # radiative power, W/m
    print(far.directivity.max(), far.gain.max(), far.realized_gain.max())
    result.save("outputs/scattering-with-contour.h5")

The reciprocal test fields solve the infinite x-stratified background, including
PEC sheets. Thus the transform handles a guide crossing the closed contour
without deleting contour segments or treating the guide as homogeneous space.
Backgrounds must be isotropic and piecewise constant in x; observation
half-spaces must be lossless with positive epsilon and mu. Callback backgrounds
require an explicit ``LayeredExterior``. Material samples outside the contour
are checked against that exterior on the FEM quadrature. With callbacks this is
a numerical check, not proof of their values between samples.

Angles are radians from +x toward +z, distinct from the solver's incidence
``angle`` in degrees. Exact grazing directions are excluded; the midpoint
angular grid above avoids them. This is a 2.5D cylindrical far field for the
prescribed ``ky``, not a finite antenna's spherical 3D pattern. The complex
``amplitude`` satisfies ``E ~ amplitude*exp(-i*q*rho-i*ky*y)/sqrt(rho)``.
``power_density`` is W/m/radian and includes the transverse ``q/k`` factor.
``directivity``, ``gain``, and ``realized_gain`` are 2D isotropic-reference
ratios: ``2*pi*power_density`` divided by integrated NF2FF radiated power,
accepted power (incident minus reflected), or incident power, respectively.
Use ``10*log10(value)`` for dB relative to a uniform line radiator; these are
not spherical 3D dBi values. Zero radiation leaves directivity undefined (NaN).
The native viewer's Radiation tab shows all four quantities in polar or
Cartesian form, with linear and dB scales, from the saved archive.
Integration excludes guided power, while ``result.nf2ff.outward_power`` includes
the guided contribution of the scattered field. The existing ``radiated_power``
remains an independent flux/modal estimate; compare it with the integrated
pattern as a convergence diagnostic rather than forcing agreement.

Archives preserve all contour samples, quadrature, and exterior layers, so
``load_result(...).far_field(new_angles)`` needs no FEM solve. Test convergence
by refining the mesh and quadrature, moving the contour, moving the ports,
and comparing modal-port termination with z PML.

The layered reciprocal approach follows the principle described by
`Yang, Hugonin, and Lalanne, Near-to-Far Field Transformations for Radiative and
Guided Waves <https://arxiv.org/abs/1510.06344>`_; the implementation uses the
project's 2.5D geometry and positive-time phasor convention.

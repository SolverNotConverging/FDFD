Computational Electromagnetics documentation
============================================

Start with the `project setup <../README.md>`_ and `runnable examples <../examples/README.rst>`_.
Python packages are independently installable; their detailed documentation lives here.

Solver families
---------------

* fdfd waveguide modes: `guide <solvers/fdfd/waveguide_modes/guide.rst>`_, `API <solvers/fdfd/waveguide_modes/API_REFERENCE.rst>`_, `examples <../examples/fdfd/waveguide_modes/README.rst>`_
* fdfd periodic modes: `guide <solvers/fdfd/periodic_modes/guide.rst>`_, `API <solvers/fdfd/periodic_modes/API_REFERENCE.rst>`_, `examples <../examples/fdfd/periodic_modes/README.rst>`_
* fdfd band structure: `guide <solvers/fdfd/band_structure/guide.rst>`_, `API <solvers/fdfd/band_structure/API_REFERENCE.rst>`_, `examples <../examples/fdfd/band_structure/README.rst>`_
* fdfd scattering: `guide <solvers/fdfd/scattering/guide.rst>`_, `API <solvers/fdfd/scattering/API_REFERENCE.rst>`_, `examples <../examples/fdfd/scattering/README.rst>`_
* fem waveguide modes: `guide <solvers/fem/waveguide_modes/guide.rst>`_, `API <solvers/fem/waveguide_modes/API_REFERENCE.rst>`_, `examples <../examples/fem/waveguide_modes/README.rst>`_
* fem periodic modes: `guide <solvers/fem/periodic_modes/guide.rst>`_, `API <solvers/fem/periodic_modes/API_REFERENCE.rst>`_, `examples <../examples/fem/periodic_modes/README.rst>`_
* fem waveguide scattering: `guide <solvers/fem/waveguide_scattering/guide.rst>`_, `API <solvers/fem/waveguide_scattering/API_REFERENCE.rst>`_, `examples <../examples/fem/waveguide_scattering/README.rst>`_
* fem electrostatics: `guide <solvers/fem/electrostatics/guide.rst>`_, `API <solvers/fem/electrostatics/API_REFERENCE.rst>`_, `examples <../examples/fem/electrostatics/README.rst>`_

Shared libraries
----------------

* cem_common: `guide <libraries/cem_common/guide.rst>`_, `API <libraries/cem_common/API_REFERENCE.rst>`_
* fem_adaptivity: `guide <libraries/fem_adaptivity/guide.rst>`_, `API <libraries/fem_adaptivity/API_REFERENCE.rst>`_
* periodic_eigensolver: `guide <libraries/periodic_eigensolver/guide.rst>`_, `API <libraries/periodic_eigensolver/API_REFERENCE.rst>`_

Contributors
------------

* `Release history <development/release_history.md>`_
* `Curated FEM public API inventory <public_api.json>`_
* `Curated FDFD public API inventory <fdfd_public_api.json>`_

Solver formulation and development notes are included after the first example in
each family's ``guide.rst``.
Native application build and usage guides remain alongside their applications:

* `FEM Periodic Mode Viewer <../apps/fem_periodic_mode_viewer/README.rst>`_
* `FEM Waveguide Scattering Viewer <../apps/fem_waveguide_scattering_viewer/README.rst>`_
* `Transmission Line Calculator <../apps/transmission_line_calculator/README.rst>`_

Run ``python scripts/check_documentation.py`` from the repository root to validate
RST documents and local links. API generators write into this central tree.

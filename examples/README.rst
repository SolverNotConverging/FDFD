Computational Electromagnetics examples
=======================================

Follow the `project setup and quick start <../README.md>`_ before running examples.
Every script also runs from its own directory. Imports use installed packages,
without modifying Python's search path. Importing a script does not run a solve.

Choose a solver family
----------------------

Each family index lists a recommended order, purpose, runtime expectations, and
viewer requirements:

* `FDFD waveguide modes <fdfd/waveguide_modes/README.rst>`_
* `FDFD periodic modes <fdfd/periodic_modes/README.rst>`_
* `FDFD band structure <fdfd/band_structure/README.rst>`_
* `FDFD scattering <fdfd/scattering/README.rst>`_
* `FEM waveguide modes <fem/waveguide_modes/README.rst>`_
* `FEM periodic modes <fem/periodic_modes/README.rst>`_
* `FEM waveguide scattering <fem/waveguide_scattering/README.rst>`_
* `FEM electrostatics <fem/electrostatics/README.rst>`_

The shared `periodic eigensolver example <libraries/periodic_eigensolver/README.rst>`_
demonstrates the matrix API independently of a physical solver.

Naming and results
------------------

Runnable solver examples use ``<physical_problem>_<dimension>[_<feature>].py``.
Names use lowercase words separated by underscores. Dimensions describe the mesh;
waveguide scattering is 2.5D full-vector physics on a ``2d`` mesh. The enclosing
directories identify the method and solver family. Multi-dimension demonstrations
are separate scripts. Learning order belongs in the family index, not filenames.

Scripts that save files use
``outputs/examples/<method>/<family>/<example>/`` relative to the repository root,
independent of the current working directory. These outputs are ignored by Git.
Examples that only display results do not create output files.

Supporting scripts live in each family's ``postprocessing/`` directory and use
action names such as ``plot_dispersion_2d.py``. They accept an input path; without
one they use the corresponding example's output. Generated plots are placed next
to their input data. Run the producer example first; no numerical datasets are
bundled with this tutorial collection.

Validation
----------

Run every solver tutorial with viewer windows suppressed::

    python scripts/qualify_examples.py

This uses installed packages in isolated Python processes. Numerical settings
match the interactive scripts. Logs are written under
``outputs/example-qualification/``; saved numerical results retain the standard
``outputs/examples/`` locations. Full FDFD dispersion and band-structure studies
take longer than the introductory cases.

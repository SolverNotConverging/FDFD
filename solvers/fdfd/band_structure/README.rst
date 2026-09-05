FDFD Band Structure Solver
==========================

Part of **Computational Electromagnetics**, version 1.0.0.

The ``fdfd_band_structure`` package exports BandStructureSolver2D.
This release reorganizes imports and packaging while preserving the FDFD
numerical algorithms and their existing solver-specific workflow. The uniform
``mesh / solve / show`` interface applies to the FEM packages.

Install from the repository root::

    conda env create -f environment.yml
    conda activate cem
    python scripts/install_python.py

This family is independently installable from its wheel. Bundled examples are
in ``examples/`` and can be run directly with the installed package.
See `API_REFERENCE.rst <API_REFERENCE.rst>`_ for the supported user methods.

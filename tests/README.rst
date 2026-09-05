FDFD tests
==========

Run all Python tests from the repository root::

    python -m pytest

Tests are organized by method and family under ``fdfd/`` and ``fem/``. Shared
library tests are under ``libraries/``; cross-family lifecycle, archive, convention,
and integration checks remain directly in this directory. For example::

    python -m pytest tests/fem/electrostatics
    python -m pytest tests/fdfd/waveguide_modes
    python -m pytest tests/libraries/periodic_eigensolver

Use the installed packages and dependencies described in the
`root README <../README.md>`_. The native eigensolver tests include additional cases
when its compiled extension is available. Native C++ application tests remain
registered with CTest in their application directories.

`Analytical benchmarks <../benchmarks/README.md>`_ produce theory-versus-solver
tables and plots separately from the automated test suite.

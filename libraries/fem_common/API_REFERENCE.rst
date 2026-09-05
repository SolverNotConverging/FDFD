Shared FEM contracts
====================

``FEMError`` is the common base for errors raised by FEM applications.
``ConfigurationError``, ``GeometryError``, ``MeshError``, ``SolverError``,
``NoResultError``, ``PersistenceError``, and ``ViewerError`` identify the operation
that failed. ``MeshSnapshot`` exposes physical ``coordinates``, ``elements``,
``axes``, ``info``, and ``metadata`` on saved and loaded results.

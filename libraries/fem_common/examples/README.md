Applications can catch `fem_common.FEMError` across all FEM families. Numerical
examples using these contracts are in `solvers/fem/*/examples/`; each returns a
typed result exposing `mesh_data`, `metadata`, `solve_info`, `plot`, `show`, and
`save`. Import `load_result` from the corresponding family to inspect an archive.

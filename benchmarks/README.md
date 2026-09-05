# Computational Electromagnetics benchmarks

Run these with the installed project packages in the `cem` environment:

```console
python benchmarks/periodic_eigensolver/benchmark_mgs.py --enforce
python benchmarks/periodic_eigensolver/benchmark_end_to_end.py --enforce
```

These performance gates require an otherwise idle machine and the compiled
periodic eigensolver extension. Numerical regression belongs in `tests/` and
package tests. Store generated benchmark reports under `outputs/`.

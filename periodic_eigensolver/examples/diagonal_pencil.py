"""Solve a small generalized pencil using the supported library API."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "cem_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
from scipy.sparse import diags, eye
from periodic_eigensolver import solve_generalized


def main():
    # A x = lambda B x has exact eigenvalues 1, 2, ..., 20 here.
    # Request the two eigenvalues nearest the shift, so expect 3 and 4.
    result = solve_generalized(
        diags(np.arange(1., 21.)), eye(20), sigma=3.1, num_modes=2,
    )
    print("Eigenvalues (expected 3 and 4):", result.eigenvalues)
    print("Original-pencil residuals:", result.residuals)
    return result


if __name__ == '__main__':
    main()

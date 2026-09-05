"""Solve a small generalized pencil using the supported library API."""
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

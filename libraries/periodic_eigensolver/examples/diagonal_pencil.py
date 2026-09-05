"""Solve a small generalized pencil using the supported library API."""
import numpy as np
from scipy.sparse import diags,eye
from periodic_eigensolver import solve_generalized


def main():
    result=solve_generalized(diags(np.arange(1.,21.)),eye(20),sigma=3.1,num_modes=2)
    print(result.eigenvalues)
    print(result.physical_residuals)


if __name__=='__main__':main()

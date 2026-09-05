import math


def calculate_sigma_max(n, R, eta, d):
    """
    Calculate maximum sigma for PML.

    This returns the positive conductivity magnitude.  With the repository's
    exp(+j*omega*t) convention, the electromagnetic solvers apply it through
    the negative-imaginary stretch ``1 - j*sigma/(omega*epsilon_0)``.

    Parameters:
    - n (int): Polynomial order of sigma profile
    - R (float): Desired reflection coefficient (e.g., 1e-6)
    - eta (float): Wave impedance (e.g., 377 for vacuum)
    - d (float): Thickness of the PML (in meters)

    Returns:
    - sigma_max (float): Maximum conductivity sigma
    """
    ln_R = math.log(R)  # natural log
    sigma_max = -((n + 1) * ln_R) / (2 * eta * d)
    return sigma_max


def main():
    # Example usage:
    n = 3  # Polynomial order
    R = 1e-10  # Desired reflection coefficient
    eta = 377  # Wave impedance (Ohms), for EM in vacuum
    d = 10e-3  # PML thickness (meters)

    sigma_max = calculate_sigma_max(n, R, eta, d)
    print(f"Maximum sigma: {sigma_max:.6f} S/m")


if __name__ == "__main__":
    main()

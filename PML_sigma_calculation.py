import math


def calculate_sigma_max(n, R, eta, d):
    """
    Calculate maximum sigma for PML.

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


# Example usage:
n = 2  # Polynomial order
R = 1e-50  # Desired reflection coefficient
eta = 377/math.sqrt(10.2)  # Wave impedance (Ohms), for EM in vacuum
d = 10e-3  # PML thickness (meters)

sigma_max = calculate_sigma_max(n, R, eta, d)
print(f"Maximum sigma: {sigma_max:.6f} S/m")

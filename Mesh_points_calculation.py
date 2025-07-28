import numpy as np


def suggest_mesh_points(ranges, epsilon_max, mu_max, frequency):
    """
    Suggest mesh points for FDFD simulation in 1D or 2D.

    Parameters:
        ranges (float or list): Range(s) in meter(s).
                                If 1D: a single float (x_range).
                                If 2D: a list [x_range, y_range].
        epsilon_max (float): Maximum relative permittivity.
        mu_max (float): Maximum relative permeability.
        frequency (float): Operating frequency (Hz).

    Returns:
        mesh_points (int or list): Suggested number of points.
                                   If 1D: a single int.
                                   If 2D: a list [Nx, Ny].
    """
    # Ensure ranges is a list for uniform processing
    if isinstance(ranges, (int, float)):
        ranges = [ranges]  # Convert to list for 1D

    # Speed of light in vacuum
    c0 = 3e8  # m/s

    # Compute minimum wavelength in the material
    lambda_min = c0 / (frequency * np.sqrt(epsilon_max * mu_max))

    # Desired grid resolution (10 points per wavelength for accuracy)
    d_min = lambda_min / 10

    # Calculate the number of points for each range
    mesh_points = [int(np.ceil(r / d_min)) for r in ranges]

    # Return an integer for 1D, or a list for 2D
    return mesh_points if len(mesh_points) > 1 else mesh_points[0]


# Example input parameters
x_range = 20e-3  # meters (for 1D)
y_range = 10e-3  # meters (only needed for 2D)
epsilon_max = 10.2  # relative permittivity
mu_max = 1.0  # relative permeability
frequency = 100e9  # Hz (1 GHz)

# Example usage for 1D
mesh_1D = suggest_mesh_points(x_range, epsilon_max, mu_max, frequency)
print(f"Suggested mesh points for 1D: {mesh_1D}")

# Example usage for 2D
mesh_2D = suggest_mesh_points([x_range, y_range], epsilon_max, mu_max, frequency)
print(f"Suggested mesh points for 2D: {mesh_2D}")
